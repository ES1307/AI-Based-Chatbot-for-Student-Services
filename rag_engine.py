"""Small, explainable RAG pipeline used by the Streamlit interface."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable
import json
import re

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

GENERATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RETRIEVAL_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "course", "content", "describe", "do",
    "explain", "for", "from", "give", "how", "i", "in", "is", "it", "list", "me", "of", "on",
    "or", "please", "show", "subject", "syllabus", "the", "to", "unit", "units", "what", "when",
    "where", "with", "you", "your",
    "answer", "available", "document", "documents", "information", "student", "students", "university",
    "uploaded", "provide", "please", "tell", "about", "details", "there", "these", "this",
}


@dataclass
class SourceChunk:
    text: str
    source_name: str
    chunk_id: int
    score: float = 0.0


@dataclass
class RAGResult:
    answer: str
    key_points: list[str]
    next_step: str
    sources: list[SourceChunk]
    used_generator: bool
    is_grounded: bool


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        # Preserve page boundaries; they often correspond to a new handbook section.
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported document type: {path.suffix}")


def split_text(text: str, chunk_size: int = 300, overlap: int = 55) -> list[str]:
    """Chunk on sentence boundaries so policies never begin halfway through a rule."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Some manuals place an all-caps heading and its content on one line
        # (for example, "DRESS CODE For Boys: ..."). Split that boundary too.
        inline_heading = re.match(r"^([A-Z][A-Z &()/,-]{3,}(?:\s+For\s+[A-Za-z]+)?\s*:)(.+)$", line)
        if inline_heading:
            if lines and lines[-1] != "":
                lines.append("")
            lines.extend([inline_heading.group(1), inline_heading.group(2).strip()])
            continue
        is_numbered_heading = bool(re.match(r"^\d{1,2}(?:\.\d+)+\s+[A-Z]", line))
        is_uppercase_heading = bool(re.match(r"^[A-Z][A-Z &()/,-]{5,}(?::|\s|$)", line))
        if lines and (is_numbered_heading or is_uppercase_heading) and lines[-1] != "":
            lines.append("")
        lines.append(line)
    text = "\n".join(lines)
    # Protect common PDF abbreviations such as "Clause No. 16" before splitting.
    text = re.sub(r"\b(No|Rs|Dr|Mr|Ms)\.", r"\1§", text)
    units = re.split(r"(?<=[.!?])\s+(?=[A-Z(0-9])|\n\s*\n+", text)
    units = [" ".join(unit.replace("§", ".").split()) for unit in units if unit.strip()]
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        unit_words = len(unit.split())
        # Extremely long PDF-table lines are split only as a last resort.
        if unit_words > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            words = unit.split()
            for start in range(0, len(words), chunk_size - overlap):
                chunks.append(" ".join(words[start:start + chunk_size]))
            continue
        if current and current_words + unit_words > chunk_size:
            chunks.append(" ".join(current))
            overlap_words = " ".join(current).split()[-overlap:]
            current = [" ".join(overlap_words)] if overlap_words else []
            current_words = len(overlap_words)
        current.append(unit)
        current_words += unit_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def load_documents(paths: Iterable[Path]) -> list[SourceChunk]:
    documents: list[SourceChunk] = []
    for path in paths:
        for chunk_id, text in enumerate(split_text(read_document(path))):
            documents.append(SourceChunk(text=text, source_name=path.name, chunk_id=chunk_id))
    if not documents:
        raise ValueError("No readable text was found in the selected documents.")
    return documents


class RAGEngine:
    """Semantic retrieval with a deliberately transparent, local-first RAG design."""

    def __init__(self, chunks: list[SourceChunk], embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self.chunks = chunks
        self.embedder = SentenceTransformer(embedding_model)
        self.chunk_embeddings = self.embedder.encode([item.text for item in chunks], normalize_embeddings=True)
        self.document_terms = _meaningful_terms(" ".join(f"{item.source_name} {item.text}" for item in chunks))

    def retrieve(self, question: str, top_k: int = 6) -> list[SourceChunk]:
        """Rank by meaning and subject-specific words to avoid cross-subject matches."""
        question_embedding = self.embedder.encode([question], normalize_embeddings=True)[0]
        semantic_scores = np.dot(self.chunk_embeddings, question_embedding)
        question_terms = _meaningful_terms(question)
        lexical_scores: list[float] = []
        for chunk in self.chunks:
            searchable_text = f"{chunk.source_name} {chunk.text}"
            chunk_terms = _meaningful_terms(searchable_text)
            overlap = len(question_terms & chunk_terms)
            lexical_scores.append(overlap / max(len(question_terms), 1))

        # Semantic search finds related policy/syllabus content; the lexical boost
        # makes a named subject (for example, English) outrank another subject with
        # similarly worded "Unit" headings (for example, Maths).
        scores = semantic_scores + 0.40 * np.array(lexical_scores)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            SourceChunk(
                text=self.chunks[index].text,
                source_name=self.chunks[index].source_name,
                chunk_id=self.chunks[index].chunk_id,
                score=float(scores[index]),
            )
            for index in top_indices
        ]

    def answer(self, question: str) -> RAGResult:
        if not self._has_question_terms_in_documents(question):
            return self._not_found_result()

        # Keep a wider evidence set for broad questions such as facilities, hostel,
        # scholarships, or campus services.
        sources = self.retrieve(question, top_k=8)
        if not self._has_document_evidence(question, sources):
            return self._not_found_result()

        extracted_facts = self._extract_key_points(question, sources, limit=6)
        if not extracted_facts:
            return self._not_found_result()

        # Qwen receives the focused retrieved passages and writes the final response.
        # It is not asked to summarize or follow a forced response template.
        prompt = self._build_response_prompt(question, sources)
        generated = self._generate_response(prompt)
        if generated and self._is_generated_response_grounded(generated, sources):
            return RAGResult(
                answer=generated,
                key_points=[],
                next_step="Review the cited source passage below for the complete official wording.",
                sources=sources,
                used_generator=True,
                is_grounded=True,
            )

        fallback = " ".join(extracted_facts)
        return RAGResult(
            answer=fallback,
            key_points=[],
            next_step="Check the source passages below or contact the relevant university office for confirmation.",
            sources=sources,
            used_generator=False,
            is_grounded=True,
        )

    def _has_document_evidence(self, question: str, sources: list[SourceChunk]) -> bool:
        """Reject out-of-document questions before the generative model can improvise."""
        question_terms = _meaningful_terms(question)
        if not self._has_question_terms_in_documents(question):
            return False

        source_terms = _meaningful_terms(" ".join(f"{source.source_name} {source.text}" for source in sources))
        source_matches = question_terms & source_terms
        if not source_matches:
            return False

        # The evidence must appear in one retrieved chunk, not merely somewhere in
        # another page of the uploaded file. This prevents an LLM from using an
        # unrelated but semantically similar section as permission to answer.
        match_ratios = []
        for source in sources:
            chunk_terms = _meaningful_terms(f"{source.source_name} {source.text}")
            match_ratios.append(len(question_terms & chunk_terms) / len(question_terms))
        best_match_ratio = max(match_ratios, default=0.0)
        best_retrieval_score = sources[0].score if sources else 0.0
        minimum_ratio = 0.50 if len(question_terms) <= 2 else 0.34
        return bool(source_matches) and best_match_ratio >= minimum_ratio and best_retrieval_score >= 0.32

    def _is_generated_response_grounded(
        self, response: str, sources: list[SourceChunk]
    ) -> bool:
        """Reject claims that cannot be tied back to the retrieved source vocabulary.

        This intentionally favours a safe refusal over a polished but unsupported
        answer. It is a lightweight post-generation guardrail for a local model.
        """
        source_text = " ".join(source.text for source in sources)
        source_terms = _meaningful_terms(source_text)
        response_terms = _meaningful_terms(response)
        if not response_terms:
            return False
        # A polished summary can use different wording from the PDF. Accept that
        # when it is semantically close to the retrieved evidence and still keeps
        # some factual source vocabulary; reject unrelated model knowledge.
        lexical_support = len(response_terms & source_terms) / len(response_terms)
        vectors = self.embedder.encode([response, source_text], normalize_embeddings=True)
        semantic_support = float(np.dot(vectors[0], vectors[1]))
        return lexical_support >= 0.25 and semantic_support >= 0.52

    @staticmethod
    def _build_response_prompt(question: str, sources: list[SourceChunk]) -> str:
        context = "\n\n".join(f"[Retrieved passage {index + 1}]\n{source.text}" for index, source in enumerate(sources))
        return f"""You are a careful university-document assistant.
Write a complete, polished response to the student's question using ONLY the retrieved passages below.
You may paraphrase and organise the information naturally, but never add a policy, requirement, date, amount,
office, eligibility condition, or conclusion that is not supported by the passages.
Do not copy raw PDF headings, page numbers, broken table text, or incomplete sentence fragments into the answer.
If the passages provide only part of the answer, say clearly what they state and do not fill gaps from general knowledge.

Question: {question}

Retrieved passages:
{context}

Respond directly to the question in the level of detail the retrieved passages support. Do not stop mid-sentence, mid-rule, or mid-list. Do not pad with general knowledge.
Choose the clearest Markdown format yourself: natural paragraphs, a short list, or a combination. Use bullets only when they improve readability.
Return only the answer, with no preamble about being an AI."""

    def _has_question_terms_in_documents(self, question: str) -> bool:
        question_terms = _meaningful_terms(question)
        return bool(question_terms and question_terms & self.document_terms)

    @staticmethod
    def _not_found_result() -> RAGResult:
        return RAGResult(
            answer="Sorry, I could not find information about that in the uploaded documents.",
            key_points=[],
            next_step="Try using the exact subject, topic, policy name, or unit title from the uploaded file.",
            sources=[],
            used_generator=False,
            is_grounded=False,
        )

    @staticmethod
    def _extract_key_points(question: str, sources: list[SourceChunk], limit: int = 3) -> list[str]:
        """Create a short extractive answer when the optional generator is unavailable."""
        query_words = _meaningful_terms(question)
        candidates: list[tuple[float, int, str]] = []
        for source_index, source in enumerate(sources):
            sentences = re.split(r"(?<=[.!?])\s+", source.text)
            for sentence_index, sentence in enumerate(sentences):
                sentence = " ".join(sentence.split())
                if len(sentence) < 35:
                    continue
                words = _meaningful_terms(sentence)
                overlap = len(query_words & words)
                # A high-ranked chunk can contain neighbouring policy sections.
                # A sentence with no question-term match is never a fallback fact.
                if query_words and overlap == 0:
                    continue
                if len(query_words) >= 2 and overlap / len(query_words) < 0.50:
                    continue
                score = overlap * 3 + source.score + (0.15 if sentence_index < 2 else 0)
                candidates.append((score, source_index * 1000 + sentence_index, sentence))
        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
        # For broad questions (for example, "What scholarships are available?"),
        # select facts from different retrieved chunks before selecting a second fact
        # from the same chunk. This avoids returning three versions of one rule.
        selected: list[tuple[float, int, str]] = []
        used_sources: set[int] = set()
        for candidate in ranked:
            source_index = candidate[1] // 1000
            if source_index not in used_sources:
                selected.append(candidate)
                used_sources.add(source_index)
            if len(selected) == limit:
                break
        if len(selected) < limit:
            for candidate in ranked:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) == limit:
                    break
        # Preserve the document's natural order after choosing the most relevant sentences.
        return [_shorten_text(item[2]) for item in sorted(selected, key=lambda item: item[1])]

    @staticmethod
    def _generate_response(prompt: str, max_new_tokens: int = 1200) -> str | None:
        """Generate and validate a polished response from the local instruction model.

        The exception fallback keeps retrieval demonstrable on machines without the model.
        """
        try:
            generator = _load_generator()
            messages = [
                {
                    "role": "system",
                    "content": "You are CampusGuide, a careful university student-support assistant.",
                },
                {"role": "user", "content": prompt},
            ]
            # This is a safety ceiling, not an answer-length instruction. Most complete
            # student answers finish naturally well before it, while CPU response times
            # stay practical for an interactive app.
            output = generator(messages, max_new_tokens=max_new_tokens, do_sample=False)
            generated = output[0]["generated_text"]
            if isinstance(generated, list):
                generated = generated[-1].get("content", "")
            return generated.strip() or None
        except Exception:
            return None


@lru_cache(maxsize=1)
def _load_generator():
    """Load the model once per process rather than once per student question."""
    from transformers import pipeline

    return pipeline("text-generation", model=GENERATION_MODEL, device=-1)


def _parse_structured_answer(text: str) -> dict[str, str | list[str]] | None:
    """Accept only schema-valid JSON; raw model prose is never shown as an answer."""
    text = text.strip()
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group())
            answer = payload.get("answer")
            if isinstance(answer, str) and answer.strip():
                return {"answer": answer.strip(), "key_points": []}
        except json.JSONDecodeError:
            pass

    return None


def _meaningful_terms(text: str) -> set[str]:
    """Terms that identify a subject or topic rather than generic question wording."""
    return {
        term
        for term in re.findall(r"[a-zA-Z]{3,}", text.lower())
        if term not in RETRIEVAL_STOP_WORDS
    }


def _shorten_text(text: str, maximum_characters: int = 240) -> str:
    """Keep source-only fallback points readable when PDF text lacks punctuation."""
    if len(text) <= maximum_characters:
        return text
    return f"{text[:maximum_characters].rsplit(' ', 1)[0]}..."
