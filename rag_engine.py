"""Efficient, source-grounded RAG pipeline for CampusGuide."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log
import os
from pathlib import Path
import re
import time
from typing import Iterable

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit as st

# A CPU-friendly default for Streamlit Community Cloud. Set CAMPUSGUIDE_GENERATION_MODEL
# in Streamlit Secrets to opt into a larger compatible Qwen model.
GENERATION_MODEL = os.getenv("CAMPUSGUIDE_GENERATION_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# 180 words keeps most chunks within MiniLM's 256-token encoder window. The
# generator receives a focused subset, so retrieval quality is preserved.
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 24
RETRIEVAL_CANDIDATES = 6
MAX_CONTEXT_CHUNKS = 4
CONTEXT_WORD_BUDGET = 720
MAX_NEW_TOKENS = 384  # A safety ceiling only; Qwen normally stops at EOS.
_GENERATOR_READY = False

RETRIEVAL_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "course", "content", "describe", "do",
    "explain", "for", "from", "give", "how", "i", "in", "is", "it", "list", "me", "of", "on",
    "or", "please", "show", "subject", "syllabus", "the", "to", "unit", "units", "what", "when",
    "where", "with", "you", "your", "answer", "available", "document", "documents", "information",
    "student", "students", "university", "uploaded", "provide", "tell", "about", "details", "there",
    "these", "this",
}


@dataclass
class PerformanceMetrics:
    """Low-overhead timings displayed only when diagnostics are enabled."""

    timings: dict[str, float] = field(default_factory=dict)
    values: dict[str, int | float | str] = field(default_factory=dict)
    _wall_start: float = field(default_factory=time.perf_counter, repr=False)
    _cpu_start: float = field(default_factory=time.process_time, repr=False)
    _rss_before: float | None = field(default_factory=lambda: _process_rss_mib(), repr=False)

    def record(self, name: str, duration: float) -> None:
        self.timings[name] = round(duration, 4)

    def finish(self, total_name: str = "total_request") -> None:
        wall_time = time.perf_counter() - self._wall_start
        cpu_time = time.process_time() - self._cpu_start
        self.timings[total_name] = round(wall_time, 4)
        self.timings["process_cpu_seconds"] = round(cpu_time, 4)
        if wall_time:
            self.values["average_process_cpu_percent"] = round((cpu_time / wall_time) * 100, 1)
        rss_after = _process_rss_mib()
        if rss_after is not None:
            self.values["rss_mib"] = round(rss_after, 1)
            if self._rss_before is not None:
                self.values["rss_change_mib"] = round(rss_after - self._rss_before, 1)


@dataclass(frozen=True)
class SourceChunk:
    text: str
    source_name: str
    chunk_id: int
    terms: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)
    score: float = 0.0


@dataclass
class RAGResult:
    answer: str
    key_points: list[str]
    next_step: str
    sources: list[SourceChunk]
    used_generator: bool
    is_grounded: bool
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)


def read_document(path: Path) -> str:
    """Read supported documents once, preserving PDF page boundaries."""
    if path.suffix.lower() == ".pdf":
        with path.open("rb") as file_handle:
            return "\n\n".join(page.extract_text() or "" for page in PdfReader(file_handle).pages)
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported document type: {path.suffix}")


def split_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Use retrieval-sized, sentence-aware chunks compatible with MiniLM's context."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = re.match(r"^([A-Z][A-Z &()/,-]{3,}(?:\s+For\s+[A-Za-z]+)?\s*:)(.+)$", line)
        if heading:
            if lines and lines[-1] != "":
                lines.append("")
            lines.extend([heading.group(1), heading.group(2).strip()])
            continue
        numbered = bool(re.match(r"^\d{1,2}(?:\.\d+)+\s+[A-Z]", line))
        uppercase = bool(re.match(r"^[A-Z][A-Z &()/,-]{5,}(?::|\s|$)", line))
        if lines and (numbered or uppercase) and lines[-1] != "":
            lines.append("")
        lines.append(line)

    text = re.sub(r"\b(No|Rs|Dr|Mr|Ms)\.", r"\1§", "\n".join(lines))
    units = re.split(r"(?<=[.!?])\s+(?=[A-Z(0-9])|\n\s*\n+", text)
    units = [" ".join(unit.replace("§", ".").split()) for unit in units if unit.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        words = unit.split()
        if len(words) > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            step = max(chunk_size - overlap, 1)
            chunks.extend(" ".join(words[start:start + chunk_size]) for start in range(0, len(words), step))
            continue
        if current and current_words + len(words) > chunk_size:
            chunks.append(" ".join(current))
            tail = " ".join(current).split()[-overlap:]
            current, current_words = ([" ".join(tail)] if tail else []), len(tail)
        current.append(unit)
        current_words += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def load_documents(paths: Iterable[Path]) -> tuple[list[SourceChunk], PerformanceMetrics]:
    """Extract and chunk a document set only when its cache key changes."""
    metrics = PerformanceMetrics()
    chunks: list[SourceChunk] = []
    extraction_seconds = chunking_seconds = 0.0
    for path in paths:
        started = time.perf_counter()
        document_text = read_document(path)
        extraction_seconds += time.perf_counter() - started
        started = time.perf_counter()
        parts = split_text(document_text)
        chunking_seconds += time.perf_counter() - started
        chunks.extend(
            SourceChunk(text=part, source_name=path.name, chunk_id=index, terms=frozenset(_meaningful_terms(part)))
            for index, part in enumerate(parts)
        )
    if not chunks:
        raise ValueError("No readable text was found in the selected documents.")
    metrics.record("pdf_text_extraction", extraction_seconds)
    metrics.record("chunking", chunking_seconds)
    metrics.values.update(document_count=len({item.source_name for item in chunks}), chunk_count=len(chunks))
    return chunks, metrics


class RAGEngine:
    """Semantic retrieval with a shared encoder and focused local generation."""

    def __init__(self, chunks: list[SourceChunk], index_metrics: PerformanceMetrics | None = None) -> None:
        self.chunks = chunks
        self.index_metrics = index_metrics or PerformanceMetrics()
        started = time.perf_counter()
        self.embedder = _load_embedder()
        self.index_metrics.record("embedding_model_acquisition", time.perf_counter() - started)
        started = time.perf_counter()
        self.chunk_embeddings = self.embedder.encode(
            [item.text for item in chunks], normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        self.index_metrics.record("embedding_generation", time.perf_counter() - started)
        self.chunk_terms = [item.terms or frozenset(_meaningful_terms(item.text)) for item in chunks]
        self.document_terms = frozenset().union(*self.chunk_terms)
        document_frequency: dict[str, int] = {}
        for terms in self.chunk_terms:
            for term in terms:
                document_frequency[term] = document_frequency.get(term, 0) + 1
        chunk_count = len(self.chunks)
        self.term_idf = {
            term: log((chunk_count + 1) / (frequency + 1)) + 1
            for term, frequency in document_frequency.items()
        }
        self.chunk_positions = {
            (chunk.source_name, chunk.chunk_id, chunk.text): index for index, chunk in enumerate(self.chunks)
        }
        self.index_metrics.finish("document_indexing_total")

    def retrieve(self, question: str, top_k: int = RETRIEVAL_CANDIDATES) -> list[SourceChunk]:
        question_embedding = self.embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        semantic_scores = self.chunk_embeddings @ question_embedding
        question_terms = _meaningful_terms(question)
        query_weight = sum(self.term_idf.get(term, 1.0) for term in question_terms) or 1.0
        lexical_scores = np.fromiter(
            (sum(self.term_idf.get(term, 1.0) for term in question_terms & terms) / query_weight for terms in self.chunk_terms),
            dtype=np.float32,
            count=len(self.chunks),
        )
        # Exact terms improve ranking but cannot dominate the semantic match.
        scores = semantic_scores + 0.22 * lexical_scores
        count = min(top_k, len(self.chunks))
        if not count:
            return []
        indices = np.argpartition(scores, -count)[-count:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        return [
            SourceChunk(self.chunks[index].text, self.chunks[index].source_name, self.chunks[index].chunk_id,
                        self.chunk_terms[index], float(scores[index]))
            for index in indices
        ]

    def answer(self, question: str) -> RAGResult:
        metrics = PerformanceMetrics()
        if not self._has_question_terms_in_documents(question):
            return self._not_found_result(metrics)
        started = time.perf_counter()
        candidates = self.retrieve(question)
        metrics.record("retrieval", time.perf_counter() - started)
        if not self._has_document_evidence(question, candidates):
            return self._not_found_result(metrics)

        sources = self._select_context_sources(candidates)
        metrics.values.update(retrieved_chunk_count=len(sources), retrieval_candidate_count=len(candidates))
        fallback = self._extract_key_points(question, sources, limit=6)
        if not fallback:
            return self._not_found_result(metrics)
        started = time.perf_counter()
        prompt = self._build_response_prompt(question, sources)
        metrics.record("prompt_construction", time.perf_counter() - started)
        metrics.values.update(
            prompt_word_estimate=len(prompt.split()),
            prompt_token_estimate=round(len(prompt.split()) * 1.3),
            prompt_character_count=len(prompt),
        )
        generated, generation_metrics = self._generate_response(prompt)
        metrics.timings.update(generation_metrics.timings)
        metrics.values.update(generation_metrics.values)
        if generated and self._is_generated_response_grounded(generated, sources, metrics):
            metrics.finish()
            return RAGResult(generated, [], "Review the cited source passage below for the complete official wording.", sources, True, True, metrics)
        metrics.finish()
        return RAGResult(" ".join(fallback), [], "Check the source passages below or contact the relevant university office for confirmation.", sources, False, True, metrics)

    def _has_question_terms_in_documents(self, question: str) -> bool:
        terms = _meaningful_terms(question)
        return bool(terms and terms & self.document_terms)

    def _has_document_evidence(self, question: str, sources: list[SourceChunk]) -> bool:
        terms = _meaningful_terms(question)
        if not sources or not self._has_question_terms_in_documents(question):
            return False
        minimum_ratio = 0.50 if len(terms) <= 2 else 0.34
        return max((len(terms & source.terms) / len(terms) for source in sources), default=0.0) >= minimum_ratio and sources[0].score >= 0.32

    def _is_generated_response_grounded(self, response: str, sources: list[SourceChunk], metrics: PerformanceMetrics) -> bool:
        response_terms = _meaningful_terms(response)
        if not response_terms:
            return False
        started = time.perf_counter()
        source_terms = frozenset().union(*(source.terms for source in sources))
        lexical_support = len(response_terms & source_terms) / len(response_terms)
        response_vector = self.embedder.encode([response], normalize_embeddings=True, show_progress_bar=False)[0]
        selected_vectors = np.stack(
            [self.chunk_embeddings[self.chunk_positions[(source.source_name, source.chunk_id, source.text)]] for source in sources]
        )
        semantic_support = float(np.max(selected_vectors @ response_vector))
        metrics.record("grounding_check", time.perf_counter() - started)
        metrics.values.update(lexical_grounding=round(lexical_support, 3), semantic_grounding=round(semantic_support, 3))
        return lexical_support >= 0.25 and semantic_support >= 0.52

    def _select_context_sources(self, candidates: list[SourceChunk]) -> list[SourceChunk]:
        """Keep a small, contiguous evidence window for the answer model.

        A policy often continues across a chunk boundary. Neighbours of the best
        match are therefore more useful than several weak, unrelated top-k hits.
        """
        if not candidates:
            return []
        primary = candidates[0]
        primary_position = self.chunk_positions[(primary.source_name, primary.chunk_id, primary.text)]
        nearby: list[SourceChunk] = []
        # Chunks carry an overlap from their predecessor, so the best chunk already
        # contains its opening context. Keep the following chunk for a continuation
        # but never pull a preceding, potentially different PDF section into Qwen.
        for position in (primary_position, primary_position + 1):
            if 0 <= position < len(self.chunks):
                chunk = self.chunks[position]
                if chunk.source_name == primary.source_name:
                    nearby.append(SourceChunk(chunk.text, chunk.source_name, chunk.chunk_id, chunk.terms, primary.score))

        selected: list[SourceChunk] = []
        selected_ids: set[tuple[str, int, str]] = set()
        words_used = 0
        # Only add a non-adjacent hit when it is nearly as relevant as the best
        # passage. This avoids polluting the prompt with a distant mention of the
        # same institution name.
        strong_candidates = [source for source in candidates if source.score >= primary.score * 0.90]
        for source in nearby + strong_candidates:
            identity = (source.source_name, source.chunk_id, source.text)
            source_words = len(source.text.split())
            if identity in selected_ids or (selected and (len(selected) >= MAX_CONTEXT_CHUNKS or words_used + source_words > CONTEXT_WORD_BUDGET)):
                continue
            selected.append(source)
            selected_ids.add(identity)
            words_used += source_words
        return selected or [primary]

    @staticmethod
    def _build_response_prompt(question: str, sources: list[SourceChunk]) -> str:
        context = "\n\n".join(f"[Source {index + 1}]\n{source.text}" for index, source in enumerate(sources))
        return (
            "Answer using only the source passages. Preserve relevant details, paraphrase naturally, and do not invent information. "
            "If the passages are incomplete, state only what they support. Use Markdown only when it improves readability.\n\n"
            f"Question: {question}\n\nSources:\n{context}"
        )

    @staticmethod
    def _extract_key_points(question: str, sources: list[SourceChunk], limit: int = 3) -> list[str]:
        terms = _meaningful_terms(question)
        candidates: list[tuple[float, int, str]] = []
        for source_index, source in enumerate(sources):
            for sentence_index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", source.text)):
                sentence = " ".join(sentence.split())
                sentence_terms = _meaningful_terms(sentence)
                overlap = len(terms & sentence_terms)
                if len(sentence) < 35 or not overlap or (len(terms) >= 2 and overlap / len(terms) < 0.50):
                    continue
                candidates.append((overlap * 3 + source.score, source_index * 1000 + sentence_index, sentence))
        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
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
            selected.extend(candidate for candidate in ranked if candidate not in selected)
        return [_shorten_text(item[2]) for item in sorted(selected[:limit], key=lambda item: item[1])]

    @staticmethod
    def _generate_response(prompt: str) -> tuple[str | None, PerformanceMetrics]:
        global _GENERATOR_READY
        metrics = PerformanceMetrics()
        try:
            started = time.perf_counter()
            generator_was_ready = _GENERATOR_READY
            generator = _load_generator()
            _GENERATOR_READY = True
            metrics.record("model_cache_acquisition" if generator_was_ready else "model_loading", time.perf_counter() - started)
            started = time.perf_counter()
            output = generator(
                [{"role": "system", "content": "You are CampusGuide. Be accurate and source-grounded."}, {"role": "user", "content": prompt}],
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.05,
                use_cache=True,
            )
            metrics.record("qwen_inference", time.perf_counter() - started)
            generated = output[0]["generated_text"]
            if isinstance(generated, list):
                generated = generated[-1].get("content", "")
            metrics.values["generated_word_count"] = len(str(generated).split())
            return str(generated).strip() or None, metrics
        except Exception as exc:
            metrics.values["generator_status"] = f"unavailable: {type(exc).__name__}"
            return None, metrics

    @staticmethod
    def _not_found_result(metrics: PerformanceMetrics) -> RAGResult:
        metrics.finish()
        return RAGResult("Sorry, I could not find information about that in the uploaded documents.", [], "Try using the exact subject, topic, policy name, or unit title from the uploaded file.", [], False, False, metrics)


@st.cache_resource(show_spinner=False)
def _load_embedder() -> SentenceTransformer:
    """One shared encoder avoids duplicate model RAM across cached indexes."""
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def _load_generator():
    """Lazy, singleton Qwen initialization: no model load until the first question."""
    from transformers import pipeline
    return pipeline("text-generation", model=GENERATION_MODEL, device=-1)


def _meaningful_terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-zA-Z]{3,}", text.lower()) if term not in RETRIEVAL_STOP_WORDS}


def _shorten_text(text: str, maximum_characters: int = 240) -> str:
    return text if len(text) <= maximum_characters else f"{text[:maximum_characters].rsplit(' ', 1)[0]}..."


def _process_rss_mib() -> float | None:
    """Current Linux RSS; Streamlit Community Cloud exposes /proc."""
    try:
        with open("/proc/self/statm", encoding="utf-8") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, AttributeError, ValueError):
        return None
