# CampusGuide: RAG Chatbot for Student Support Services

CampusGuide is a document-grounded student-support chatbot built for the IBM Generative AI Project-Based Experiential Learning Program. It answers student questions from university documents and displays the retrieved source passages with relevance scores.

## Features

- Upload PDF or TXT university documents
- Chunk and embed document text with `all-MiniLM-L6-v2`
- Retrieve the top relevant passages using cosine similarity
- Generate a grounded answer with Hugging Face `google/flan-t5-base`
- Fall back to the best source passage if the generation model cannot load
- Show retrieved evidence and similarity score for explainability
- Include a sample fictional university handbook for immediate testing

## RAG flow

1. Read documents and split them into overlapping chunks.
2. Convert chunks to vector embeddings.
3. Convert the student's question to an embedding and retrieve the nearest chunks.
4. Place the retrieved context and question into a constrained prompt.
5. Generate an answer and show the source chunks used.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the Hugging Face embedding and generation models. An internet connection is required for that initial download.

## Suggested demo questions

- What is the attendance requirement?
- How can I apply for financial assistance?
- What wellbeing support can I use?
- When is the library open?

## Project concepts demonstrated

- Natural language processing and contextual embeddings
- Semantic vector search and cosine similarity
- Retrieval-Augmented Generation (RAG)
- Hugging Face models
- Prompt engineering and source-grounded responses
- Responsible AI: source transparency, privacy awareness, and a safety disclaimer

## Important note

The sample handbook is fictional. For a final university-specific deployment, use only public, approved, up-to-date university documents and have the institution review high-stakes policies.
