# CampusGuide Performance Review

## Scope and method

This review targeted the deployed Streamlit Community Cloud application without changing its features or user interface. Measurements were collected locally on 1 August 2026 with the 12 MB `Student-Manual-SRMS-CET-CETR.pdf` and the question **“What is SRMS hostel?”**. Each end-to-end run used a new Python process, so it includes model acquisition from the local Hugging Face cache. Network download time is not included.

## Measured result

| Stage | Before | After | Change |
| --- | ---: | ---: | ---: |
| PDF text extraction | 1.10 s | 0.87 s | Input-dependent |
| Chunking | 0.01 s | 0.01 s | No material change |
| Embedding generation | 2.74 s | 3.10 s | One-time indexing cost; smaller chunks increase quality |
| Retrieval | 0.04 s | 0.04 s | Already efficient |
| Prompt | 1,077 words / 6,932 chars | 357 words / 2,209 chars | 67% smaller |
| Retrieved context | 5 chunks | 2 chunks | 60% fewer chunks |
| Qwen inference | 77.65 s | 27.45 s | 65% faster |
| Total request latency | 80.73 s | 30.43 s | 62% faster |

The exact values will vary by Streamlit Community Cloud instance load and CPU allocation. The diagnostic panel in the app reports the same stages in the deployed process.

## Bottlenecks found and corrections

1. **Qwen CPU inference was the dominant bottleneck.** It accounted for more than 96% of baseline request time. The default remains CPU-friendly `Qwen/Qwen2.5-0.5B-Instruct`, loaded lazily and once per Streamlit process. Its prompt now contains only the strongest source section and its continuation, rather than five loosely related chunks.

2. **The old 1,250-word context often contained unrelated PDF text.** It caused slower inference and encouraged the model to blend separate policy sections. The new 720-word maximum, with a measured 357-word prompt for this query, reduces prefill work and improves grounding.

3. **Chunk size exceeded the embedding model’s practical context window.** Chunks changed from 220 words with 35-word overlap to 180 words with 24-word overlap. This slightly increases one-time indexing work, but avoids truncating the tail of long chunks before they reach MiniLM and improves retrieval precision.

4. **Lexical retrieval over-weighted common institution names.** Raw term-overlap scoring treated a word such as “SRMS” as highly informative even when it occurred throughout a document. Retrieval now uses inverse document frequency (IDF) weighting and reduces the lexical boost, so rare topic words carry more value.

5. **The context selector included the preceding PDF chunk by default.** PDF extraction can interleave nearby headings. This pulled placement rules into a hostel answer. The selector now retains the best chunk and its continuation; non-adjacent chunks are added only when they are within 90% of the best relevance score.

6. **Grounding validation embedded the full concatenated context on every request.** It now embeds only the generated response and compares it to precomputed chunk vectors. This avoids re-encoding source text and reduced the measured validation time from 0.14 s to 0.03 s.

7. **`app.py` contained duplicate definitions of `build_engine`, `save_uploads`, and `render_sources`.** Python discarded the earlier definitions at runtime, but they made maintenance risky. The obsolete definitions were removed, leaving one implementation of each operation.

8. **Performance data did not distinguish index build from a question request.** Metrics now label document indexing separately, include CPU-seconds and derived average process CPU percentage, a prompt-token estimate, retrieved chunk count, response rendering time, and Streamlit script runtime.

## Production architecture after optimization

- `st.cache_resource` caches each document index (up to two document sets) across ordinary reruns.
- `st.cache_resource` also caches the MiniLM embedder and Qwen generator as process-wide singleton resources.
- Upload bytes are persisted once per browser session and content-addressed by SHA-256; documents are not re-saved during normal widget reruns.
- Extraction, chunking, and embedding occur only when a document signature changes.
- A query only embeds the question, runs vector ranking across in-memory NumPy vectors, constructs focused context, calls Qwen, then checks grounding against cached vectors.
- The debug panel remains off by default and introduces no rendering work unless toggled on.

## Resource and deployment impact

- **Warm document queries:** expected to be around 60% faster for similarly broad questions because the model processes markedly less context.
- **Cold start:** model download/load remains the unavoidable main cost. Lazy loading keeps the first landing-page render fast; the model is only initialized after a supported question reaches generation.
- **RAM:** one MiniLM model and one Qwen model are shared within the process. Eliminating duplicate model instances and limiting cached indexes to two avoids avoidable resident-memory growth. The diagnostics panel reports RSS in Community Cloud’s Linux environment.
- **Dependency footprint:** `torchvision` was removed because the project does not use computer vision; only Streamlit, Sentence Transformers, Transformers, PyTorch, pypdf, and NumPy remain.

## Files changed

- `app.py` — removed duplicate helpers; added low-overhead rendering and Streamlit-run diagnostics.
- `rag_engine.py` — tuned chunking, IDF-aware retrieval, focused context selection, singleton resource caching, concise prompt construction, and lower-cost grounding validation.
- `requirements.txt` — keeps only runtime dependencies and compatible version bounds.
- `README.md` — updates deployment and diagnostics notes.
- `PERFORMANCE_REPORT.md` — this measured engineering review.

## Operational recommendation

Use the default 0.5B Qwen model on Streamlit Community Cloud. A larger override through `CAMPUSGUIDE_GENERATION_MODEL` may improve prose quality but will materially increase cold-start time, RAM use, and CPU latency. Do not pre-load Qwen on page startup; lazy loading provides the best perceived responsiveness for this app.
