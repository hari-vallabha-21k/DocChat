# DocChat: RAG document Q&A with evaluation

Portfolio project. PDFs → hybrid search (dense + BM25, merged with RRF) → cross-encoder rerank → LLM answer with [n] citations (file + page) → evaluated with RAGAS + custom retrieval metrics, comparing retrieval modes.

## How to work with me (teaching mode)
I'm learning RAG while building this. I know Python and FastAPI. For each phase:
1. Explain the concept first in plain language, short, with an analogy.
2. Give a small experiment and ask me to predict the result before I run it.
3. I write the code. Give hints, not answers, unless I ask. Then review my code: what's right, what's wrong, and why.
4. Ask me 3 interview-style questions. Grade my answers honestly and correct misconceptions.
5. Suggest 2–3 lines for LEARNINGS.md.

Don't write a phase's code for me unless I ask. Keep explanations simple.

## Stack
- Runs in GitHub Codespaces (2-core / 8 GB, CPU only). Install CPU-only torch first.
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim, reads max 256 tokens), normalized
- Vector DB: Chroma, persist_directory="chroma", cosine distance
- Keyword search: rank_bm25, rebuilt from Chroma's stored chunks at startup (no separate save)
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- LLM: LangChain `init_chat_model`, model string from `LLM_MODEL` env var (OpenAI mini-tier by default, Gemini Flash as a swap)
- Eval: RAGAS (pin the version; its API changes between releases) + hit@5, MRR, citation accuracy, p50/p95 latency
- API: FastAPI. UI: Streamlit. Docker Compose. Public demo on Hugging Face Spaces.

## Decisions (and why)
- Chunks: 250 tokens, 32 overlap, counted with MiniLM's tokenizer. MiniLM silently drops text after 256 tokens; 250 + 2 special tokens fits.
- Chunk id `{doc_id}-{i}`, doc_id = sha256(file bytes)[:12], so re-ingesting the same file changes nothing.
- Page numbers stored 1-based (PyPDFLoader starts at 0).
- RRF written by hand: score = Σ 1/(60 + rank). Hybrid = RRF(dense top 20, BM25 top 20). Rerank top 20 → keep 5.
- BM25 tokens: lowercase `\w+`. Guard the empty corpus (rank_bm25 divides by zero).
- Eval compares 4 modes: dense, bm25, hybrid, hybrid+rerank. eval.py calls rag.py directly, not over HTTP.

## Layout
```
rag.py  api.py  ui.py  eval.py  test_rag.py
eval/test_queries.json   # {question, reference, source, page}, 30–50 items incl. 3–5 unanswerable
data/pdfs/               # 3–5 public 10-K PDFs
```

## API (api.py is a thin layer over rag.py)
| Method | Path | Auth | Success |
|---|---|---|---|
| GET | /health | public | 200 {status, chunks} |
| POST | /api/v1/answers | public, global limit 10/min | 200 {answer, citations[{n, doc_id, filename, page, snippet}], latency_ms} |
| GET | /api/v1/chunks?q=&mode=&rerank=&k= | public | 200 [{doc_id, filename, page, text, rank, score}] |
| GET | /api/v1/documents | public | 200 [{id, filename, pages, chunks, ingested_at}] |
| POST | /api/v1/documents | X-API-Key | 201 new / 200 already ingested |
| DELETE | /api/v1/documents/{doc_id} | X-API-Key | 204 |

- AnswerRequest: question 1–1000 chars (whitespace stripped), mode dense|bm25|hybrid (default hybrid), rerank default true, k 1–10 (default 5). /chunks allows k up to 20.
- Errors use FastAPI's default {"detail"}: 409 no documents yet · 413 over 20 MB · 415 not a PDF (check the `%PDF-` bytes) · 422 encrypted or no text · 429 with Retry-After · 502 LLM failed (30s timeout, 2 retries).
- Admin key: `ADMIN_API_KEY` env var, compared with `secrets.compare_digest`. If unset, uploads return 403 (fail closed).
- One lock around Chroma upsert + BM25 rebuild. Plain `def` endpoints (blocking work runs in the threadpool). No CORS (Streamlit calls the API server-side).
- Streamlit shows the upload panel only after the admin key is typed into a password field.

## Phases
0. Setup: repo + Codespace
1. Ingest (rag.py) ← DONE
2. Retrieval: dense, bm25, rrf, rerank ← DONE
3. Answers + citations ← code pushed, not yet run in the Codespace
4. Evaluation: test set + eval.py → eval/results.md ← IN PROGRESS (code pushed, not yet run)
5. API + UI
6. Docker + README (Mermaid diagram, results table, design decisions)
7. Optional: Hugging Face Spaces

## Progress
- Lesson 0 (what RAG is; retrieval failures vs generation failures): done.
- Lesson 1 (embeddings, cosine similarity, truncation, chunk size): done. Corrected two misconceptions: text past 256 tokens is dropped, not moved to another chunk; "context window" (LLM limit) is not "max sequence length" (embedder limit). I needed help on why a chunk shouldn't be one sentence or a whole page.
- Phase 1 (ingest): DONE. 4 PDFs, 1824 chunks, max 250 tokens. Known limits: same bytes under two filenames (last wins); encrypted/corrupt PDFs crash the CLI loop (map to 422 in the API); models load at import of rag.py.
- Phase 2 (retrieval): DONE. dense, bm25, rrf, hybrid, rerank, search() in rag.py; 9 tests pass in the Codespace; all 4 modes return sensible pages. Reviews fixed: BM25 index+corpus are one `_index` tuple (thread-safe reads), RRF key includes page, CLI usage guard.
  - Experiment: dense beats BM25 on "What was Tesla's total revenue in 2024?" (BM25 returns Apple pages: common words, no company name in chunk text). BM25 beats dense on "Cybertruck" (dense returns cybersecurity/Nike). Rerank fixed the first but pushed the Cybertruck page to rank 3.
  - Deferred to the Phase 4 eval (decide with numbers): fuse rerank order with the hybrid order via rrf (fifth mode); prepend filename to chunk text (needs re-ingest); stopword filter in `_tokens`; `min_score` for unanswerable questions (needed in Phase 3); `Hit` TypedDict and `rank` field (Phase 5). Eval slices: rare-token, company-specific numeric, unanswerable. Call `_reranker()` before timing so p95 excludes model load.
- Preference: build the code, don't teach, unless asked.
- Phase 3 (answers): `answer()` + `ask` CLI in rag.py, `MIN_RERANK_SCORE` refusal (default 0, tune from the eval), 3 tests. Needs a Codespace run with `LLM_MODEL` set.
- Phase 4 (eval): `eval/test_queries.json` has 48 questions (28 numeric, 9 rare-token, 7 prose, 4 unanswerable), gold pages checked against the PDF text; `page` may be a list. `eval.py`: `python eval.py` (retrieval only, free), `--answers` (LLM), `--ragas` (pinned ragas==0.3.9, more LLM calls), `--limit N` smoke test. Writes eval/results.md, including a table that shows where to set `MIN_RERANK_SCORE`. 6 helper tests in test_eval.py pass locally with rag stubbed; not yet run in the Codespace. Both Tesla filings are in the corpus, so questions name the year.
- Next: run Phase 3 + Phase 4 in the Codespace, tune `MIN_RERANK_SCORE`, decide the deferred retrieval changes from the numbers.

## Setup
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
API keys (OPENAI_API_KEY, ADMIN_API_KEY) go in Codespaces secrets, never in files. Stop the Codespace when done (free quota: 60 h/month on 2-core).
