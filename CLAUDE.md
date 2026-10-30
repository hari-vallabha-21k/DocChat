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
2. Retrieval: dense, bm25, rrf, rerank ← NEXT
3. Answers + citations
4. Evaluation: test set + eval.py → eval/results.md
5. API + UI
6. Docker + README (Mermaid diagram, results table, design decisions)
7. Optional: Hugging Face Spaces

## Progress
- Lesson 0 (what RAG is; retrieval failures vs generation failures): done.
- Lesson 1 (embeddings, cosine similarity, truncation, chunk size): done. Corrected two misconceptions: text past 256 tokens is dropped, not moved to another chunk; "context window" (LLM limit) is not "max sequence length" (embedder limit). I needed help on why a chunk shouldn't be one sentence or a whole page.
- Next: I fill the 5 TODOs in rag.py. Before running I predict (A) how many chunks a 100-page PDF makes and (B) whether running twice doubles the Chroma count. Then I run `python rag.py data/pdfs` plus the check below, and you review.

```python
from rag import store, tokenizer
r = store.get(include=["documents", "metadatas"])
print(len(r["ids"]), "chunks total")
print(r["metadatas"][0])                                        # doc_id, filename, page
print(max(len(tokenizer.tokenize(d)) for d in r["documents"]))  # should be <= 250
```

## Setup
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
API keys (OPENAI_API_KEY, ADMIN_API_KEY) go in Codespaces secrets, never in files. Stop the Codespace when done (free quota: 60 h/month on 2-core).
