# Phase 1: ingest (python rag.py data/pdfs). Phase 2: retrieval (python rag.py search "question"). Phase 3: python rag.py ask "question".
import functools
import hashlib
import os
import re
import sys
import time
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_N = 20      # candidates per retriever, and into the reranker
RRF_K = 60
tokenizer = AutoTokenizer.from_pretrained(MODEL)
embeddings = HuggingFaceEmbeddings(model_name=MODEL, encode_kwargs={"normalize_embeddings": True})
store = Chroma("docchat", embeddings, persist_directory="chroma",
               collection_metadata={"hnsw:space": "cosine"})  # cosine, same as the Colab experiment
# 240, not 256: the model adds 2 special tokens ([CLS], [SEP]) and ingest() prepends a ~10-token "[tesla 10k 2025]" header
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=240, chunk_overlap=32)

_index: tuple[BM25Okapi, list[dict]] | None = None  # (index, its chunks), one value so readers never see a mismatch; reset by ingest()


def ingest(path: str) -> int:
    global _index
    doc_id = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    pages = PyPDFLoader(path).load()
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError(f"{path}: no extractable text")
    header = Path(path).stem.replace("-", " ").replace("_", " ")[:40]  # 10-K chunks rarely name the company or year
    for c in chunks:
        c.metadata.update(doc_id=doc_id, filename=Path(path).name, page=c.metadata["page"] + 1)
        c.page_content = f"[{header}] {c.page_content}"
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    store.delete(where={"filename": Path(path).name})  # drop stale chunks of an edited/re-chunked file
    store.add_documents(chunks, ids=ids)  # same ids -> same rows, count stays flat
    _index = None  # BM25 index is stale now
    return len(chunks)


# ---- Phase 2: retrieval. A hit is {doc_id, filename, page, text, score}. ----

def _hit(text: str, meta: dict, score: float) -> dict:
    return {"doc_id": meta["doc_id"], "filename": meta["filename"], "page": meta["page"],
            "text": text, "score": score}


def _key(hit: dict) -> tuple:
    return (hit["doc_id"], hit["page"], hit["text"])  # page: identical boilerplate on two pages stays two hits


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def dense(query: str, k: int = TOP_N) -> list[dict]:
    # Chroma cosine distance = 1 - cosine similarity
    return [_hit(d.page_content, d.metadata, 1 - dist)
            for d, dist in store.similarity_search_with_score(query, k=k)]


def build_bm25() -> tuple[BM25Okapi, list[dict]] | None:
    global _index
    r = store.get(include=["documents", "metadatas"])
    corpus = [_hit(d, m, 0.0) for d, m in zip(r["documents"], r["metadatas"])]
    _index = (BM25Okapi([_tokens(h["text"]) for h in corpus]), corpus) if corpus else None  # rank_bm25 divides by zero on empty
    return _index


def bm25(query: str, k: int = TOP_N) -> list[dict]:
    idx = _index or build_bm25()  # read the global once; ingest() may reset it concurrently
    q = _tokens(query)
    if idx is None or not q:
        return []
    model, corpus = idx
    scores = model.get_scores(q)
    top = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
    return [{**corpus[i], "score": float(scores[i])} for i in top if scores[i] > 0]


def rrf(rankings: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank), rank starts at 1."""
    scores: dict[tuple, float] = {}
    first: dict[tuple, dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = _key(hit)
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            first.setdefault(key, hit)
    return [{**first[key], "score": s} for key, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def hybrid(query: str, k: int = TOP_N) -> list[dict]:
    return rrf([dense(query), bm25(query)])[:k]


@functools.cache
def _reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


def rerank(query: str, hits: list[dict], keep: int = 5) -> list[dict]:
    if not hits:
        return []
    scores = _reranker().predict([(query, h["text"]) for h in hits])
    ranked = sorted(zip(hits, scores), key=lambda hs: hs[1], reverse=True)
    return [{**h, "score": float(s)} for h, s in ranked[:keep]]


def search(query: str, mode: str = "hybrid", use_rerank: bool = True, k: int = 5) -> list[dict]:
    retrievers = {"dense": dense, "bm25": bm25, "hybrid": hybrid}
    if mode not in retrievers:
        raise ValueError(f"mode must be one of {list(retrievers)}")
    candidates = retrievers[mode](query)
    return rerank(query, candidates, k) if use_rerank else candidates[:k]


# ---- Phase 3: answers with [n] citations. ----

NO_ANSWER = "I can't find this in the documents."
# cross-encoder logits: relevant chunks score well above 0, unrelated ones go strongly negative. Tune in Phase 4 on the unanswerable slice.
MIN_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0"))
SYSTEM = (
    "Answer the question using ONLY the numbered context passages. "
    "Cite every claim with the passage number in square brackets, like [1] or [2][3]. "
    f"If the passages do not contain the answer, reply exactly: {NO_ANSWER}"
)


@functools.cache
def _llm():
    # LLM_MODEL like "openai:gpt-4o-mini" or "google_genai:gemini-2.5-flash"
    return init_chat_model(os.environ["LLM_MODEL"], temperature=0, timeout=30, max_retries=2)


def _text(msg) -> str:
    """Reply text; some providers (Gemini 3.x) return a list of content blocks instead of a string."""
    c = msg.content
    return c if isinstance(c, str) else "".join(b if isinstance(b, str) else b.get("text", "") for b in c)


def _citations(text: str, hits: list[dict]) -> list[dict]:
    """Passages the answer cites as [n]; numbers outside 1..len(hits) are dropped."""
    nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    return [{"n": n, "doc_id": h["doc_id"], "filename": h["filename"], "page": h["page"], "snippet": h["text"][:200]}
            for n in nums if 1 <= n <= len(hits) for h in [hits[n - 1]]]


def answer(question: str, mode: str = "hybrid", use_rerank: bool = True, k: int = 5) -> dict:
    t0 = time.perf_counter()
    hits = search(question, mode, use_rerank, k)
    if not hits or (use_rerank and hits[0]["score"] < MIN_SCORE):  # score is only comparable after rerank
        text, citations = NO_ANSWER, []
    else:
        context = "\n\n".join(f"[{i}] ({h['filename']} p{h['page']})\n{h['text']}" for i, h in enumerate(hits, 1))
        text = _text(_llm().invoke([("system", SYSTEM), ("human", f"Context:\n{context}\n\nQuestion: {question}")]))
        citations = _citations(text, hits)
    return {"answer": text, "citations": citations, "latency_ms": int((time.perf_counter() - t0) * 1000),
            "contexts": [h["text"] for h in hits]}  # for eval.py (RAGAS); the API layer leaves it out


if __name__ == "__main__":
    if len(sys.argv) < 2 or (sys.argv[1] in ("search", "ask") and len(sys.argv) != 3):
        sys.exit('usage: rag.py <pdf_dir> | rag.py search "question" | rag.py ask "question"')
    if sys.argv[1] == "ask":  # python rag.py ask "question"
        r = answer(sys.argv[2])
        print(r["answer"])
        for c in r["citations"]:
            print(f"  [{c['n']}] {c['filename']} p{c['page']}: {c['snippet'][:80]!r}")
        print(f"({r['latency_ms']} ms)")
    elif sys.argv[1] == "search":  # python rag.py search "question"
        for mode, use_rerank in [("dense", False), ("bm25", False), ("hybrid", False), ("hybrid", True)]:
            print(f"\n== {mode}{' + rerank' if use_rerank else ''} ==")
            for h in search(sys.argv[2], mode, use_rerank):
                print(f"{h['score']:.3f}  {h['filename']} p{h['page']}  {h['text'][:80]!r}")
    else:  # python rag.py data/pdfs
        for pdf in Path(sys.argv[1]).glob("*.pdf"):
            try:
                print(pdf.name, ingest(str(pdf)), "chunks")
            except ValueError as e:
                print("skipped:", e)
