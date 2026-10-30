# Phase 1: ingest (python rag.py data/pdfs). Phase 2: retrieval (python rag.py search "question").
import functools
import hashlib
import re
import sys
from pathlib import Path

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
# 250, not 256: the model adds 2 special tokens ([CLS], [SEP]) to every input
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=250, chunk_overlap=32)

_bm25: BM25Okapi | None = None  # built lazily from Chroma; reset by ingest()
_corpus: list[dict] = []


def ingest(path: str) -> int:
    global _bm25
    doc_id = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    pages = PyPDFLoader(path).load()
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError(f"{path}: no extractable text")
    for c in chunks:
        c.metadata.update(doc_id=doc_id, filename=Path(path).name, page=c.metadata["page"] + 1)
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    store.delete(where={"filename": Path(path).name})  # drop stale chunks of an edited/re-chunked file
    store.add_documents(chunks, ids=ids)  # same ids -> same rows, count stays flat
    _bm25 = None  # BM25 index is stale now
    return len(chunks)


# ---- Phase 2: retrieval. A hit is {doc_id, filename, page, text, score}. ----

def _hit(text: str, meta: dict, score: float) -> dict:
    return {"doc_id": meta["doc_id"], "filename": meta["filename"], "page": meta["page"],
            "text": text, "score": score}


def _key(hit: dict) -> tuple:
    return (hit["doc_id"], hit["text"])


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def dense(query: str, k: int = TOP_N) -> list[dict]:
    # Chroma cosine distance = 1 - cosine similarity
    return [_hit(d.page_content, d.metadata, 1 - dist)
            for d, dist in store.similarity_search_with_score(query, k=k)]


def build_bm25() -> None:
    global _bm25, _corpus
    r = store.get(include=["documents", "metadatas"])
    _corpus = [_hit(d, m, 0.0) for d, m in zip(r["documents"], r["metadatas"])]
    _bm25 = BM25Okapi([_tokens(h["text"]) for h in _corpus]) if _corpus else None  # rank_bm25 divides by zero on empty


def bm25(query: str, k: int = TOP_N) -> list[dict]:
    if _bm25 is None:
        build_bm25()
    q = _tokens(query)
    if _bm25 is None or not q:
        return []
    scores = _bm25.get_scores(q)
    top = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
    return [{**_corpus[i], "score": float(scores[i])} for i in top if scores[i] > 0]


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


if __name__ == "__main__":
    if sys.argv[1] == "search":  # python rag.py search "question"
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
