import pytest

import rag


def hit(doc_id, text="t", page=1):
    return {"doc_id": doc_id, "filename": "f.pdf", "page": page, "text": text, "score": 0.0}


def test_rrf_single_list_rank_one_scores_1_over_61():
    out = rag.rrf([[hit("a")]])
    assert out[0]["score"] == pytest.approx(1 / 61)


def test_rrf_doc_in_both_lists_beats_doc_in_one():
    a, b, c = hit("a"), hit("b"), hit("c")
    # b is rank 1 in list 1 only; a is rank 2 in both lists
    out = rag.rrf([[b, a], [c, a]])
    assert out[0]["doc_id"] == "a"
    assert [h["doc_id"] for h in out] == ["a", "b", "c"]


def test_bm25_empty_corpus_returns_empty(monkeypatch):
    monkeypatch.setattr(rag.store, "get", lambda **kw: {"documents": [], "metadatas": []})
    monkeypatch.setattr(rag, "_index", None)
    assert rag.bm25("anything") == []


def test_bm25_ranks_keyword_match_first(monkeypatch):
    meta = lambda i: {"doc_id": f"d{i}", "filename": "f.pdf", "page": i}  # noqa: E731
    docs = ["revenue grew in cloud", "employees and offices", "risk factors and litigation"]
    monkeypatch.setattr(rag.store, "get", lambda **kw: {"documents": docs, "metadatas": [meta(i) for i in range(3)]})
    monkeypatch.setattr(rag, "_index", None)
    out = rag.bm25("Revenue!")
    assert out[0]["text"] == docs[0]
    assert len(out) == 1  # zero-score chunks are dropped


def test_hybrid_fuses_dense_and_bm25(monkeypatch):
    a, b = hit("a"), hit("b")
    monkeypatch.setattr(rag, "dense", lambda q, k=20: [b, a])
    monkeypatch.setattr(rag, "bm25", lambda q, k=20: [a])
    assert [h["doc_id"] for h in rag.hybrid("q")] == ["a", "b"]


def test_rrf_same_text_on_different_pages_stays_two_hits():
    out = rag.rrf([[hit("a", "footer", page=1), hit("a", "footer", page=2)]])
    assert [h["page"] for h in out] == [1, 2]


def test_rerank_orders_by_cross_encoder_score_and_keeps_top(monkeypatch):
    class Stub:
        def predict(self, pairs):
            return [0.1, 0.9, 0.5]
    monkeypatch.setattr(rag, "_reranker", lambda: Stub())
    out = rag.rerank("q", [hit("a"), hit("b"), hit("c")], keep=2)
    assert [h["doc_id"] for h in out] == ["b", "c"]


def test_rerank_empty_input_does_not_load_model(monkeypatch):
    monkeypatch.setattr(rag, "_reranker", lambda: pytest.fail("model loaded"))
    assert rag.rerank("q", []) == []


def test_search_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        rag.search("q", mode="nope")
