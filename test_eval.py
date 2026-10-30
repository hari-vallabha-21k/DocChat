import pytest

import eval as ev


def item(page=3, source="a.pdf", type="numeric"):
    return {"question": "q", "reference": "r", "source": source, "page": page, "type": type}


def unanswerable():
    return item(source=None, page=None, type="unanswerable")


def hit(filename="a.pdf", page=3, score=0.0):
    return {"filename": filename, "page": page, "score": score}


def test_is_gold_matches_file_and_any_listed_page():
    it = item(page=[3, 4])
    assert ev.is_gold(hit(page=4), it)
    assert not ev.is_gold(hit(filename="b.pdf", page=3), it)
    assert not ev.is_gold(hit(page=5), it)


def test_first_rank_is_one_based_and_none_when_missing():
    hits = [hit(page=1), hit(page=3)]
    assert ev.first_rank(hits, item()) == 2
    assert ev.first_rank(hits, item(page=9)) is None


def test_pct_nearest_rank():
    assert ev.pct([10, 20, 30, 40], 0.5) == 20
    assert ev.pct([10, 20, 30, 40], 0.95) == 40
    assert ev.pct([], 0.5) == 0.0


def test_retrieval_metrics_hit_and_mrr_skip_unanswerable():
    rows = [{"item": item(), "hits": [hit(page=3)], "ms": 1.0},
            {"item": item(), "hits": [hit(page=1), hit(page=3)], "ms": 3.0},
            {"item": item(page=9), "hits": [hit()], "ms": 2.0},
            {"item": unanswerable(), "hits": [hit()], "ms": 2.0}]
    m = ev.retrieval_metrics(rows)
    assert m["hit"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx(0.5)  # (1 + 1/2 + 0) / 3


def test_threshold_stats_splits_answerable_from_unanswerable(monkeypatch):
    monkeypatch.setattr(ev.rag, "MIN_SCORE", 0.0)
    row = lambda it, s: {"item": it, "hits": [hit(score=s)], "ms": 0}  # noqa: E731
    t = ev.threshold_stats([row(item(), 5), row(item(), -1), row(unanswerable(), -3), row(unanswerable(), 2)])
    assert t["refused_answerable"] == 0.5
    assert t["refused_unanswerable"] == 0.5
    assert t["unanswerable_max"] == 2


def test_answer_metrics_citation_and_refusal():
    ok = {"item": item(), "answer": "x [1]", "citations": [{"filename": "a.pdf", "page": 3}], "latency_ms": 100}
    miss = {"item": item(), "answer": ev.rag.NO_ANSWER, "citations": [], "latency_ms": 300}
    una = {"item": unanswerable(), "answer": ev.rag.NO_ANSWER, "citations": [], "latency_ms": 50}
    m = ev.answer_metrics([ok, miss, una])
    assert m["citation"] == 0.5
    assert m["refused_answerable"] == 0.5
    assert m["refused_unanswerable"] == 1.0
