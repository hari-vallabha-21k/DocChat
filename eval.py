# Phase 4: python eval.py            retrieval metrics only (no LLM calls)
#          python eval.py --answers  + answer metrics (one LLM call per question per mode)
#          python eval.py --ragas    + RAGAS scores (many more LLM calls)
import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import rag

K = 5
MODES = [("dense", "dense", False), ("bm25", "bm25", False),
         ("hybrid", "hybrid", False), ("hybrid+rerank", "hybrid", True)]  # (label, mode, rerank)
QUERIES = Path("eval/test_queries.json")
OUT = Path("eval/results.md")


def pages(item: dict) -> list[int]:
    p = item["page"]
    return p if isinstance(p, list) else [p]


def is_gold(hit: dict, item: dict) -> bool:  # hits and citations both carry filename + page
    return hit["filename"] == item["source"] and hit["page"] in pages(item)


def first_rank(hits: list[dict], item: dict) -> int | None:
    return next((r for r, h in enumerate(hits, 1) if is_gold(h, item)), None)


def pct(xs: list[float], q: float) -> float:  # nearest-rank percentile
    xs = sorted(xs)
    return xs[max(0, math.ceil(q * len(xs)) - 1)] if xs else 0.0


def retrieval_rows(items: list[dict], mode: str, rerank: bool) -> list[dict]:
    rows = []
    for it in items:
        t0 = time.perf_counter()
        hits = rag.search(it["question"], mode, rerank, K)
        rows.append({"item": it, "hits": hits, "ms": (time.perf_counter() - t0) * 1000})
    return rows


def retrieval_metrics(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["item"]["source"]]
    ranks = [first_rank(r["hits"], r["item"]) for r in ans]
    by_type = defaultdict(list)
    for r, rk in zip(ans, ranks):
        by_type[r["item"]["type"]].append(rk is not None)
    ms = [r["ms"] for r in rows]
    return {"hit": fmean(rk is not None for rk in ranks), "mrr": fmean(1 / rk if rk else 0 for rk in ranks),
            "p50": pct(ms, 0.5), "p95": pct(ms, 0.95), "by_type": {t: fmean(v) for t, v in by_type.items()}}


def threshold_stats(rows: list[dict]) -> dict:
    """Rerank rows only: how MIN_RERANK_SCORE splits answerable from unanswerable questions."""
    top = lambda r: r["hits"][0]["score"] if r["hits"] else float("-inf")  # noqa: E731
    ans = [top(r) for r in rows if r["item"]["source"]]
    una = [top(r) for r in rows if not r["item"]["source"]]
    return {"refused_unanswerable": fmean(s < rag.MIN_SCORE for s in una) if una else 0.0,
            "refused_answerable": fmean(s < rag.MIN_SCORE for s in ans) if ans else 0.0,
            "answerable_p10": pct(ans, 0.1), "unanswerable_max": max(una, default=float("nan"))}


def answer_rows(items: list[dict], mode: str, rerank: bool) -> list[dict]:
    return [{"item": it, **rag.answer(it["question"], mode, rerank, K)} for it in items]


def answer_metrics(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["item"]["source"]]
    una = [r for r in rows if not r["item"]["source"]]
    refused = lambda r: rag.NO_ANSWER in r["answer"]  # noqa: E731
    lat = [r["latency_ms"] for r in rows]
    return {"citation": fmean(any(is_gold(c, r["item"]) for c in r["citations"]) for r in ans),
            "refused_unanswerable": fmean(refused(r) for r in una) if una else 0.0,
            "refused_answerable": fmean(refused(r) for r in ans),
            "p50": pct(lat, 0.5), "p95": pct(lat, 0.95)}


def ragas_scores(rows: list[dict]) -> dict:
    # imports here so a plain `python eval.py` works without ragas installed
    from langchain.chat_models import init_chat_model
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
    from ragas.run_config import RunConfig

    # ponytail: judge defaults to the answering model (self-preference bias); set JUDGE_MODEL for a different one
    llm = LangchainLLMWrapper(init_chat_model(os.getenv("JUDGE_MODEL", os.environ["LLM_MODEL"]), temperature=0))
    emb = LangchainEmbeddingsWrapper(rag.embeddings)  # local MiniLM: no embedding API cost
    data = EvaluationDataset(samples=[
        SingleTurnSample(user_input=r["item"]["question"], response=r["answer"],
                         retrieved_contexts=r["contexts"], reference=r["item"]["reference"]) for r in rows])
    metrics = [Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=emb),
               LLMContextPrecisionWithReference(llm=llm), LLMContextRecall(llm=llm)]
    df = evaluate(data, metrics, run_config=RunConfig(max_workers=4, timeout=120)).to_pandas()
    return df.select_dtypes("number").mean().to_dict()  # column names differ between ragas versions


def md_table(header: list[str], body: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(map(str, row)) + " |" for row in body])


def report(items, ret, thr, ans, ragas) -> str:
    n_un = sum(not it["source"] for it in items)
    types = sorted({it["type"] for it in items} - {"unanswerable"})
    f = lambda x: f"{x:.0%}"  # noqa: E731
    out = [f"# Evaluation results\n\nGenerated by `python eval.py`. {len(items)} questions "
           f"({len(items) - n_un} answerable, {n_un} unanswerable), top-{K}. Hits match on filename + page.\n",
           "## Retrieval (no LLM)\n",
           md_table(["Mode", f"hit@{K}", f"MRR@{K}", "p50 ms", "p95 ms"],
                    [[m, f(v["hit"]), f"{v['mrr']:.3f}", f"{v['p50']:.0f}", f"{v['p95']:.0f}"] for m, v in ret.items()]),
           f"\n### hit@{K} by question type\n",
           md_table(["Mode"] + types, [[m] + [f(v["by_type"].get(t, 0)) for t in types] for m, v in ret.items()]),
           f"\n## Refusal threshold (hybrid+rerank, MIN_RERANK_SCORE={rag.MIN_SCORE})\n",
           f"- Unanswerable refused: {f(thr['refused_unanswerable'])}; answerable wrongly refused: {f(thr['refused_answerable'])}",
           f"- Top rerank score: answerable 10th percentile {thr['answerable_p10']:.2f}; unanswerable max {thr['unanswerable_max']:.2f}. "
           "A good threshold sits between the two."]
    if ans:
        out += [f"\n## Answers (LLM: {os.environ['LLM_MODEL']})\n",
                md_table(["Mode", "citation accuracy", "unanswerable refused", "answerable refused", "p50 ms", "p95 ms"],
                         [[m, f(v["citation"]), f(v["refused_unanswerable"]), f(v["refused_answerable"]),
                           f"{v['p50']:.0f}", f"{v['p95']:.0f}"] for m, v in ans.items()])]
    if ragas:
        names = list(next(iter(ragas.values())))
        out += ["\n## RAGAS (answerable questions)\n",
                md_table(["Mode"] + names, [[m] + [f"{v[n]:.3f}" for n in names] for m, v in ragas.items()])]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true")
    ap.add_argument("--ragas", action="store_true", help="implies --answers")
    ap.add_argument("--limit", type=int, help="first N questions only (smoke test)")
    a = ap.parse_args()
    items = json.loads(QUERIES.read_text(encoding="utf-8"))[: a.limit]
    rag.dense("warmup"), rag.bm25("warmup"), rag._reranker()  # keep model loading and index build out of the timings

    ret, ans, ragas, rerank_rows = {}, {}, {}, []
    for label, mode, rerank in MODES:
        rows = retrieval_rows(items, mode, rerank)
        ret[label] = retrieval_metrics(rows)
        if rerank:
            rerank_rows = rows
        if a.answers or a.ragas:
            arows = answer_rows(items, mode, rerank)
            ans[label] = answer_metrics(arows)
            if a.ragas:
                ragas[label] = ragas_scores([r for r in arows if r["item"]["source"]])
        print(label, ret[label])
    OUT.write_text(report(items, ret, threshold_stats(rerank_rows), ans, ragas), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
