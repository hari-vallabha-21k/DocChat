# Baseline results (before adding a company/year header to chunks)

Colab CPU runtime, 48 questions (44 answerable, 4 unanswerable), top-5. Hits match on filename + page.
Kept so the header experiment in `eval/results.md` can be compared against it.

## Retrieval (no LLM)

| Mode | hit@5 | MRR@5 | p50 ms | p95 ms |
|---|---|---|---|---|
| dense | 48% | 0.280 | 24 | 28 |
| bm25 | 77% | 0.498 | 7 | 12 |
| hybrid | 68% | 0.449 | 33 | 39 |
| hybrid+rerank | 73% | 0.577 | 2499 | 3911 |

### hit@5 by question type

| Mode | numeric | prose | rare-token |
|---|---|---|---|
| dense | 50% | 14% | 67% |
| bm25 | 71% | 71% | 100% |
| hybrid | 71% | 43% | 78% |
| hybrid+rerank | 68% | 71% | 89% |

## Refusal threshold (hybrid+rerank, MIN_RERANK_SCORE=0.0)

- Unanswerable refused: 25%; answerable wrongly refused: 5%
- Top rerank score: answerable 10th percentile 1.83; unanswerable max 7.61. No threshold separates them, because the cross-encoder scores topical relevance, not answerability.
