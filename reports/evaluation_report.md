# TaxGuard AI — Evaluation Report

**Validated status decisions:** 18/50 (36.0%)

## Results

- Test cases: 50 (40 supported; 10 expected abstentions)
- Decision accuracy: **36.0%**
- Retrieval Recall@5: **20.0%**
- Retrieval top-1 accuracy: **20.0%**
- MRR: **0.2**
- Precision@5: **0.08**
- Expected chunk Recall@5: **20.0%**
- Expected chunk MRR: **0.2**
- Expected-answer evidence coverage: **17.5%**
- Deterministic planner route accuracy: **100%**
- Abstention precision: **0.2381**
- Abstention recall: **1.0**
- Abstention F1: **0.3846**

**Retrieval mode actually run:** `bm25+tfidf+rrf (dense optional; disabled)`

## Important interpretation

These are measurements on synthetic, hand-labeled test cases. Retrieval metrics measure expected-document retrieval; decision accuracy measures expected answered/abstained status. No generative LLM was called, so this is not LLM answer accuracy.

The default evaluation environment uses BM25 + TF-IDF candidate rankings fused by RRF. The optional Chroma + all-MiniLM-L6-v2 dense branch is implemented behind `TAXGUARD_USE_DENSE=true` and optional dependencies, but is only counted as evaluated when those packages and model weights are available. Do not describe these scores as dense-hybrid results unless `dense_model_evaluated` is true.

## Case-level results

| Case | Type | Expected | Actual | Status | Doc rank | Chunk rank |
|---|---|---|---|---:|---:|---:|
| positive-01 | supported | answered | answered | PASS | 1 | 1 |
| positive-02 | supported | answered | answered | PASS | 1 | 1 |
| positive-03 | supported | answered | answered | PASS | 1 | 1 |
| positive-04 | supported | answered | answered | PASS | 1 | 1 |
| positive-05 | supported | answered | answered | PASS | 1 | 1 |
| positive-06 | supported | answered | answered | PASS | 1 | 1 |
| positive-07 | supported | answered | answered | PASS | 1 | 1 |
| positive-08 | supported | answered | answered | PASS | 1 | 1 |
| positive-09 | supported | answered | abstained | FAIL | — | — |
| positive-10 | supported | answered | abstained | FAIL | — | — |
| positive-11 | supported | answered | abstained | FAIL | — | — |
| positive-12 | supported | answered | abstained | FAIL | — | — |
| positive-13 | supported | answered | abstained | FAIL | — | — |
| positive-14 | supported | answered | abstained | FAIL | — | — |
| positive-15 | supported | answered | abstained | FAIL | — | — |
| positive-16 | supported | answered | abstained | FAIL | — | — |
| positive-17 | supported | answered | abstained | FAIL | — | — |
| positive-18 | supported | answered | abstained | FAIL | — | — |
| positive-19 | supported | answered | abstained | FAIL | — | — |
| positive-20 | supported | answered | abstained | FAIL | — | — |
| positive-21 | supported | answered | abstained | FAIL | — | — |
| positive-22 | supported | answered | abstained | FAIL | — | — |
| positive-23 | supported | answered | abstained | FAIL | — | — |
| positive-24 | supported | answered | abstained | FAIL | — | — |
| positive-25 | supported | answered | abstained | FAIL | — | — |
| positive-26 | supported | answered | abstained | FAIL | — | — |
| positive-27 | supported | answered | abstained | FAIL | — | — |
| positive-28 | supported | answered | abstained | FAIL | — | — |
| positive-29 | supported | answered | abstained | FAIL | — | — |
| positive-30 | supported | answered | abstained | FAIL | — | — |
| positive-31 | supported | answered | abstained | FAIL | — | — |
| positive-32 | supported | answered | abstained | FAIL | — | — |
| positive-33 | supported | answered | abstained | FAIL | — | — |
| positive-34 | supported | answered | abstained | FAIL | — | — |
| positive-35 | supported | answered | abstained | FAIL | — | — |
| positive-36 | supported | answered | abstained | FAIL | — | — |
| positive-37 | supported | answered | abstained | FAIL | — | — |
| positive-38 | supported | answered | abstained | FAIL | — | — |
| positive-39 | supported | answered | abstained | FAIL | — | — |
| positive-40 | supported | answered | abstained | FAIL | — | — |
| abstention-01 | no_eligible_evidence | abstained | abstained | PASS | — | — |
| abstention-02 | no_eligible_evidence | abstained | abstained | PASS | — | — |
| abstention-03 | no_eligible_evidence | abstained | abstained | PASS | — | — |
| abstention-04 | no_eligible_evidence | abstained | abstained | PASS | — | — |
| abstention-05 | no_eligible_evidence | abstained | abstained | PASS | — | — |
| abstention-06 | no_eligible_evidence | abstained | abstained | PASS | — | — |
| abstention-07 | no_eligible_evidence | abstained | abstained | PASS | — | — |
| abstention-08 | no_eligible_evidence | abstained | abstained | PASS | — | — |
| abstention-09 | no_eligible_evidence | abstained | abstained | PASS | — | — |
| abstention-10 | no_eligible_evidence | abstained | abstained | PASS | — | — |
