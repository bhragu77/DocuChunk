# RAGAS cross-check

Our harness computes RAGAS-*aligned* metrics with its own implementations. This artifact scores the SAME run with the real `ragas` library and reports the gap, so "RAGAS-aligned" is a measured claim rather than a naming choice.

- **Source report:** `generation_quality`, profile `neural`, 12 queries
- **RAGAS judge model:** `gemini-flash-lite-latest`

| metric | ours | RAGAS | |Δ| |
|---|:---:|:---:|:---:|
| answer_correctness | 0.9778 | — | — |
| answer_relevancy | 1.0 | nan | nan |
| context_precision | 0.9589 | 0.6528 | 0.3061 |
| context_recall | 0.9481 | 0.5833 | 0.3648 |
| faithfulness | 0.8882 | 0.4389 | 0.4493 |

**Max absolute deviation: 0.4493**

### How to read this

- `context_precision` / `context_recall` are computed by us DETERMINISTICALLY from retrieval ground truth, while RAGAS infers relevance with an LLM — some gap here is expected and is not an error in either implementation.
- `faithfulness` / `answer_relevancy` are LLM-judged on both sides, so they carry the judge's own variance; agreement within a few points is the realistic bar, not equality.

### Status of this run: NOT VALIDATING

This run does **not** earn the "RAGAS-aligned (validated within ±ε)" claim. It
proves the cross-check executes end to end and nothing more. Four known defects,
in the order they need fixing:

1. **Truncated contexts (now fixed, needs a re-run).** `contexts` was exported
   clipped to 300 chars, and 149 of 225 exported contexts hit that cap. RAGAS
   re-judges faithfulness and context-recall against exactly these strings, so a
   supporting sentence past the cap made RAGAS score the claim unsupported. This
   is the most likely source of the large gaps above.
2. **Mismatched judge model.** RAGAS ran on `gemini-flash-lite-latest` while the
   harness baseline was judged by `gemini-3.1-flash-lite`, because the latter's
   free-tier daily quota was exhausted. Any faithfulness gap is confounded.
3. **`answer_relevancy` did not score.** RAGAS requests 3 candidate generations
   per answer; this model rejects n>1 outright, so all 12 of its jobs failed.
4. **12 of 45 queries only**, for the same quota reason.

Note that `context_precision` / `context_recall` are expected to differ by
construction: ours are DETERMINISTIC from retrieval ground truth (expected chunk
ids), while RAGAS infers relevance with an LLM from the context strings. Those two
are measuring related but different things, so exact agreement was never the bar.
`faithfulness` is the metric where close agreement is genuinely expected.

Re-run after a quota reset with the matched judge and full contexts before quoting
any number from this file.
