# Stage 5C: Controlled Resume–JD Matching Experiments

## Status

**Incomplete — no Stage 5C model comparison was produced.**

The implementation was started with the existing raw train/test Parquet
datasets, a seed-2026 20% resume-group holdout, three grouped development
folds, a maximum of five rows per resume-group/label for each training fit,
sampling seed 6842, and three grouped folds for fusion OOF probabilities.
The new vectorizers and model preprocessing were configured to fit only on
their corresponding training partitions. No synthetic resume fixture was
used.

The run remained CPU-bound for about 66 minutes and was stopped at the
interactive runtime limit. The last process check recorded approximately
3,711 seconds of CPU time, about 712 MiB process working set, and about
729 MiB available system RAM. Periodic memory checks did not observe the
evaluator's 256 MiB stop floor being crossed.

No fold metrics were persisted. Because the run produced no progress log or
intermediate checkpoints, the number of folds completed in memory cannot be
confirmed. No final holdout metrics or confusion matrices were emitted or
saved, and no fitted model bundle was saved. The configured split and training
protocol therefore were not confirmed by completed Stage 5C outputs.

## Planned comparison

| Model / ablation | Description | Stage 5C result |
|---|---|---|
| Existing text baseline | Concatenated resume and JD TF-IDF Logistic Regression; reference implementation is the saved Stage 2 baseline | Not completed |
| Separate resume/JD | Shared, training-only vocabulary with distinct resume and JD TF-IDF blocks | Not completed |
| Separate + interaction | Separate blocks plus element-wise TF-IDF overlap and cosine similarity | Not completed |
| Structured skill interaction | Existing required/preferred skill-group features plus overlap ratios, Jaccard overlap, and alternative-group counts | Not completed |
| Combined fusion | Structured features plus grouped OOF probabilities from the separate-text interaction model | Not completed |

Unavailable coverage or overlap ratios were represented as missing with
availability indicators in the implementation. The fusion design generated
OOF probabilities using grouped folds, keeping each resume group out of the
fold model that scores it. These implementation details were unit-tested,
but the complete training/evaluation workflow was not validated by a finished
run.

## Existing baseline reference only

The following are previously saved Stage 2 holdout metrics, included only as
context; they are **not** Stage 5C results and are not directly comparable to
the incomplete new experiments:

| Existing Stage 2 model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Text Logistic Regression | 0.6177 | 0.4896 | 0.6544 |
| Structured-only | 0.5621 | 0.4108 | 0.6119 |
| Existing hybrid | 0.6567 | 0.5355 | 0.6914 |

There are no persisted Stage 5C per-class scores, grouped-fold summary,
holdout confusion matrices, or ablation measurements to report. No claim of
improvement is supported.

## Runtime investigation and next step

The first implementation streams source records to limit memory use, but
extracts structured features in 1,024-row batches. The existing deterministic
feature generator caches repeated resume and JD skill extraction within each
batch; repeating that work across batch boundaries is a likely source of the
observed runtime. This is a performance hypothesis, not a measured profile.

Before rerunning, optimize only this deterministic feature-extraction
execution path (for example, share memoized skill results across batches)
while preserving the exact feature definitions, random seeds, grouped split,
cap, fold counts, model settings, and holdout isolation. Then run the complete
protocol. Do not reduce folds or cap size, use the holdout for tuning, or use
synthetic resumes to select a model. The Stage 2 text baseline remains the
only supported reference until the full Stage 5C comparison completes.

## Limitations

- The Stage 5C experiment is incomplete; no candidate model can be selected.
- The runtime limit prevented verification of grouped-fold and holdout
  results; no intermediate metrics were persisted.
- Memory was constrained at initial preflight (about 1.4 GiB available), but
  the run was stopped for duration rather than crossing its memory safety
  floor.
- Keyword skill extraction indicates textual mentions, not proficiency or
  candidate competence.
- The capped training sample is not representative of the original row-level
  class prevalence; results must retain that limitation if the experiment is
  rerun.
- Even after completion, a single grouped holdout is an estimate, not evidence
  of real-world hiring validity.

## Verification

The focused Stage 5C representation tests passed: **3 tests**. They verify
separate resume/JD feature blocks, class-probability alignment, and the
distinction between unavailable and zero skill overlap. They do not validate
model quality or the end-to-end evaluation.

The original production and prior experimental artifacts/results were
hash-checked before and after the run and remained unchanged. No Stage 5C
model weights were written.
