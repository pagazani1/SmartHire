# Unseen-Resume Prediction Error Analysis

## Data inspected

This analysis uses only `predictions.csv` (18,913 test rows). Its seven columns are:
`anonymous_row_id`, `actual_label`, the two model prediction columns, and three
Logistic Regression class-probability columns. There are no missing values and
the anonymous row IDs are unique. No resume or job-description text was read,
printed, or saved in this analysis.

The class order used below is **Good Fit, No Fit, Potential Fit**. Confusion
matrix rows are actual labels; columns are predicted labels.

## Overall comparison

| Model | Accuracy | Macro-F1 | Incorrect predictions |
|---|---:|---:|---:|
| Logistic Regression | 0.721832 | 0.605955 | 5,261 |
| LinearSVC | 0.729287 | 0.568731 | 5,120 |

LinearSVC has slightly higher accuracy (about 0.0075), while Logistic
Regression has higher Macro-F1 (about 0.0372). Because Macro-F1 weights the
classes equally, the difference indicates a better class-balanced F1 for
Logistic Regression despite its slightly larger total error count.

### Per-class scores

| Model | Class | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Logistic Regression | Good Fit | 0.432479 | 0.625464 | 0.511369 |
| Logistic Regression | No Fit | 0.888053 | 0.777207 | 0.828941 |
| Logistic Regression | Potential Fit | 0.424073 | 0.546475 | 0.477556 |
| LinearSVC | Good Fit | 0.496962 | 0.454883 | 0.474992 |
| LinearSVC | No Fit | 0.833748 | 0.848046 | 0.840837 |
| LinearSVC | Potential Fit | 0.396148 | 0.384748 | 0.390365 |

Potential Fit has the lowest F1 for both classifiers. Its support is 3,475,
compared with 13,820 No Fit and 1,618 Good Fit.

## Confusion matrices

### Logistic Regression

| Actual \ Predicted | Good Fit | No Fit | Potential Fit |
|---|---:|---:|---:|
| Good Fit | 1,012 | 226 | 380 |
| No Fit | 880 | 10,741 | 2,199 |
| Potential Fit | 448 | 1,128 | 1,899 |

### LinearSVC

| Actual \ Predicted | Good Fit | No Fit | Potential Fit |
|---|---:|---:|---:|
| Good Fit | 736 | 483 | 399 |
| No Fit | 461 | 11,720 | 1,639 |
| Potential Fit | 284 | 1,854 | 1,337 |

## Error overlap

| Outcome | Rows |
|---|---:|
| Both models wrong | 3,580 |
| Only Logistic Regression wrong | 1,681 |
| Only LinearSVC wrong | 1,540 |
| Both models correct | 12,112 |

The counts partition all 18,913 rows. The large shared-error count shows that
the models often fail on the same examples, although this file does not contain
the resume/JD content needed to characterize those examples.

## Potential Fit errors

For actual Potential Fit rows:

| Model | Correctly predicted Potential Fit | Predicted Good Fit | Predicted No Fit |
|---|---:|---:|---:|
| Logistic Regression | 1,899 | 448 | 1,128 |
| LinearSVC | 1,337 | 284 | 1,854 |

For predictions labeled Potential Fit that were false positives:

| Model | Actual Good Fit predicted Potential Fit | Actual No Fit predicted Potential Fit |
|---|---:|---:|
| Logistic Regression | 380 | 2,199 |
| LinearSVC | 399 | 1,639 |

Logistic Regression has higher Potential Fit recall (0.546475 vs. 0.384748)
and higher precision (0.424073 vs. 0.396148) than LinearSVC. It recovers more
actual Potential Fit rows, with false positives mainly coming from No Fit.

## Logistic Regression probability patterns

“Confidence” here means the largest of the three saved Logistic Regression
class probabilities for a row. These are descriptive statistics; probability
calibration was not evaluated.

| Group | Rows | Accuracy | Mean maximum probability | Median maximum probability |
|---|---:|---:|---:|---:|
| Correct predictions | 13,652 | 1.000000 | 0.731299 | 0.740699 |
| Incorrect predictions | 5,261 | 0.000000 | 0.585100 | 0.561624 |

| Maximum probability range | Rows | Accuracy |
|---|---:|---:|
| Below 0.50 | 2,865 | 0.475044 |
| 0.50 to below 0.75 | 8,785 | 0.647923 |
| 0.75 or higher | 7,263 | 0.908578 |

Among actual Potential Fit rows, the model's Potential Fit probability averaged
0.630781 for correctly classified rows (1,899 rows) and 0.285202 for
misclassified rows (1,576 rows). For the misclassified subset, the mean
maximum class probability was still 0.614176, showing that some errors were
made with another class favored rather than with a uniformly diffuse
probability distribution. Across all rows, the Logistic Regression predicted
label agrees with the largest saved class probability for every row, and the
three probabilities sum to 1 within floating-point precision.

## Possible explanations (not established by this file)

- Potential Fit has fewer test examples than No Fit, and its F1 is lower for
  both models. This supports class frequency as a possible contributing
  factor, but does not prove that imbalance caused the errors.
- Potential Fit is confused with No Fit in both directions. The labels may
  represent a difficult decision boundary, but confirming that requires
  reviewing label definitions and examples.
- The saved probabilities show that many incorrect Potential Fit rows assign
  more probability to another class. This is consistent with ambiguous or
  overlapping signals, but the CSV has no text or feature information with
  which to verify that explanation.
- The probability/confidence relationship is descriptive only. No
  calibration metric or reliability analysis was computed, so high maximum
  probability should not be read as a calibrated likelihood.

## Limits

`predictions.csv` contains no raw text, feature vectors, or row-level model
scores for LinearSVC. Therefore this analysis can quantify outcomes and
Logistic Regression probability patterns, but cannot inspect representative
misclassified content, explain specific decisions, analyze LinearSVC
confidence, or assess probability calibration.

## Aggregate tables

The associated CSV summaries are:

- `model_comparison.csv`
- `per_class_metrics.csv`
- `confusion_matrix_comparison.csv`
- `joint_error_counts.csv`
- `potential_fit_analysis.csv`
- `logistic_confidence_analysis.csv`
