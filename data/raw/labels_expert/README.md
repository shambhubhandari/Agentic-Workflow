# Expert labels — the reference standard

Every precision, recall, F1 and MCC value the manuscript reports is computed against
these labels. 25 papers, 201 scorable judgements, produced by a domain expert reading the
papers directly.

## The two batches

| file | how it was made |
|---|---|
| `batch_01_with_values.csv` | 15 papers, read from the published PDFs, recording the stated value alongside each decision |
| `batch_02_decisions.csv` | 12 papers, read from the retrieved text, recording the reported / not-reported decision only |

Precision, recall and the reproducibility verdict all read the decision alone, so the two
batches are equivalent for scoring. Values are needed only for the conversion-loss
measurement, which is therefore scoped to batch 1.

`scripts/merge_labels.py` joins them into `data/interim/labels_expert_merged.csv`, the
long-form table every metric reads.

## The three decisions

| value | meaning |
|---|---|
| `y` | the paper states this parameter |
| `n` | the parameter applies to what this paper did, and is not stated — a reporting failure |
| `na` | the parameter does not apply — e.g. basis size on a plane-wave code, force threshold where nothing was relaxed |

`n` versus `na` is the distinction that matters most, and the one that most often
separates a human reading from an automated one. Rows marked `na` carry no decision and
are skipped by every metric.

## Two papers carry no judgement

Two of the 27 reached the pipeline as abstract-only stubs — retrieval returned a landing
page rather than an article. Every field is `na` for them, so they contribute nothing to
the 201 judgements and nothing to any reported figure.

## Limits worth stating

These are one expert's judgements. No second annotator completed an independent pass, so
no inter-annotator agreement statistic exists, and the subset is small enough that the
recall interval is correspondingly wide (59.8–74.0%).
