# Data

Four directories, following standard data engineering practices aligned with the **Medallion Data Architecture** (`Bronze` $\rightarrow$ `Silver` $\rightarrow$ `Gold`). Which directory a file belongs to is strictly determined by *how it was produced*, guaranteeing end-to-end data lineage and reproducibility.

| Directory | Medallion Tier | Contract & Invariant |
|---|---|---|
| `external/` | **Bronze (External)** | Third-party immutable assets (e.g. norm-conserving ONCVPSP `.psml` pseudopotentials). Never edited. |
| `raw/` | **Bronze (Raw Ingest)** | Raw, immutable ingest snapshots exactly as received (frozen 298-paper corpus, fulltext manifest, human expert labels). Never edited. |
| `interim/` | **Silver (Cleaned & Enriched)** | Cleaned, normalized, and schema-validated extractions (multi-pass extractions, consensus gating, merged expert labels). Regenerated deterministically from Bronze. |
| `processed/` | **Gold (Curated / Consumption)** | Final analysis-ready, manuscript-facing business deliverables (verification matrix, axis-matched SIESTA parity points, `numbers.tex` LaTeX macros, evidence digests). |

### Core Architectural Rules:

- **Unidirectional Lineage (No Upstream Pollution)**: Lower tiers never read from higher tiers (`raw/` never reads `interim/`; `interim/` never reads `processed/`).
- **Deterministic Re-computability**: Both **Silver** (`interim/`) and **Gold** (`processed/`) layers can be wiped and fully re-derived from **Bronze** (`raw/` + `external/`) via the automated pipeline. Bronze is the single, irreplaceable source of truth.

## raw

| path | |
|---|---|
| `corpus/corpus_298_locked.jsonl` | the frozen OpenAlex query the study used, 298 records |
| `fulltext_manifest.tsv` | key, DOI, open-access URL, SHA-256 and length of all 69 retrieved documents |
| `labels_expert/` | the reference standard, as the annotator entered it |
| `labels_model/` | model-generated labels; no reported metric derives from these |
| `prototypes/` | pentagonal starting geometries |
| `reported_values.yaml` | the registry: what the manuscript reports, and how to recompute it |
| `runtime_logs/` | log excerpts backing values no data artefact records |

Article full text is **not** here and is not redistributed. The manifest records which
documents were used; `scripts/fetch_fulltext.py` retrieves and hash-checks them.

## Two populations

`interim/extraction/` holds the pipeline's output over the **69 retrieved papers** — that
is the set behind pass stability and, after exclusions, the 57-paper reporting audit.
`interim/labels_expert_merged.csv` holds the **25 hand-labelled papers, 201 judgements**,
and every accuracy figure in the manuscript is computed against those alone. The two are
not interchangeable: the first says what the pipeline produced, the second says how far
it can be believed.

## interim

`extraction/<configuration>/` holds one directory per hardware configuration —
`rtx3050_q4_0`, `tesla_t4_q4_0`, `tesla_t4_f16`. Within each, `pass_a/b/c.jsonl` are the
individual extraction passes, `union.jsonl` their grounded union, and `union_gated.jsonl`
that union after the numeric value gate. `labels_expert_merged.csv` joins the two label
batches into the 201 scorable judgements every reported metric is computed from.

## processed

Reportability, evaluation and extraction summaries; per-configuration verification
records; `parity_points.jsonl` (the 8 axis-matched lattice comparisons, distilled so the
parity figure needs no SIESTA scratch); `numbers.tex` (the generated macros); and
`verification_report.json`, written by `make values`.

## Figures

Figures are named for content, not for position in the paper, and live in `figures/` at
the repository root — they are publication artefacts rather than a data layer. See
`figures/README.md` for which file is which figure.
