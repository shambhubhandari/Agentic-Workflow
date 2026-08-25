# Configuration — Single Source of Truth (SSOT)

Every tuned parameter, physical threshold, model hyperparameter, and execution rule across the entire study is externalized in this directory. 

### Core Design Principle: Zero Hardcoded Magic Constants
The Python pipeline strictly enforces the **Single Source of Truth (SSOT)** pattern:
- **No buried hyperparameters**: Code modules in `src/acv/` consume settings directly from these YAML files.
- **Audit transparency**: Changing an extraction threshold, k-point mesh density, or model context window requires an explicit configuration diff rather than code edits.
- **Epistemic boundary enforcement**: Physical limits and convergence ladders are centralized for automated verification.

---

### Configuration Registry

| Configuration File | Scope & Role | Tuned Parameters & Policies |
|:---|:---|:---|
| **`corpus.yaml`** | **Corpus Assembly & Filtering** | OpenAlex query keywords, concept filters, language restrictions, and title/abstract deduplication rules. |
| **`models.yaml`** | **Inference & Quantization** | Model tag (`qwen3:4b`), temperature ($T = 0.0$), `num_predict`, context window bucketing ($4\text{k} \rightarrow 32\text{k}$ tokens), and per-agent parameter overrides. |
| **`tier0.yaml`** | **Extraction & Reportability** | Prompt schema version (`extract_method_v2`), full-text section slicing, evidence quote requirements, and public OPTIMADE cross-reference providers. |
| **`tier2.yaml`** | **SIESTA DFT Recomputation** | Physical convergence ladders (mesh cutoff energy, Monkhorst-Pack $k$-grid), relaxation force tolerances ($F_{\text{tol}}$), SCF thresholds, and timeout/memory execution limits. |

---

### Model Tag Pinning Note
`models.yaml` specifies models by tag. Because registry tags can occasionally be updated upstream, reproduction pipelines requiring bit-level weight reproducibility can pin the Ollama model digest directly in `models.yaml`.
