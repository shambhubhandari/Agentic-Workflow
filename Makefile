# =============================================================================
#              ********* ACV — REPRODUCIBILITY AUDITING *********
#          Pentagonal 2D materials manuscript pipeline orchestrator.
# =============================================================================
#
#   make env        create venv and install dependencies              <- first time
#   make values     recompute every reported value, PASS/FAIL each   <- start here
#   make check      which resources this machine has
#   make help       everything

SRC := PYTHONPATH=src
VENV := .venv

# After `make env`, every target uses the venv python directly.
PY := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)

.DEFAULT_GOAL := help
.PHONY: help env values labels numbers test lint figures evidence \
        check fulltext corpus fetch extract audit evaluate recompute clean

help:  ## show this help
	@grep -hE '^[a-z][a-z0-9_ -]*:.*?##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  %-11s %s\n", $$1, $$2}'

# =============================================================================
#                       ********* ENVIRONMENT *********
#             Virtual environment creation and dependency resolution.
# =============================================================================

env:  ## create .venv and install all dependencies (auto-detects python 3.11+)
	@FOUND=""; \
	for candidate in python3.13 python3.12 python3.11 python3; do \
	    if command -v $$candidate >/dev/null 2>&1; then \
	        VER=$$($$candidate -c "import sys; print(sys.version_info[:2])" 2>/dev/null); \
	        MAJ=$$(echo $$VER | tr -d '(,' | awk '{print $$1}'); \
	        MIN=$$(echo $$VER | tr -d '),' | awk '{print $$2}'); \
	        if [ "$$MAJ" -ge 3 ] 2>/dev/null && [ "$$MIN" -ge 11 ] 2>/dev/null; then \
	            FOUND=$$candidate; break; \
	        fi; \
	    fi; \
	done; \
	if [ -z "$$FOUND" ]; then \
	    echo ""; \
	    echo "  ERROR: no python >= 3.11 found on PATH."; \
	    echo "  install one and retry, or override:  PY_CREATE=/path/to/python3.12 make env"; \
	    echo ""; \
	    exit 1; \
	fi; \
	echo "  using $$FOUND ($$($$FOUND --version 2>&1))"; \
	test -d $(VENV) || $$FOUND -m venv $(VENV); \
	$(VENV)/bin/pip install --upgrade pip -q; \
	$(VENV)/bin/pip install -e ".[dev,verify,notebooks]" -q; \
	echo ""; \
	echo "  environment ready. run:"; \
	echo ""; \
	echo "    source $(VENV)/bin/activate"; \
	echo "    make values"; \
	echo ""

notebook:  ## launch Jupyter in notebooks/
	@$(PY) -m jupyter notebook --notebook-dir=notebooks
# =============================================================================
#                    ********* AUDIT (OFFLINE) *********
#           Deterministic recomputation from shipped artefacts only.
# =============================================================================

values:  ## recompute every reported value; non-zero exit on any failure
	@$(SRC) $(PY) scripts/verify.py --json

labels:  ## precision/recall of the shipped extraction against the expert labels
	@$(SRC) $(PY) scripts/score_labels.py

numbers:  ## regenerate data/processed/numbers.tex from the artefacts
	@$(SRC) $(PY) scripts/manuscript_numbers.py

test:  ## unit tests
	@$(SRC) $(PY) -m pytest -q tests/

lint:  ## byte-compile every module
	@$(SRC) $(PY) -m compileall -q src scripts && echo "  ok"

figures:  ## regenerate the figures into figures/
	@$(SRC) $(PY) -m acv.visualization

evidence:  ## regenerate the log-derived reports into data/processed/evidence/
	@$(SRC) $(PY) scripts/evidence/offload_table.py \
	   data/raw/runtime_logs/ollama_layer_offload.txt > data/processed/evidence/layer_residency.txt
	@for p in ab ac bc; do \
	   l=$$(echo $$p | cut -c1); r=$$(echo $$p | cut -c2); \
	   $(SRC) $(PY) scripts/evidence/pass_agreement.py \
	     data/interim/extraction/rtx3050_q4_0/pass_$$l.jsonl \
	     data/interim/extraction/rtx3050_q4_0/pass_$$r.jsonl \
	     > data/processed/evidence/pass_agreement_$$p.txt; done
	@$(SRC) $(PY) scripts/evidence/frontier.py > data/processed/evidence/memory_time_frontier.txt
	@$(SRC) $(PY) scripts/evidence/target_funnel.py > data/processed/evidence/target_funnel.txt
	@echo "  regenerated $$(ls data/processed/evidence | wc -l) reports"

# =============================================================================
#                ********* RE-RUNNING (NEEDS RESOURCES) *********
#        Targets requiring model server, article text, or SIESTA binary.
# =============================================================================

check:  ## report which resources are present: model, SIESTA, article text
	@$(SRC) $(PY) -c "from acv import llm; h=llm.health(); \
	  [print(f'  model    {k:<15} {v}') for k,v in h.items()]" 2>/dev/null || \
	  echo "  model    not reachable"
	@$(SRC) $(PY) -c "from acv.executors.local import find_siesta; \
	  print('  siesta   binary          ', find_siesta())" 2>/dev/null || \
	  echo "  siesta   not found"
	@$(PY) scripts/fetch_fulltext.py --check || true

fulltext:  ## fetch the article text the manifest lists as missing
	@$(PY) scripts/fetch_fulltext.py

corpus:  ## rebuild the corpus from OpenAlex
	@$(SRC) $(PY) -m acv.cli -v corpus

fetch extract audit evaluate:  ## individual pipeline stages
	@$(SRC) $(PY) -m acv.cli -v $@

recompute:  ## Tier 2 — translate, run SIESTA, adjudicate
	@$(SRC) $(PY) -m acv.cli -v verify --execute

# =============================================================================
#                       ********* MAINTENANCE *********
#                    Cache invalidation and workspace cleanup.
# =============================================================================

clean:  ## remove caches
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null; true
	@rm -rf .pytest_cache && echo "  cleaned"
