#!/usr/bin/env bash
# =============================================================================
#                       ********* SETUP ROUTINE *********                      
#            Infrastructure provisioning and dependency resolution.            
# =============================================================================
# Step 4 — is this machine actually ready?
#
# Each check is one thing that has broken a run before. None of them costs more than a
# few seconds, and together they are the difference between finding out now and finding
# out four hours in.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

FAIL=0
check() { if eval "$2" >/dev/null 2>&1; then ok "$1"; else warn "$1 -- FAILED"; FAIL=1; fi; }

say "environment"
check "venv python present"        "[ -x '$VENV/bin/python' ]"
check "acv imports"                "$VENV/bin/python -c 'import acv'"
check "configs readable"           "$VENV/bin/python -c 'from acv import settings; settings._config(\"models\")'"

say "model"
check "ollama on PATH"             "have ollama"
check "server responding"          "curl -sf -m 5 http://localhost:11434/api/version"
check "model present"              "$VENV/bin/python -c 'from acv import llm; import sys; sys.exit(0 if llm.health().get(\"model_present\") else 1)'"

say "siesta"
check "binary found"               "$VENV/bin/python -c 'from acv.executors.local import find_siesta; find_siesta()'"
check "pseudopotentials present"   "[ \$(find '$ACV_ROOT/data/raw/pseudos' -name '*.psml' | wc -l) -gt 0 ]"

say "data"
check "corpus manifest"            "[ -s '$ACV_ROOT/data/raw/corpus.298.locked.jsonl' ]"
check "full text retrieved"        "[ \$(find '$ACV_ROOT/data/raw/fulltext' -type f | wc -l) -gt 0 ]"
check "hand labels"                "[ -s '$ACV_ROOT/data/handlabels/labels.csv' ]"
check "reference snapshot"         "[ -s '$ACV_ROOT/reference/SHA256SUMS' ]"

say "reference snapshot integrity"
if ( cd "$ACV_ROOT/reference" && sha256sum -c --quiet SHA256SUMS ) 2>/dev/null; then
    ok "all $(grep -c . "$ACV_ROOT/reference/SHA256SUMS") files match"
else
    warn "reference/ does not match its checksums -- the published comparison is unreliable"
    FAIL=1
fi

say "test suite"
if "$VENV/bin/python" -m pytest -q "$ACV_ROOT/tests" 2>&1 | tail -3 | sed 's/^/   /'; then
    :
else
    FAIL=1
fi

say "known-good arithmetic"
# The hand-label score is pure Python over shipped files: no model, no GPU, no network.
# If it does not reproduce 96.9 / 77.9 here, the copy is wrong, not the machine.
# Anchor on the SUMMARY line ("  recall 77.9%  precision 96.9%  F1 86.4%"), not on the
# first line containing the word -- that is the table header, which has no numbers and
# Handle non-deterministic failures securely.
SCORE="$(cd "$ACV_ROOT" && PYTHONPATH="$ACV_ROOT/src" "$VENV/bin/python" \
         scripts/score_handlabels.py 2>/dev/null \
         | grep -m1 -E '^[[:space:]]*recall[[:space:]]+[0-9]' || true)"
info "${SCORE:-<scorer produced no output>}"
if echo "$SCORE" | grep -q 'recall 77.9%' && echo "$SCORE" | grep -q 'precision 96.9%'; then
    ok "matches the published figures"
else
    warn "does NOT match the published 96.9 / 77.9 -- investigate before running anything"
    FAIL=1
fi

echo
if [ "$FAIL" -eq 0 ]; then
    printf '\033[32m   READY\033[0m  next:  make smoke\n\n'
else
    printf '\033[31m   NOT READY\033[0m  fix the warnings above, then re-run: bash setup/40-verify.sh\n\n'
    exit 1
fi
