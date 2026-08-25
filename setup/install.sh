#!/usr/bin/env bash
# Fresh Azure GPU VM -> ready to run, in one command.
#
#   bash setup/install.sh            everything
#   bash setup/install.sh 20 30      just those steps
#
# Every step is idempotent: re-running after a failure resumes rather than repeating.
# Steps are numbered in dependency order and each is runnable on its own.

# =============================================================================
#                       ********* SETUP ROUTINE *********                      
#            Infrastructure provisioning and dependency resolution.            
# =============================================================================

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

STEPS=(00-system 10-ollama 20-siesta 30-python 40-verify)

if [ $# -gt 0 ]; then
    WANTED=()
    for arg in "$@"; do
        for s in "${STEPS[@]}"; do
            [[ "$s" == "$arg"* ]] && WANTED+=("$s")
        done
    done
    STEPS=("${WANTED[@]}")
    [ ${#STEPS[@]} -gt 0 ] || die "no step matched: $*"
fi

printf '\n\033[1mACV — setup\033[0m\n'
printf '   repository : %s\n' "$ACV_ROOT"
printf '   prefix     : %s\n' "$OPT_PREFIX"
printf '   steps      : %s\n' "${STEPS[*]}"

START=$SECONDS
for s in "${STEPS[@]}"; do
    bash "$HERE/$s.sh"
done

printf '\n\033[1mdone in %dm%02ds\033[0m\n\n' $(( (SECONDS-START)/60 )) $(( (SECONDS-START)%60 ))
