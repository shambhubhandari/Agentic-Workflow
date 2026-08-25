"""Knowledge sources the agents can consult.

Three kinds, deliberately separated:

  prototypes   STRUCTURAL — what a pentagonal prototype is and what symmetry it must
               retain. Curated, small, and the thing that makes a symmetry check
               possible without a human supplying the answer.
  literature   EMPIRICAL — what the corpus we mined actually reports. Derived from
               Tier 0 output, so it grows as the audit grows.
  (memory)     LEARNED — calibration offsets and converged settings, in acv.memory,
               written by the pipeline itself.

Knowledge is exposed to agents as tools, not injected into prompts. An agent that must
ask for a fact leaves a record of having asked; a fact pasted into a prompt is
indistinguishable from something the model made up.
"""

# =============================================================================
#                      ********* KNOWLEDGE BASES *********                     
#                       Strict definitions for __init__.                       
# =============================================================================

from . import literature, prototypes, siesta

__all__ = ["literature", "prototypes", "siesta"]
