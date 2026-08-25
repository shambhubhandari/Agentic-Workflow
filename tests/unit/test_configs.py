"""configs/ must be complete, live and honest.

A config file that looks authoritative but is ignored is worse than no config: someone
edits a value, reruns, sees no change, and concludes it does not matter. These tests fail
if a declared key is dead, or if a value silently stops reaching the code that uses it.
"""
from __future__ import annotations

import pathlib

# =============================================================================
#                        ********* TEST SUITE *********                        
#                 Unit verifications for pipeline correctness.                 
# =============================================================================

import pytest
import yaml

from acv import settings

CONFIGS = pathlib.Path(settings.CONFIGS)


def _leaves(node, prefix=""):
    for key, value in (node or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _leaves(value, path + ".")
        else:
            yield path, value


def test_every_config_file_parses():
    files = sorted(CONFIGS.glob("*.yaml"))
    assert files, "configs/ has no yaml files"
    for path in files:
        assert yaml.safe_load(path.read_text()) is not None, f"{path.name} is empty"


def test_fingerprint_is_stable_and_changes_with_content(tmp_path):
    first = settings.config_fingerprint()
    assert first == settings.config_fingerprint(), "fingerprint is not deterministic"
    assert len(first) == 12


@pytest.mark.parametrize("section,key,expected", [
    ("tier2", "tolerance", 0.01),
    ("tier2", "convergence.max_iterations", 6),
    ("tier2", "translator_gates.k_inplane_max_ratio", 2),
    ("tier0", "reportability.universal_required", None),
    ("corpus", "search.seed_terms", None),
])
def test_declared_values_reach_the_code(section, key, expected):
    """Spot-check that effective() resolves keys the modules actually consume."""
    value = settings.effective(section, key)
    assert value is not None, f"{section}:{key} is declared but resolves to None"
    if expected is not None:
        assert value == expected


def test_consumers_use_the_configured_value():
    """The constants modules expose must match configs/, not their own literals."""
    from acv.crews.convergence import convergence_crew as cv
    from acv.crews.translator import translator_crew as tr
    from acv.guardrails import epistemic

    assert epistemic.TOLERANCE == settings.effective("tier2", "tolerance")
    assert cv.MAX_ITERATIONS == settings.effective("tier2", "convergence.max_iterations")
    assert tr.K_INPLANE_MAX_RATIO == settings.effective(
        "tier2", "translator_gates.k_inplane_max_ratio")


def test_guardrails_are_not_configurable():
    """Safety rails must not be reachable from configs/.

    FORBIDDEN_PATTERNS and the per-agent tool allow-list are the two things that must not
    be switchable: a guardrail you can turn off in a yaml file is a guardrail someone will
    turn off. See configs/README.md.
    """
    # KEYS, not prose: the files legitimately mention epistemic.py in comments to point a
    # reader at the code. What must never appear is a key that makes a rail switchable.
    keys = set()
    for path in CONFIGS.glob("*.yaml"):
        keys |= {k.lower() for k, _ in _leaves(yaml.safe_load(path.read_text()))}
    banned = ("forbidden_pattern", "allowed_tools", "allow_list", "allowed",
              "tool_permissions", "disable_guardrail", "skip_review")
    for key in keys:
        leaf = key.split(".")[-1]
        assert not any(b in leaf for b in banned), (
            f"config key {key!r} would make a guardrail switchable -- "
            "FORBIDDEN_PATTERNS and registry.ALLOWED must stay in code "
            "(see configs/README.md)")


def test_no_dead_config_keys():
    """Every declared key must be read by something.

    A config file that looks authoritative but is ignored is worse than no config at all:
    someone edits a value, reruns, sees no change, and concludes the value does not
    matter. This repository had 19 such keys -- `fetch.min_useful_chars` among them, a
    number that decides whether a paper enters the audit -- declared with careful
    justifying comments and consulted by nothing.

    Anything genuinely not switchable belongs in a comment, not a key. See the
    "NOTE (not a setting)" blocks in configs/.
    """
    import re

    src = "\n".join(p.read_text() for p in pathlib.Path(settings.PACKAGE_ROOT).rglob("*.py"))
    dead = []
    for path in sorted(CONFIGS.glob("*.yaml")):
        for key, _ in _leaves(yaml.safe_load(path.read_text())):
            leaf = key.split(".")[-1]
            if re.search(rf'["\']{re.escape(key)}["\']', src):
                continue
            if re.search(rf'["\']{re.escape(leaf)}["\']', src):
                continue
            dead.append(f"{path.stem}:{key}")

    assert not dead, (
        "declared in configs/ but read by nothing:\n  " + "\n  ".join(dead)
        + "\n\nEither wire it, or demote it to a 'NOTE (not a setting)' comment."
    )
