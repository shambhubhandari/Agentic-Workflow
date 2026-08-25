"""Tier 2: translator_crew module.

Provides strict, deterministic logic and strict typing for translator_crew operations.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from ... import llm
from ...flows.state import SiestaParameters
from ..loader import prompt as load_prompt
from ...hooks import audit_log

log = logging.getLogger(__name__)

# Persona and task now live in config/agents.yaml and config/tasks.yaml, beside this
# module. See crews/loader.py for why they are externalised.

# Bounds the agent may not leave. Same band validate.py applies to values extracted from
# papers -- a number the agent invented has earned no more trust than one a paper printed.
from ...settings import effective as _eff          # noqa: E402

MESH_CUTOFF_MIN_RY = float(_eff('tier2', 'translator_gates.mesh_cutoff_ry.min', None, 50.0))
MESH_CUTOFF_MAX_RY = float(_eff('tier2', 'translator_gates.mesh_cutoff_ry.max', None, 2000.0))
MESH_CUTOFF_DEFAULT_RY = float(_eff('tier2', 'translator_gates.mesh_cutoff_ry.default', None, 350.0))

# Map composite XC string to SIESTA split directives (family, parameterisation).
XC_FAMILIES = {"LDA", "GGA", "VDW"}
XC_AUTHORS = {
    "CA", "PZ", "PW92", "PBE", "REVPBE", "RPBE", "WC", "AM05", "PBESOL", "BLYP",
    "DRSLL", "LMKLL", "KBM", "C09", "BH", "VV", "PBEJSJRLO",
}
XC_FAMILY_DEFAULT = "GGA"

# Enforce symmetric in-plane k-grids for near-square penta-cells to prevent density artifacts.
K_INPLANE_MAX_RATIO = int(_eff('tier2', 'translator_gates.k_inplane_max_ratio', None, 2))

# XC.authors expects a parameterisation, never a code name.
XC_AUTHORS_DEFAULT = "PBE"
CODE_NAMES = {
    "VASP", "SIESTA", "QE", "QUANTUM ESPRESSO", "QUANTUM-ESPRESSO", "ESPRESSO",
    "CASTEP", "ABINIT", "CP2K", "GAUSSIAN", "WIEN2K", "CRYSTAL", "OPENMX",
}


def _describe(method: dict[str, Any]) -> str:
    """Render the extracted method parameters as text, marking absences explicitly.

    Absences are stated rather than omitted: "not reported" is information the agent
    needs, and leaving a field out invites it to assume a default.
    """
    lines = []
    for name, entry in sorted(method.items()):
        if not isinstance(entry, dict):
            continue
        if entry.get("reported"):
            value = entry.get("value")
            evidence = (entry.get("evidence") or "").strip()
            lines.append(f"- {name}: {value}" + (f"   [paper: \"{evidence[:120]}\"]" if evidence else ""))
        else:
            lines.append(f"- {name}: NOT REPORTED")
    return "\n".join(lines) if lines else "(no method parameters extracted)"


# =============================================================================
#                      ********* AGENT DISPATCH *********                    
#         Method translation, explicit overrides, and JSON serialization.      
# =============================================================================

def translate(
    method: dict[str, Any],
    formula: str = "",
    n_atoms: int = 0,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> SiestaParameters:
    """Decide SIESTA settings for a paper's method. Never invents silently."""
    instructions, prompt_id = load_prompt("translator", "translator", "translate_method")
    prompt = (
        f"{instructions}\n\nSystem: {formula} ({n_atoms} atoms), a 2D monolayer.\n\n"
        f"{_describe(method)}\n"
    )

    _t0 = time.time()
    params, model_used = llm.structured(
        prompt, SiestaParameters, agent="translator", provider=provider, model=model
    )
    _proposed = params.model_dump()
    _overrides: list[str] = []

    # Enforce strict vacuum sampling constraint (k_z = 1).
    if len(params.kgrid) != 3:
        _overrides.append("kgrid: malformed, reset to 9x9x1")
        log.warning("translator returned kgrid %r; falling back to 9x9x1", params.kgrid)
        params.kgrid = [9, 9, 1]
    if params.kgrid[2] != 1:
        _overrides.append(f"kgrid: k_z={params.kgrid[2]} forced to 1 (2D vacuum axis)")
        log.warning("translator returned k_z=%d for a 2D system; forcing to 1",
                    params.kgrid[2])
        params.kgrid = [params.kgrid[0], params.kgrid[1], 1]
    # Prevent zero-division grids.
    if min(params.kgrid[:2]) < 1:
        _overrides.append(
            f"kgrid: in-plane divisions {params.kgrid[:2]} below 1, reset to 9x9"
        )
        log.warning("translator returned in-plane kgrid %r; falling back to 9x9",
                    params.kgrid[:2])
        params.kgrid = [9, 9, 1]

    # Bound max grid asymmetry.
    kx, ky = params.kgrid[0], params.kgrid[1]
    if max(kx, ky) > K_INPLANE_MAX_RATIO * min(kx, ky):
        balanced = max(kx, ky)
        _overrides.append(
            f"kgrid: in-plane {[kx, ky]} is lopsided by more than {K_INPLANE_MAX_RATIO}x "
            f"on a near-square cell, balanced to {[balanced, balanced]}"
        )
        log.warning("translator returned lopsided in-plane kgrid %r; balancing to %d",
                    [kx, ky], balanced)
        params.kgrid = [balanced, balanced, 1]

    # Bound mesh cutoff values.
    if not (MESH_CUTOFF_MIN_RY <= float(params.mesh_cutoff_ry or 0) <= MESH_CUTOFF_MAX_RY):
        _overrides.append(
            f"mesh_cutoff_ry: {params.mesh_cutoff_ry} outside "
            f"[{MESH_CUTOFF_MIN_RY}, {MESH_CUTOFF_MAX_RY}] Ry, reset to {MESH_CUTOFF_DEFAULT_RY}"
        )
        log.warning("translator returned mesh cutoff %r Ry; falling back to %s Ry",
                    params.mesh_cutoff_ry, MESH_CUTOFF_DEFAULT_RY)
        params.unmapped.append(
            f"proposed mesh cutoff {params.mesh_cutoff_ry} Ry is not a usable value; "
            f"{MESH_CUTOFF_DEFAULT_RY} Ry used instead"
        )
        params.mesh_cutoff_ry = MESH_CUTOFF_DEFAULT_RY

    # Split a combined "GGA/PBE" before anything else looks at the two fields.
    for attr in ("xc", "xc_authors"):
        raw = str(getattr(params, attr) or "").strip()
        if "/" in raw:
            family, _, authors = raw.partition("/")
            family, authors = family.strip().upper(), authors.strip().upper()
            if family in XC_FAMILIES and authors in XC_AUTHORS:
                _overrides.append(f"{attr}: split {raw!r} into {family}/{authors}")
                params.xc, params.xc_authors = family, authors
                break

    if params.xc.strip().upper() not in XC_FAMILIES:
        _overrides.append(
            f"xc: {params.xc!r} is not a SIESTA XC family, reset to {XC_FAMILY_DEFAULT}"
        )
        log.warning("translator gave xc=%r; using %s", params.xc, XC_FAMILY_DEFAULT)
        params.xc = XC_FAMILY_DEFAULT
    if params.xc_authors.strip().upper() not in XC_AUTHORS:
        _overrides.append(
            f"xc_authors: {params.xc_authors!r} is not a SIESTA parameterisation, "
            f"reset to {XC_AUTHORS_DEFAULT}"
        )
        log.warning("translator gave xc_authors=%r; using %s",
                    params.xc_authors, XC_AUTHORS_DEFAULT)
        params.xc_authors = XC_AUTHORS_DEFAULT

    # Sanitize invalid XC.authors parameterisation strings.
    if params.xc_authors.strip().upper() in CODE_NAMES:
        _overrides.append(
            f"xc_authors: {params.xc_authors!r} is a code, not a parameterisation; "
            f"reset to {XC_AUTHORS_DEFAULT}"
        )
        log.warning("translator gave xc_authors=%r (a code name); using %s",
                    params.xc_authors, XC_AUTHORS_DEFAULT)
        params.unmapped.append(
            f"xc_authors {params.xc_authors!r} names a code rather than an "
            f"exchange-correlation parameterisation; {XC_AUTHORS_DEFAULT} used instead"
        )
        params.xc_authors = XC_AUTHORS_DEFAULT

    # Ensure known unmappable parameters are explicitly logged.
    def _reported(name: str):
        entry = method.get(name)
        return entry if isinstance(entry, dict) and entry.get("reported") else None

    functional = str((_reported("xc_functional") or {}).get("value") or "")
    for hybrid in ("HSE", "PBE0", "B3LYP", "SCAN"):
        if hybrid in functional.upper() and not any(
            hybrid.lower() in u.lower() for u in params.unmapped
        ):
            _overrides.append(f"unmapped: agent omitted {hybrid}")
            params.unmapped.append(
                f"paper used {hybrid}; SIESTA cannot reproduce it in a standard run, "
                f"substituted {params.xc}/{params.xc_authors}"
            )
    if _reported("plane_wave_cutoff_ev") and not any(
        "plane" in u.lower() or "cutoff" in u.lower() for u in params.unmapped
    ):
        pw = _reported("plane_wave_cutoff_ev")["value"]
        _overrides.append("unmapped: agent omitted the plane-wave cutoff")
        params.unmapped.append(
            f"paper's plane-wave cutoff ({pw} eV) has no NAO equivalent; "
            f"mesh cutoff {params.mesh_cutoff_ry} Ry chosen independently"
        )
    if _reported("dispersion_correction") and not any(
        "disp" in u.lower() or "d3" in u.lower() for u in params.unmapped
    ):
        _overrides.append("unmapped: agent omitted the dispersion correction")
        params.unmapped.append(
            f"paper used dispersion correction "
            f"({_reported('dispersion_correction')['value']}); not applied here"
        )

    if params.basis.upper() not in {"SZ", "SZP", "DZ", "DZP", "TZP"}:
        _overrides.append(f"basis: {params.basis!r} not on the ladder, reset to DZP")
        log.warning("unknown basis %r from translator; falling back to DZP", params.basis)
        params.unmapped.append(f"basis {params.basis!r} not recognised, used DZP")
        params.basis = "DZP"

    audit_log.record(
        "translator", prompt_id, model=model_used,
        prompt_tail=_describe(method), response=_proposed,
        proposed=_proposed, applied=params.model_dump(), overrides=_overrides,
        seconds=time.time() - _t0,
        context={"formula": formula, "n_atoms": n_atoms},
    )
    return params
