"""System: inspect module.

Provides strict, deterministic logic and strict typing for inspect operations.
"""
from __future__ import annotations

# =============================================================================
#                        ********* AGENT TOOLS *********                       
#                        Strict definitions for inspect.                       
# =============================================================================

import logging
from pathlib import Path
from .registry import register

log = logging.getLogger(__name__)


@register(
    "read_structure",
    "Read a structure file and report its cell, composition, layer thickness and "
    "in-plane axis ratio b/a.",
    {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "path to a .vasp/.cif"}},
        "required": ["path"],
    },
)
def read_structure(path: str) -> dict:
    from ase.io import read

    p = Path(path)
    if not p.exists():
        return {"error": f"no such file: {path}"}
    # ase.io.read is typed as possibly returning a list; index=-1 always returns one
    # Atoms object, but the type checker cannot see that.
    atoms = read(str(p), index=-1)
    if isinstance(atoms, list):             # defensive; never taken with index=-1
        atoms = atoms[-1]
    lengths = sorted(float(x) for x in atoms.cell.lengths())
    vac_axis = int(max(range(3), key=lambda i: atoms.cell.lengths()[i]))
    z = sorted(float(pos[vac_axis]) for pos in atoms.get_positions())
    gaps = [b - a for a, b in zip(z, z[1:])] + [lengths[2] - z[-1] + z[0]]
    return {
        "formula": atoms.get_chemical_formula(),
        "n_atoms": len(atoms),
        "a": round(lengths[0], 4),
        "b": round(lengths[1], 4),
        "vacuum_axis_length": round(lengths[2], 4),
        "b_over_a": round(lengths[1] / lengths[0], 4) if lengths[0] else None,
        "layer_thickness": round(lengths[2] - max(gaps), 3),
    }


@register(
    "check_symmetry",
    "Check whether a relaxed cell still matches its structural prototype. A tetragonal "
    "prototype must keep b/a near 1; leaving it means a different structure was relaxed.",
    {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
            "prototype": {
                "type": "string",
                "description": "tetragonal | orthorhombic | unknown",
            },
            "tolerance": {"type": "number", "description": "fractional, default 0.02"},
        },
        "required": ["a", "b", "prototype"],
    },
)
def check_symmetry(a: float, b: float, prototype: str, tolerance: float = 0.02) -> dict:
    if not a:
        return {"error": "a must be non-zero"}
    ratio = b / a
    if prototype.lower() == "tetragonal":
        ok = abs(ratio - 1.0) <= tolerance
        return {
            "b_over_a": round(ratio, 4),
            "prototype": prototype,
            "consistent": ok,
            "note": (
                "cell retains tetragonal symmetry" if ok else
                f"b/a = {ratio:.3f} departs from tetragonal by "
                f"{100*abs(ratio-1):.1f}%; the relaxation has left the prototype and the "
                "comparison may be against a different structure"
            ),
        }
    return {"b_over_a": round(ratio, 4), "prototype": prototype, "consistent": None,
            "note": "no symmetry constraint known for this prototype"}


@register(
    "convert_units",
    "Convert a 2D elastic constant between N/m and GPa. The GPa value depends on an "
    "assumed thickness, so the conversion is only meaningful with one stated.",
    {
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "from_unit": {"type": "string", "description": "N/m or GPa"},
            "thickness_ang": {"type": "number"},
        },
        "required": ["value", "from_unit", "thickness_ang"],
    },
)
def convert_units(value: float, from_unit: str, thickness_ang: float) -> dict:
    if thickness_ang <= 0:
        return {"error": "thickness must be positive; without it the conversion is undefined"}
    u = from_unit.strip().lower().replace(" ", "")
    if u in ("n/m", "nm-1", "nperm"):
        return {"value_gpa": round(value / (thickness_ang * 0.1), 3),
                "thickness_ang": thickness_ang, "note": "GPa depends on the thickness used"}
    if u == "gpa":
        return {"value_n_per_m": round(value * thickness_ang * 0.1, 3),
                "thickness_ang": thickness_ang}
    return {"error": f"unknown unit {from_unit!r}"}


@register(
    "calibration_for_structure",
    "Return the calibrated SIESTA-vs-plane-wave lattice threshold. The calibration is "
    "measured on pentagonal structures only; returns an error, not a number, when it is "
    "not usable.",
    {"type": "object", "properties": {}, "required": []},
)
def calibration_for_structure() -> dict:
    """The penta lattice threshold, or an explicit refusal.

    Took a `layer_thickness_ang` argument until the calibration set was restricted to the
    penta prototype. It selected between a `flat` and a `puckered` class, but classified
    on thickness alone, so hexagonal TMDs sat in the `puckered` class and set the
    threshold actually applied to penta targets. With a penta-only set there is one
    population and nothing to select between.

    Returning an error rather than a permissive default is the point: a fabricated
    threshold turns into a published verdict.
    """
    import json

    from ..settings import PROCESSED

    path = PROCESSED / "calibration.json"
    if not path.exists():
        return {"error": "no calibration available; verdicts must be INCONCLUSIVE"}

    data = json.loads(path.read_text())
    lattice = data.get("lattice") or {}
    threshold = lattice.get("p90_abs_offset")
    if data.get("usable") is False or threshold is None:
        return {
            "error": "calibration not usable; verdicts must be INCONCLUSIVE",
            "n_points": data.get("n_points"),
            "distinct_structures": data.get("distinct_structures"),
            "reason": data.get("note"),
        }
    return {
        "threshold_pct": round(100 * threshold, 3),
        "n_values": lattice.get("n_values"),
        "n_structures": data.get("n_usable"),
        "note": "measured on pentagonal structures only",
    }


@register(
    "search_output",
    "Search a SIESTA output file for lines matching a pattern. Use this to verify a "
    "diagnosis against the actual log rather than a supplied excerpt.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "pattern": {"type": "string", "description": "case-insensitive substring"},
            "max_hits": {"type": "integer"},
        },
        "required": ["path", "pattern"],
    },
)
def search_output(path: str, pattern: str, max_hits: int = 20) -> dict:
    p = Path(path)
    if not p.exists():
        return {"error": f"no such file: {path}"}
    needle = pattern.lower()
    hits = [
        {"line_no": i, "text": line.rstrip()[:200]}
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1)
        if needle in line.lower()
    ]
    return {"n_hits": len(hits), "hits": hits[:max_hits]}


@register(
    "machine_resources",
    "Current free RAM, free disk and physical core count. Use to confirm or rule out a "
    "resource-related failure.",
    {"type": "object", "properties": {}},
)
def machine_resources() -> dict:
    from ..guardrails.resources import snapshot

    r = snapshot()
    return {
        "free_ram_gb": r.free_ram_gb,
        "free_disk_gb": r.free_disk_gb,
        "physical_cores": r.physical_cores,
    }


@register(
    "list_pseudopotentials",
    "List the chemical elements for which a pseudopotential is installed.",
    {"type": "object", "properties": {}},
)
def list_pseudopotentials() -> dict:
    from ..settings import PSEUDO_DIR

    if not PSEUDO_DIR.exists():
        return {"error": f"pseudopotential directory missing: {PSEUDO_DIR}"}
    elements = sorted(p.stem for p in PSEUDO_DIR.glob("*.psml"))
    return {"n": len(elements), "elements": elements}


@register(
    "check_basis_supported",
    "Check that a basis label is on the supported ladder (SZ < SZP < DZ < DZP < TZP).",
    {
        "type": "object",
        "properties": {"basis": {"type": "string"}},
        "required": ["basis"],
    },
)
def check_basis_supported(basis: str) -> dict:
    ladder = ["SZ", "SZP", "DZ", "DZP", "TZP"]
    b = (basis or "").strip().upper()
    return {
        "basis": b,
        "supported": b in ladder,
        "ladder": ladder,
        "rank": ladder.index(b) if b in ladder else None,
    }


@register(
    "convergence_status",
    "Apply the arithmetic convergence test to a run history. Returns whether two "
    "successive settings agree within tolerance, and why not if they do not.",
    {
        "type": "object",
        "properties": {
            "energies_ev": {"type": "array", "items": {"type": "number"}},
            "cells_a": {"type": "array", "items": {"type": "number"}},
            "n_atoms": {"type": "integer"},
        },
        "required": ["energies_ev", "cells_a", "n_atoms"],
    },
)
def convergence_status(energies_ev: list, cells_a: list, n_atoms: int) -> dict:
    from ..crews.convergence.convergence_crew import (
        CELL_TOL_PCT,
        ENERGY_TOL_EV_PER_ATOM,
    )

    if len(energies_ev) < 2 or len(cells_a) < 2:
        return {"converged": False,
                "reason": f"only {len(energies_ev)} calculation(s); at least 2 required"}
    d_e = abs(energies_ev[-1] - energies_ev[-2]) / max(n_atoms, 1)
    d_c = 100 * abs(cells_a[-1] - cells_a[-2]) / cells_a[-2] if cells_a[-2] else None
    ok = d_e <= ENERGY_TOL_EV_PER_ATOM and (d_c is not None and d_c <= CELL_TOL_PCT)
    return {
        "converged": ok,
        "delta_energy_mev_per_atom": round(1000 * d_e, 2),
        "delta_cell_pct": round(d_c, 3) if d_c is not None else None,
        "tolerances": {"energy_mev_per_atom": 1000 * ENERGY_TOL_EV_PER_ATOM,
                       "cell_pct": CELL_TOL_PCT},
    }


@register(
    "estimate_cost",
    "Estimate the wall time and memory of a proposed SIESTA calculation before running "
    "it, so a step that cannot fit is not attempted.",
    {
        "type": "object",
        "properties": {
            "n_atoms": {"type": "integer"},
            "basis": {"type": "string"},
            "mesh_cutoff_ry": {"type": "number"},
        },
        "required": ["n_atoms", "basis"],
    },
)
def estimate_cost(n_atoms: int, basis: str, mesh_cutoff_ry: float = 350.0) -> dict:
    from ..guardrails.resources import estimate_ram_gb, snapshot

    weight = {"SZ": 0.4, "SZP": 0.6, "DZ": 0.8, "DZP": 1.0, "TZP": 1.8}
    factor = weight.get((basis or "DZP").upper(), 1.0) * (mesh_cutoff_ry / 350.0) ** 0.5
    # ~95 s for a 6-atom DZP/350 Ry cell on 6 ranks, scaling ~N^2.5 in practice
    seconds = 95.0 * factor * (max(n_atoms, 1) / 6.0) ** 2.5
    ram = estimate_ram_gb(n_atoms, 1, basis)
    free = snapshot()
    return {
        "estimated_seconds": round(seconds),
        "estimated_ram_gb": ram,
        "fits_in_memory": ram < free.free_ram_gb - 2.0,
        "free_ram_gb": free.free_ram_gb,
    }


@register(
    "expected_symmetry",
    "Look up the structural prototype for a composition and the in-plane symmetry a "
    "relaxed cell must retain. Use this instead of assuming; a composition with no "
    "recorded prototype returns unknown rather than a guess.",
    {
        "type": "object",
        "properties": {"reduced_formula": {"type": "string"}},
        "required": ["reduced_formula"],
    },
)
def expected_symmetry(reduced_formula: str) -> dict:
    from ..knowledge import prototypes

    return prototypes.expected_symmetry(reduced_formula)


@register(
    "literature_methods",
    "What computational settings have OTHER papers in the corpus used for this "
    "composition? Returns per-parameter distributions with counts, not a single "
    "recommendation.",
    {
        "type": "object",
        "properties": {"reduced_formula": {"type": "string"}},
        "required": ["reduced_formula"],
    },
)
def literature_methods(reduced_formula: str) -> dict:
    from ..knowledge import literature

    return literature.methods_for(reduced_formula)


@register(
    "literature_claims",
    "Every published value for a composition in the corpus, with source and evidence, "
    "plus the spread across papers. Use to see whether the field agrees with itself "
    "before judging one paper against it.",
    {
        "type": "object",
        "properties": {
            "reduced_formula": {"type": "string"},
            "property_kind": {"type": "string", "description": "e.g. lattice"},
        },
        "required": ["reduced_formula"],
    },
)
def literature_claims(reduced_formula: str, property_kind: str = "") -> dict:
    from ..knowledge import literature

    return literature.claims_for(reduced_formula, property_kind or None)


@register(
    "siesta_directive",
    "Look up a SIESTA fdf directive in the index built from our binary's own source: "
    "does it exist in this version, what is its default, what unit does it expect. "
    "Use before proposing any parameter, because one SIESTA does not read is silently "
    "ignored rather than rejected.",
    {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
)
def siesta_directive(name: str) -> dict:
    from ..knowledge import siesta

    return siesta.lookup(name)


@register(
    "siesta_search",
    "Search SIESTA directives by name fragment when the exact spelling is unknown.",
    {
        "type": "object",
        "properties": {
            "fragment": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["fragment"],
    },
)
def siesta_search(fragment: str, limit: int = 20) -> dict:
    from ..knowledge import siesta

    return siesta.search(fragment, limit=limit)
