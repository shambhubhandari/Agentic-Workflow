"""Tier 0: SIESTA AST metadata extraction and compilation.

Executes a recursive file scan across the SIESTA 5.4.2 Fortran source 
tree to compile a JSON index of configuration directives, types, and 
units for downstream generative validation.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

# =============================================================================
#                      ********* REGEX PATTERNS *********                      
#         Capture groups for standard FDF_GET Fortran function calls.          
# =============================================================================

CALL = re.compile(
    r"fdf_(get|string|integer|single|double|boolean|physical|block|islist|isblock|deprecated)"
    r"\s*\(\s*"
    r"['\"]([^'\"]+)['\"]"
    r"(?:\s*,\s*([^,()]+(?:\([^)]*\))?))?"
    r"(?:\s*,\s*['\"]([^'\"]+)['\"])?",
    re.IGNORECASE,
)

# =============================================================================
#                     ********* DEFAULT INFERENCE *********                    
#          Cast extracted Fortran literals into strict Python datatypes.       
# =============================================================================

def _clean_default(raw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if raw is None:
        return None, None
    d = raw.strip()
    if d.lower() in (".true.", ".false."):
        return d.lower().strip("."), "logical"
    d = re.sub(r"_dp\b|_sp\b", "", d)
    if re.fullmatch(r"-?\d+", d):
        return d, "integer"
    if re.fullmatch(r"-?\d*\.?\d*([eEdD][-+]?\d+)?", d) and any(c.isdigit() for c in d):
        return d.rstrip("."), "real"
    if d.startswith(("'", '"')):
        return d.strip("'\""), "string"
    return d, "expression"

# =============================================================================
#                    ********* INDEX CONSTRUCTION *********                    
#       AST compilation routines outputting deterministic JSON vocabularies.   
# =============================================================================

def build(source_root: Path) -> dict[str, Any]:
    src = source_root / "Src"
    if not src.exists():
        raise SystemExit(f"no Src/ under {source_root}")

    index: dict[str, dict[str, Any]] = {}
    patterns = ("*.F90", "*.f90", "*.F", "*.f", "*.T90", "*.inc")
    files = [p for pat in patterns for p in src.rglob(pat)]
    for path in files:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for kind, name, default, unit in CALL.findall(text):
            value, vtype = _clean_default(default)
            k = kind.lower()
            entry = index.setdefault(
                name,
                {
                    "name": name,
                    "kind": "block" if "block" in k else "value",
                    "type": vtype,
                    "default": value,
                    "unit": unit or None,
                    "sources": [],
                },
            )
            if entry["default"] is None and value is not None:
                entry["default"], entry["type"] = value, vtype
            if entry["unit"] is None and unit:
                entry["unit"] = unit
            if entry["type"] is None:
                entry["type"] = {
                    "string": "string", "integer": "integer", "boolean": "logical",
                    "double": "real", "single": "real", "physical": "real",
                }.get(k)
            rel = str(path.relative_to(source_root))
            if rel not in entry["sources"]:
                entry["sources"].append(rel)

    for entry in index.values():
        entry["sources"] = entry["sources"][:3]

    return {
        "siesta_version": source_root.name.replace("siesta-", ""),
        "n_directives": len(index),
        "directives": dict(sorted(index.items(), key=lambda kv: kv[0].lower())),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    data = build(root)

    out = Path(__file__).resolve().parent / "siesta_directives.json"
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"{data['n_directives']} directives from SIESTA {data['siesta_version']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
