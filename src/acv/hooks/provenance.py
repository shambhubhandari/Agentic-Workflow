"""System: provenance module.

Provides strict, deterministic logic and strict typing for provenance operations.
"""
from __future__ import annotations

# =============================================================================
#                      ********* LIFECYCLE HOOKS *********                     
#                      Strict definitions for provenance.                      
# =============================================================================

import hashlib
import json
import logging
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..settings import PROCESSED

log = logging.getLogger(__name__)

MANIFEST_DIR = PROCESSED / "provenance"


def file_digest(path: Path) -> Optional[str]:
    """SHA-256 of a file, or None if absent."""
    if not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def siesta_version(binary: Path) -> str:
    """Version string of the executed SIESTA binary."""
    try:
        out = subprocess.run(
            [str(binary), "--version"], capture_output=True, timeout=30, text=True
        )
        for line in (out.stdout or "").splitlines():
            if line.strip().startswith("Version"):
                return line.split(":", 1)[1].strip()
    except Exception as exc:               # noqa: BLE001 - provenance must never crash a run
        log.warning("could not read SIESTA version: %s", exc)
    return "unknown"


@dataclass
class RunManifest:
    """Everything needed to repeat one calculation."""

    label: str
    created_at: str
    kind: str = "siesta"

    # what was run
    code: str = ""
    code_version: str = ""
    host: str = ""

    # on what
    formula: str = ""
    n_atoms: int = 0
    cell_in: Optional[list] = None
    structure_source: Optional[str] = None
    structure_digest: Optional[str] = None

    # with what
    parameters: dict[str, Any] = field(default_factory=dict)
    pseudopotentials: dict[str, str] = field(default_factory=dict)
    input_digest: Optional[str] = None

    # results
    converged: bool = False
    energy_ev: Optional[float] = None
    cell_out: Optional[list] = None
    max_force: Optional[float] = None
    seconds: Optional[float] = None

    def content_hash(self) -> str:
        """Hash of the inputs only, so identical inputs collide and results do not."""
        payload = {
            "code": self.code, "code_version": self.code_version,
            "formula": self.formula, "n_atoms": self.n_atoms,
            "cell_in": self.cell_in, "parameters": self.parameters,
            "pseudopotentials": self.pseudopotentials,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def record(
    result,
    atoms=None,
    structure_source: Optional[str] = None,
    binary: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> RunManifest:
    """Build and persist a manifest for a completed SIESTA run."""
    from ..executors.local import find_siesta

    out_dir = Path(out_dir or MANIFEST_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = binary or find_siesta()
    workdir = Path(result.workdir)

    pseudos = {
        path.stem: file_digest(path) or ""
        for path in sorted(workdir.glob("*.psml"))
    }

    manifest = RunManifest(
        label=result.label,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        code=str(binary),
        code_version=siesta_version(binary),
        host=platform.node(),
        formula=atoms.get_chemical_formula() if atoms is not None else "",
        n_atoms=len(atoms) if atoms is not None else 0,
        cell_in=[[round(float(v), 6) for v in row] for row in atoms.cell] if atoms is not None else None,
        structure_source=structure_source,
        structure_digest=file_digest(Path(structure_source)) if structure_source else None,
        parameters=dict(result.params),
        pseudopotentials=pseudos,
        input_digest=file_digest(workdir / f"{result.label}.fdf"),
        converged=bool(result.converged),
        energy_ev=result.energy_ev,
        max_force=result.max_force,
        seconds=result.seconds,
    )

    # Final cell, parsed from SIESTA's own report rather than assumed.
    out_file = workdir / f"{result.label}.out"
    if out_file.exists():
        for line in out_file.read_text(errors="replace").splitlines():
            if "outcell: Cell vector modules" in line:
                # The line carries two colons: "outcell: Cell vector modules (Ang) : a b c"
                try:
                    manifest.cell_out = [float(x) for x in line.rsplit(":", 1)[1].split()]
                except (ValueError, IndexError):
                    pass

    path = out_dir / f"{result.label}.{manifest.content_hash()}.json"
    path.write_text(json.dumps(asdict(manifest), indent=2, default=str), encoding="utf-8")
    log.info("provenance -> %s", path.name)
    return manifest


def load_all(out_dir: Optional[Path] = None) -> list[dict]:
    out_dir = Path(out_dir or MANIFEST_DIR)
    if not out_dir.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
