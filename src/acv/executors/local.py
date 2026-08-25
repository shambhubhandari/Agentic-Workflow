"""System: local module.

Provides strict, deterministic logic and strict typing for local operations.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..guardrails import resources as guard
from ..settings import (LOCAL_MPI_RANKS, MPIRUN, PROJECT_ROOT, PSEUDO_DIR,
                        SIESTA_CANDIDATES)

log = logging.getLogger(__name__)

# conda-forge siesta 5.4.2. Serial build: fine for cells this size, and the resource
# guardrail refuses anything large enough to need MPI.
# =============================================================================
#                    ********* CONSTANTS & THRESHOLDS *********                  
#         Filesystem paths and cleanup exclusions for SIESTA execution.        
# =============================================================================

RUN_ROOT = PROJECT_ROOT / "data" / "interim" / "siesta_runs"

# Files SIESTA writes that are large and reconstructible. Deleted after parsing unless
# explicitly kept -- with ~50 GB free, a few hundred runs would otherwise fill the disk.
BULKY_SUFFIXES = (".DM", ".HSX", ".TSDE", ".TSHS", ".RHO", ".VT", ".VH", ".LWF", ".PSF")


@dataclass
class SiestaResult:
    label: str
    workdir: Path
    converged: bool = False
    energy_ev: Optional[float] = None
    cell: Optional[list] = None
    max_force: Optional[float] = None
    seconds: Optional[float] = None
    returncode: Optional[int] = None
    error: Optional[str] = None
    params: dict = field(default_factory=dict)


def find_siesta(explicit: Optional[Path] = None) -> Path:
    """Locate the SIESTA binary, or raise with instructions."""
    candidates = [explicit, *SIESTA_CANDIDATES, shutil.which("siesta")]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "SIESTA not found. Install with:\n"
        "  micromamba create -p <prefix>/siesta-env -c conda-forge siesta"
    )


# =============================================================================
#                    ********* INPUT GENERATION *********                    
#         FDF compilation logic mapping strict arguments to SIESTA syntax.     
# =============================================================================

def write_fdf(
    atoms,
    workdir: Path,
    label: str = "acv",
    mesh_cutoff_ry: float = 350.0,
    basis: str = "DZP",
    kgrid: tuple[int, int, int] = (10, 10, 1),
    xc: str = "GGA",
    xc_authors: str = "PBE",
    relax: bool = True,
    vacuum_axis: Optional[int] = None,
    max_scf: int = 200,
) -> Path:
    """Write a SIESTA .fdf input for a 2D system.

    The k-grid is forced to 1 along the vacuum axis. Sampling vacuum is meaningless and
    multiplies the cost for nothing -- and getting it wrong is one of the commonest
    errors in the audited 2D literature.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    species = sorted(set(atoms.get_chemical_symbols()))

    if vacuum_axis is None:
        lengths = atoms.cell.lengths()
        vacuum_axis = int(max(range(3), key=lambda i: lengths[i]))
    kgrid = tuple(1 if i == vacuum_axis else k for i, k in enumerate(kgrid))

    lines = [
        f"SystemName        {label}",
        f"SystemLabel       {label}",
        f"NumberOfAtoms     {len(atoms)}",
        f"NumberOfSpecies   {len(species)}",
        "",
        "%block ChemicalSpeciesLabel",
    ]
    from ase.data import atomic_numbers

    for i, sym in enumerate(species, start=1):
        lines.append(f"  {i}  {atomic_numbers[sym]}  {sym}")
    lines += [
        "%endblock ChemicalSpeciesLabel",
        "",
        f"PAO.BasisSize     {basis}",
        f"MeshCutoff        {mesh_cutoff_ry} Ry",
        f"XC.functional     {xc}",
        f"XC.authors        {xc_authors}",
        "",
        "%block kgrid_Monkhorst_Pack",
        f"  {kgrid[0]}  0  0  0.0",
        f"  0  {kgrid[1]}  0  0.0",
        f"  0  0  {kgrid[2]}  0.0",
        "%endblock kgrid_Monkhorst_Pack",
        "",
        "ElectronicTemperature  25 meV",
        "DM.MixingWeight        0.1",
        f"MaxSCFIterations       {max_scf}",
        "DM.Tolerance           1.d-4",
        "",
    ]
    # Warn on ACV_CG_STEPS=0, which aborts relaxation for testing purposes.
    cg_steps = int(os.getenv("ACV_CG_STEPS", "200"))
    if cg_steps == 0:
        log.warning(
            "ACV_CG_STEPS=0: single-point SCF, geometry NOT relaxed. Smoke-test only; "
            "any lattice comparison from this run is meaningless."
        )

    if relax:
        lines += [
            "MD.TypeOfRun        CG",
            f"MD.NumCGsteps       {cg_steps}",
            "MD.MaxForceTol      0.02 eV/Ang",
            "MD.VariableCell     .true.",
            "MD.MaxStressTol     0.1 GPa",
            "",
        ]
    else:
        # Fixed cell, atoms free. Required for strain-stress elastic constants: letting
        # the cell relax would undo the applied strain and return zero stiffness.
        lines += [
            "MD.TypeOfRun        CG",
            f"MD.NumCGsteps       {min(cg_steps, 100)}",
            "MD.MaxForceTol      0.02 eV/Ang",
            "MD.VariableCell     .false.",
            "",
        ]
    lines += ["WriteStress  .true.", ""]
    lines += ["LatticeConstant   1.0 Ang", "%block LatticeVectors"]
    for row in atoms.cell:
        lines.append("  " + "  ".join(f"{v:.10f}" for v in row))
    lines += ["%endblock LatticeVectors", "", "AtomicCoordinatesFormat  Ang",
              "%block AtomicCoordinatesAndAtomicSpecies"]
    for atom in atoms:
        idx = species.index(atom.symbol) + 1
        lines.append(
            f"  {atom.position[0]:.10f}  {atom.position[1]:.10f}  "
            f"{atom.position[2]:.10f}  {idx}"
        )
    lines += ["%endblock AtomicCoordinatesAndAtomicSpecies", ""]

    path = workdir / f"{label}.fdf"
    path.write_text("\n".join(lines), encoding="utf-8")

    # SIESTA looks for <Symbol>.psml in the working directory.
    for sym in species:
        src = PSEUDO_DIR / f"{sym}.psml"
        if not src.exists():
            raise FileNotFoundError(f"no pseudopotential for {sym} at {src}")
        shutil.copy(src, workdir / f"{sym}.psml")

    return path


# =============================================================================
#                     ********* OUTPUT PARSING *********                    
#       Deterministic scraping of scalar quantities from SIESTA stdout.        
# =============================================================================

def parse_output(text: str) -> dict:
    """Extract requisite quantities from SIESTA stdout."""
    out: dict = {"converged": False}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("siesta:         Total ="):
            try:
                out["energy_ev"] = float(stripped.split("=")[1])
            except (ValueError, IndexError):
                pass
        elif "Max " in stripped and "constrained" in stripped:
            parts = stripped.split()
            try:
                out["max_force"] = float(parts[1])
            except (ValueError, IndexError):
                pass
        elif "SCF Convergence by DM" in stripped or "SCF cycle converged" in stripped:
            out["converged"] = True
        elif "Job completed" in stripped:
            out["completed"] = True
    return out


# =============================================================================
#                       ********* EXECUTION *********                        
#         Process invocation, concurrency barriers, and filesystem state.      
# =============================================================================

def run(
    atoms,
    label: str = "acv",
    workdir: Optional[Path] = None,
    concurrency: int = 1,
    ranks: int = 1,
    dry_run: bool = True,
    keep_bulky: bool = False,
    timeout_s: int = 3600,
    **fdf_kwargs,
) -> SiestaResult:
    """Run SIESTA locally.

    dry_run defaults to True: the input is written and the guardrails are checked, but
    nothing executes. Actually spending compute must be an explicit decision.
    """
    workdir = Path(workdir or (RUN_ROOT / label))

    # Ensure output is fully decoupled from prior runs.
    lock = workdir / ".acv-running"
    if workdir.exists():
        if lock.exists():
            # Fail-fast on concurrent runs accessing the same path.
            raise RuntimeError(
                f"{workdir.name} is already in use by another run (remove {lock} if that "
                "run is dead). Concurrent runs on the same paper are not supported."
            )
        for stale in workdir.iterdir():
            if stale.is_file():
                stale.unlink()
            else:
                shutil.rmtree(stale, ignore_errors=True)
    result = SiestaResult(label=label, workdir=workdir, params=dict(fdf_kwargs))

    basis = fdf_kwargs.get("basis", "DZP")
    guard.check(n_atoms=len(atoms), concurrency=concurrency, basis=basis,
                workdir=workdir.parent)

    fdf = write_fdf(atoms, workdir, label=label, **fdf_kwargs)
    lock.touch()
    if dry_run:
        result.error = "dry run: input written, nothing executed"
        log.info("dry run -> %s", fdf)
        return result

    binary = find_siesta()
    started = time.time()
    try:
        # Pin OpenMP/BLAS threads to prevent oversubscription under concurrency.
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        if ranks > 1:
            cmd = [MPIRUN, "-np", str(ranks), "--oversubscribe", str(binary)]
        else:
            cmd = [str(binary)]
        with open(fdf, "rb") as stdin, open(workdir / f"{label}.out", "wb") as stdout:
            proc = subprocess.run(
                cmd, stdin=stdin, stdout=stdout,
                stderr=subprocess.PIPE, cwd=workdir, timeout=timeout_s, env=env,
            )
        result.returncode = proc.returncode
        text = (workdir / f"{label}.out").read_text(errors="replace")
        parsed = parse_output(text)
        result.converged = bool(parsed.get("converged"))
        result.energy_ev = parsed.get("energy_ev")
        result.max_force = parsed.get("max_force")
        if proc.returncode != 0:
            result.error = proc.stderr.decode(errors="replace")[-400:]
    except subprocess.TimeoutExpired:
        result.error = f"timed out after {timeout_s}s"
    finally:
        result.seconds = round(time.time() - started, 1)
        lock.unlink(missing_ok=True)
        if not keep_bulky and workdir.exists():
            # Safe cleanup in `finally` block to preserve original exception context.
            for path in workdir.iterdir():
                if path.suffix in BULKY_SUFFIXES:
                    path.unlink(missing_ok=True)

    return result


def run_many(
    jobs: list[tuple],
    concurrency: int = LOCAL_MPI_RANKS,
    dry_run: bool = True,
    **common,
) -> list[SiestaResult]:
    """Execute concurrent batch of SIESTA calculations using ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not jobs:
        return []

    biggest = max(len(j[0]) for j in jobs)
    basis = common.get("basis", "DZP")
    guard.check(n_atoms=biggest, concurrency=concurrency, basis=basis, workdir=RUN_ROOT)

    def _one(job):
        atoms, label = job[0], job[1]
        overrides = job[2] if len(job) > 2 else {}
        return run(atoms, label=label, dry_run=dry_run, concurrency=concurrency,
                   **{**common, **overrides})

    results: list[SiestaResult] = []
    # ThreadPoolExecutor is sufficient as subprocess.run releases the GIL.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_one, job): job[1] for job in jobs}
        for future in as_completed(futures):
            label = futures[future]
            try:
                res = future.result()
            except Exception as exc:                       # noqa: BLE001
                res = SiestaResult(label=label, workdir=RUN_ROOT / label,
                                   error=f"{type(exc).__name__}: {exc}")
            results.append(res)
            log.info("  %-32s %s  %ss", label[:32],
                     "converged" if res.converged else (res.error or "failed")[:40],
                     res.seconds)
    return results
