"""System: resources module.

Provides strict, deterministic logic and strict typing for resources operations.
"""
from __future__ import annotations

# =============================================================================
#                     ********* SAFETY GUARDRAILS *********                    
#                       Strict definitions for resources.                      
# =============================================================================

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..settings import LOCAL_MIN_FREE_DISK_GB, LOCAL_MIN_FREE_RAM_GB, LOCAL_MPI_RANKS

log = logging.getLogger(__name__)


class ResourceRefused(RuntimeError):
    """A calculation was refused because the machine cannot safely run it."""


@dataclass
class Resources:
    free_ram_gb: float
    free_disk_gb: float
    physical_cores: int
    ranks: int


def _physical_cores() -> int:
    """Physical cores, not threads.

    Hyperthreads hurt MPI DFT codes: two ranks contending for one core's FPU run
    slower than one rank owning it. os.cpu_count() reports threads, so it would
    double the true figure on this machine (8 physical, 16 threads).
    """
    try:
        with open("/proc/cpuinfo") as fh:
            text = fh.read()
        ids = {
            line.split(":")[1].strip()
            for line in text.splitlines()
            if line.startswith("core id")
        }
        if ids:
            return len(ids)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def snapshot(path: Optional[Path] = None) -> Resources:
    """Current free resources."""
    free_ram_kb = 0
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    free_ram_kb = int(line.split()[1])
                    break
    except OSError:
        pass

    # Resolve closest existing ancestor for mount-point checks.
    target = Path(path or Path.cwd()).resolve()
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    return Resources(
        free_ram_gb=round(free_ram_kb / 1024 / 1024, 2),
        free_disk_gb=round(usage.free / 1024**3, 2),
        physical_cores=_physical_cores(),
        ranks=LOCAL_MPI_RANKS,
    )


def estimate_ram_gb(n_atoms: int, concurrency: int = 1, basis: str = "DZP") -> float:
    """Rough peak-memory estimate for `concurrency` SIESTA jobs running side by side.

    Our SIESTA build is serial, so parallelism here means N independent single-core
    jobs, not N MPI ranks sharing one calculation. Each job therefore needs its OWN
    full working set -- the cost is linear in concurrency, not sublinear as it would be
    for MPI.

    Deliberately crude and pessimistic: the purpose is refusing what obviously cannot
    fit, not predicting usage precisely.
    """
    per_atom_gb = {"SZ": 0.01, "SZP": 0.015, "DZ": 0.02, "DZP": 0.03, "TZP": 0.05}
    base = per_atom_gb.get(basis.upper(), 0.03)
    per_job = 0.15 + base * n_atoms
    return round(per_job * max(concurrency, 1), 2)


def check(
    n_atoms: int,
    concurrency: Optional[int] = None,
    basis: str = "DZP",
    workdir: Optional[Path] = None,
) -> Resources:
    """Refuse the calculation unless it fits. Returns the snapshot when it does.

    `concurrency` is the number of independent SIESTA jobs to be run side by side,
    one core each.
    """
    concurrency = concurrency or LOCAL_MPI_RANKS
    res = snapshot(workdir)

    if concurrency > res.physical_cores:
        raise ResourceRefused(
            f"{concurrency} concurrent jobs requested but only {res.physical_cores} "
            "physical cores. Oversubscribing makes the machine unusable and does not "
            "go faster."
        )

    needed = estimate_ram_gb(n_atoms, concurrency, basis)
    if needed > res.free_ram_gb - LOCAL_MIN_FREE_RAM_GB:
        raise ResourceRefused(
            f"estimated {needed} GB for {concurrency} concurrent {n_atoms}-atom jobs, "
            f"but only "
            f"{res.free_ram_gb} GB free (reserving {LOCAL_MIN_FREE_RAM_GB} GB). "
            "Reduce ranks, use a smaller basis, or free memory."
        )

    if res.free_disk_gb < LOCAL_MIN_FREE_DISK_GB:
        raise ResourceRefused(
            f"only {res.free_disk_gb} GB disk free, below the {LOCAL_MIN_FREE_DISK_GB} GB "
            "floor. SIESTA writes density and Hamiltonian files per run; filling the disk "
            "corrupts every subsequent calculation."
        )

    log.info(
        "resources ok: %d atoms x%d concurrent, ~%.1f GB of %.1f GB free, %.0f GB disk",
        n_atoms, concurrency, needed, res.free_ram_gb, res.free_disk_gb,
    )
    return res
