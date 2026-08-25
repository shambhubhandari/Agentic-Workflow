"""Tier 2: verification_flow module.

Provides strict, deterministic logic and strict typing for verification_flow operations.
"""
from __future__ import annotations

# =============================================================================
#                    ********* ORCHESTRATION FLOWS *********                   
#                   Strict definitions for verification_flow.                  
# =============================================================================

import logging
import time
import json
from pathlib import Path
from typing import Optional

from ..crews.convergence import convergence_crew as convergence
from ..crews.diagnostician import diagnostician_crew as diagnostician
from ..crews.translator import translator_crew as translator
from ..executors import local
from ..guardrails import epistemic
from ..guardrails.resources import ResourceRefused
from ..hooks import audit_log, provenance
from ..settings import LOCAL_MPI_RANKS, PROCESSED
from .state import ConvergenceStep, SiestaParameters, Stage, VerificationState


def _agent_model_name() -> str:
    """Model id for audit rows written OUTSIDE a crew (the failure paths).

    The crews get this from `llm.agent_model()`; the flow does not import llm, so it is
    resolved lazily here to keep the failure path free of import-time cost.
    """
    try:
        from ..llm import agent_model
        return agent_model()[1]
    except Exception:                                                # noqa: BLE001
        return "unknown"


log = logging.getLogger(__name__)

# Claims needed before the corpus median is treated as the prototype value.
from ..settings import effective as _eff          # noqa: E402

MIN_CLAIMS_FOR_CONSENSUS = int(_eff('tier2', 'targets.min_claims_for_consensus', None, 3))
# Fractional departure from that median beyond which a claim is taken to describe a
# different allotrope or a supercell rather than the prototype.
MAX_DEPARTURE_FROM_CONSENSUS = float(_eff('tier2', 'targets.max_departure_from_consensus', None, 0.25))

def _calibrated_offsets() -> dict[str, float]:
    """Measured SIESTA-vs-plane-wave offsets per property (plan.md 7.1).

    Read at call time rather than import time, so a calibration campaign that finishes
    while a session is open takes effect without a restart. Properties absent from the
    table are absent here too, and therefore return INCONCLUSIVE -- by construction, not
    by omission.
    """
    from ..memory import calibration_table

    return calibration_table.offsets()


def _cell_from(result_out: Path) -> Optional[list[float]]:
    """The RELAXED cell: the last `outcell` SIESTA printed, not the first.

    A variable-cell relaxation prints one `outcell` per CG step. Reading the first one
    returns the seeded cell.
    comparison was structurally guaranteed to report +0.00% and `consistent`, whatever
    the calculation actually did. Caught on penta-graphene, where 101 CG steps moved the
    cell from 3.680 x 3.773 to 3.768 x 4.870 and the verdict still read "difference 0.00%
    is within the stated tolerance".

    A verdict that cannot disagree is not a verdict.
    """
    if not result_out.exists():
        return None
    cell = None
    for line in result_out.read_text(errors="replace").splitlines():
        if "outcell: Cell vector modules" in line:
            try:
                cell = [float(x) for x in line.rsplit(":", 1)[1].split()]
            except (ValueError, IndexError):
                continue
    return cell


def _run_one(atoms, params: SiestaParameters, label: str, dry_run: bool,
             ranks: int = LOCAL_MPI_RANKS):
    # Pass explicit ranks setting downstream.
    return local.run(
        atoms,
        label=label,
        dry_run=dry_run,
        ranks=ranks,
        basis=params.basis,
        mesh_cutoff_ry=params.mesh_cutoff_ry,
        kgrid=list(params.kgrid),
        xc=params.xc,
        xc_authors=params.xc_authors,
    )


def targets(formulas: Optional[list[str]] = None) -> list[VerificationState]:
    """Build the verification targets from the extracted corpus.

    One target per (paper, composition) with a lattice claim -- not per claim, because a
    paper reporting a and b for one material is one structure to relax, not two.

    Selection is by REDUCED FORMULA and prototype, never by the formula string as written:
    "C" covers penta-graphene, penta-octa-graphene and several nanoribbons, and relaxing
    the wrong allotrope produces a divergence that says nothing about the paper. A
    composition with no recorded prototype is skipped rather than guessed at.
    """
    from ..knowledge import prototypes
    from ..pipeline import extract, normalize
    from ..types import ExtractionStatus

    # Resolution priority: CLI > configs/tier2.yaml > prototype composition.
    formulas = formulas or _eff("tier2", "targets.formulas", None)
    wanted = {f.strip() for f in formulas} if formulas else None
    out: dict[tuple[str, str], VerificationState] = {}
    candidates: dict[str, list] = {}

    # Gate recomputation conditionally on fully reported extraction parameters.
    full_only = bool(_eff("tier2", "targets.require_full_parameters", None, True))
    from ..pipeline.report import score_one

    for rec in extract.load():
        if rec.status != ExtractionStatus.OK or not rec.is_pentagonal_2d:
            continue
        if full_only and not score_one(rec).reproducible_in_principle:
            log.debug("skip %s: does not report every required parameter", rec.paper_key)
            continue
        for claim in rec.claims:
            if not normalize.property_kind(claim.property).value.startswith("lattice"):
                continue
            if not isinstance(claim.value, (int, float)):
                continue
            reduced = normalize.reduced_formula(
                normalize.normalize_formula(claim.material_formula) or ""
            )
            if not reduced or (wanted and reduced not in wanted):
                continue
            if prototypes.for_composition(reduced) is None:
                log.debug("skip %s: no recorded prototype", reduced)
                continue

            key = (rec.paper_key, reduced)
            candidates.setdefault(reduced, []).append((key, rec, claim))
            continue

    # --- reject values incompatible with the prototype ------------------------------
    # One composition covers several allotropes, and the prototype map cannot separate
    # them: "C" is penta-graphene (a = 3.64) but also penta-octa-graphene, nanoribbons and
    # supercells, which appear in the corpus at 4.94, 6.64 and 10.86 A. Relaxing
    # penta-graphene seeded at 10.86 A would diverge for a reason that has nothing to do
    # with the paper.
    #
    # The corpus is its own reference here: where several papers report a composition, the
    # median is the prototype value and a large departure is a different structure. With
    # too few claims to form a consensus the value is kept, because there is nothing to
    # judge it against and silently dropping it would be worse than flagging it.
    for reduced, group in candidates.items():
        values = sorted(float(c.value) for _, _, c in group)
        consensus = values[len(values) // 2] if len(values) >= MIN_CLAIMS_FOR_CONSENSUS else None
        for key, rec, claim in group:
            value = float(claim.value)
            if consensus and abs(value - consensus) / consensus > MAX_DEPARTURE_FROM_CONSENSUS:
                log.info(
                    "skip %s from %s: a = %.3f A departs %.0f%% from the corpus consensus "
                    "%.3f A, so it is very likely a different allotrope or a supercell",
                    reduced, rec.paper_key, value,
                    100 * abs(value - consensus) / consensus, consensus,
                )
                continue
            # Preserve origin axis and normalize across a/b variants.
            prop = str(getattr(claim, "property", "") or "")
            axis = "lattice_b" if prop.endswith("_b") else "lattice_a"
            existing = out.get(key)
            if existing and existing.claimed_value is not None:
                if float(claim.value) >= existing.claimed_value:
                    continue
            out[key] = VerificationState(
                paper_key=rec.paper_key,
                doi=rec.doi,
                formula=reduced,
                claimed_property=axis,
                claimed_value=float(claim.value),
                claimed_unit=claim.unit,
                paper_method=rec.method.model_dump() if rec.method else {},
            )
    return list(out.values())


def run_all(
    formulas: Optional[list[str]] = None,
    dry_run: bool = True,
    ranks: int = LOCAL_MPI_RANKS,
    critic_review: bool = True,
    out_dir: Optional[Path] = None,
) -> list[VerificationState]:
    """Verify every lattice claim in the corpus that has a recorded prototype."""
    from ..pipeline import generate

    states = targets(formulas)
    log.info("verification: %d target(s)", len(states))

    done: list[VerificationState] = []
    for state in states:
        try:
            atoms = generate.from_formula(state.formula, target_a=state.claimed_value)
        except Exception as exc:                                     # noqa: BLE001
            state.stage = Stage.ABANDONED
            state.abandoned_reason = f"could not build structure: {exc}"
            state.rationale = state.abandoned_reason
            log.warning("  %s: %s", state.formula, exc)
            done.append(state)
            continue
        done.append(
            verify(state, atoms, dry_run=dry_run, ranks=ranks,
                   critic_review=critic_review, out_dir=out_dir)
        )
    return done


def _calibration_shortfall() -> Optional[str]:
    """Why calibration cannot support a verdict, or None when it can.

    Read at call time so a calibration campaign that finishes mid-run takes effect.
    """
    path = PROCESSED / "calibration.json"
    if not path.exists():
        return "no calibration has been run"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "calibration.json is unreadable"
    if data.get("usable"):
        return None
    got = data.get("distinct_structures") or []
    need = int(_eff("tier2", "calibration.min_distinct_structures", None, 6))
    return (f"{len(got)} distinct structure(s) calibrated of {need} required "
            f"({', '.join(map(str, got))})")


def _apply_critic(state: VerificationState) -> None:
    """Review the finished verification before its verdict is allowed to stand.

    The Critic is advisory in one direction only. It can HOLD a verdict -- downgrade it
    to INCONCLUSIVE -- but it can never promote one, so a persuasive agent cannot talk
    the pipeline into a stronger claim than the numbers support. The arithmetic verdict
    is preserved in `provisional_verdict` so a reviewer can see exactly what was held and
    why, rather than finding an INCONCLUSIVE with no trace of the comparison behind it.

    A Critic that errors or is unavailable must not silently pass the verdict through:
    that would make the review look performed when it was not. The failure is recorded
    on the state instead.
    """
    from ..crews.critic import critic_crew
    from ..knowledge import prototypes

    # Prototype resolved systematically via structural registry.
    expected = prototypes.expected_symmetry(state.formula)

    try:
        review = critic_crew.review(state, prototype=expected.get("symmetry", "unknown"))
    except Exception as exc:                                        # noqa: BLE001
        state.critic_verdict = "unavailable"
        state.critic_reasoning = f"{type(exc).__name__}: {exc}"
        # Ensure robust error logging on failed agent invocations.
        audit_log.record("critic", "critic/failed", model=_agent_model_name(),
                         error=f"{type(exc).__name__}: {exc}",
                         context={"material": state.formula, "outcome": "COLLAPSED"})
        log.warning("critic unavailable, verdict left unreviewed: %s", exc)
        return

    state.critic_verdict = review.verdict
    state.critic_reasoning = review.reasoning
    state.critic_concerns = [c.model_dump() for c in review.concerns]
    state.critic_tools_used = review.tools_used

    blocking = [c for c in review.concerns if c.severity == "blocking"]
    if review.verdict != "hold" and not blocking:
        log.info("critic: pass (%d tool call(s))", len(review.tools_used))
        return

    state.provisional_verdict = state.verdict
    state.verdict = epistemic.Verdict.INCONCLUSIVE
    reasons = "; ".join(f"{c.category}: {c.detail}" for c in blocking) or review.reasoning
    state.rationale = (
        f"held at review ({reasons}). Provisional comparison was "
        f"{state.provisional_verdict.value if state.provisional_verdict else 'none'}: "
        f"{state.rationale}"
    )
    log.warning("critic: HOLD -- %s", reasons)


def verify(
    state: VerificationState,
    atoms,
    dry_run: bool = True,
    out_dir: Optional[Path] = None,
    ranks: int = LOCAL_MPI_RANKS,
    critic_review: bool = True,
) -> VerificationState:
    """Run one claim through the Tier 2 flow."""
    out_dir = Path(out_dir or (PROCESSED / "verification"))
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # --- translate ---------------------------------------------------------------
    try:
        state.parameters = translator.translate(
            state.paper_method, formula=state.formula, n_atoms=len(atoms)
        )
    except Exception as exc:                                          # noqa: BLE001
        state.stage = Stage.ABANDONED
        state.abandoned_reason = f"translation unavailable: {type(exc).__name__}: {exc}"
        state.rationale = state.abandoned_reason
        state.verdict = epistemic.Verdict.NOT_ATTEMPTED
        audit_log.record("translator", "translator/failed", model=_agent_model_name(),
                         error=f"{type(exc).__name__}: {exc}",
                         context={"material": state.formula, "outcome": "COLLAPSED"})
        log.warning("%s: %s", state.formula, state.abandoned_reason)
        return state
    state.stage = Stage.TRANSLATED
    log.info(
        "translated: basis=%s mesh=%s k=%s  (%d parameters unmapped)",
        state.parameters.basis, state.parameters.mesh_cutoff_ry,
        state.parameters.kgrid, len(state.parameters.unmapped),
    )

    # --- converge ----------------------------------------------------------------
    state.stage = Stage.CONVERGING
    params = state.parameters
    attempt = 0

    while True:
        iteration = len(state.steps) + 1
        # Incorporate composition in label to prevent filesystem collision.
        label = f"{state.paper_key.replace('/', '_')}-{state.formula}-i{iteration}"
        step = ConvergenceStep(
            iteration=iteration,
            parameters={
                "basis": params.basis,
                "mesh_cutoff_ry": params.mesh_cutoff_ry,
                "kgrid": list(params.kgrid),
            },
        )

        try:
            result = _run_one(atoms, params, label, dry_run, ranks=ranks)
        except ResourceRefused as exc:
            step.error = f"refused: {exc}"
            state.steps.append(step)
            state.stage = Stage.ABANDONED
            state.abandoned_reason = str(exc)
            break

        if dry_run:
            step.decision = "dry run: input written, nothing executed"
            state.steps.append(step)
            state.stage = Stage.ABANDONED
            state.abandoned_reason = "dry run"
            break

        state.n_calculations += 1
        step.seconds = result.seconds
        step.converged_scf = result.converged
        step.energy_ev = result.energy_ev
        step.cell = _cell_from(Path(result.workdir) / f"{label}.out")

        if not result.converged or result.energy_ev is None:
            # --- diagnose (failure path only) ------------------------------------
            state.stage = Stage.DIAGNOSING
            attempt += 1
            out_file = Path(result.workdir) / f"{label}.out"
            text = out_file.read_text(errors="replace") if out_file.exists() else ""
            try:
                dx = diagnostician.diagnose(text, error=result.error, attempt=attempt)
            except Exception as exc:                                  # noqa: BLE001
                # Isolate exceptions to protect the broader campaign loop.
                audit_log.record("diagnostician", "diagnostician/failed",
                                 model=_agent_model_name(),
                                 error=f"{type(exc).__name__}: {exc}",
                                 context={"material": state.formula,
                                          "outcome": "COLLAPSED"})
                log.warning("diagnostician unavailable (%s); abandoning this claim", exc)
                step.error = f"diagnostician failed: {type(exc).__name__}"
                state.steps.append(step)
                state.stage = Stage.ABANDONED
                state.abandoned_reason = f"diagnosis unavailable: {exc}"
                break
            step.error = f"{dx.failure_class.value}: {dx.evidence_line[:120]}"
            step.decision = f"diagnosis={dx.failure_class.value} retry={dx.retry_recommended}"
            state.failure_class = dx.failure_class.value
            state.steps.append(step)
            if not dx.retry_recommended:
                state.stage = Stage.ABANDONED
                state.abandoned_reason = f"{dx.failure_class.value}; no retry"
                break
            continue

        # deltas against the previous completed step
        previous = next(
            (s for s in reversed(state.steps) if s.energy_ev is not None), None
        )
        if previous and previous.energy_ev is not None and step.energy_ev is not None:
            step.delta_energy_ev = (step.energy_ev - previous.energy_ev) / max(len(atoms), 1)
            if previous.cell and step.cell:
                step.delta_cell_pct = 100 * max(
                    abs(n - o) / o for n, o in zip(step.cell[:2], previous.cell[:2]) if o
                )
        state.steps.append(step)

        
        try:
            next_params, decision, note = convergence.decide_next(state.steps, params)
        except Exception as exc:                                      # noqa: BLE001
            audit_log.record("convergence", "convergence/failed",
                             model=_agent_model_name(),
                             error=f"{type(exc).__name__}: {exc}",
                             context={"material": state.formula,
                                      "iteration": iteration,
                                      "outcome": "COLLAPSED"})
            state.stage = Stage.ABANDONED
            state.abandoned_reason = (
                f"convergence agent unavailable at iteration {iteration}: "
                f"{type(exc).__name__}: {exc}"
            )
            state.rationale = state.abandoned_reason
            log.warning("convergence unavailable (%s); abandoning this claim", exc)
            break
        step.decision = f"{decision.action}: {note}"
        log.info("  iteration %d -> %s", iteration, note)
        if next_params is None:
            converged, why = convergence.is_converged(state.steps)
            state.stage = Stage.CONVERGED if converged else Stage.ABANDONED
            if not converged:
                state.abandoned_reason = note
            break
        params = next_params

        provenance.record(result, atoms=atoms, structure_source=state.structure_path)

    # --- compare -----------------------------------------------------------------
    last = state.last_step()
    if state.stage is Stage.CONVERGED and last and last.cell:
        # Match explicit claim axis against shorter/longer in-plane vectors.
        in_plane = sorted(last.cell)[:2]
        state.our_a, state.our_b = in_plane[0], in_plane[1]
        state.our_value = (state.our_b if state.claimed_property == "lattice_b"
                           else state.our_a)
        verdict, rationale = epistemic.decide(
            state.our_value,
            state.claimed_value,
            calibrated_offset=_calibrated_offsets().get(state.claimed_property),
            converged=True,
        )
        state.verdict = verdict
        state.rationale = rationale
        state.stage = Stage.COMPARED

        
        insufficient = _calibration_shortfall()
        if insufficient and verdict is not epistemic.Verdict.NOT_ATTEMPTED:
            state.verdict = epistemic.Verdict.INCONCLUSIVE
            state.rationale = (
                f"calibration insufficient ({insufficient}); a threshold comparison is "
                f"not defined. Provisional comparison was {verdict.value}: {rationale}"
            )
    else:
        state.verdict = epistemic.Verdict.NOT_ATTEMPTED if not state.steps else \
            epistemic.Verdict.INCONCLUSIVE
        state.rationale = state.abandoned_reason or "no completed calculation"

    # --- pre-verdict review --------------------------------------------------------
    # Bypass API reviews on dry runs.
    if critic_review and not dry_run:
        _apply_critic(state)
    elif not dry_run:
        # Mark bypassed reviews explicitly in the state record.
        state.critic_verdict = "disabled_by_operator"
        state.critic_reasoning = "pre-verdict review skipped by --no-critic"
        log.warning("critic review DISABLED for %s; verdict recorded unreviewed",
                    state.formula)

    # Nothing describing a paper leaves this function without passing the guardrail.
    epistemic.check_text(state.rationale)

    # Stamp what produced this verdict. A result that cannot be traced to the settings
    # behind it is exactly what this project exists to find in other people's work.
    from ..settings import config_fingerprint

    state.config_fingerprint = config_fingerprint()
    state.tolerance_used = epistemic.TOLERANCE
    state.total_seconds = round(time.time() - started, 1)
    
    _stem = f"{state.paper_key.replace('/', '_')}-{state.formula}"
    path = out_dir / f"{_stem}.json"
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    log.info("verdict: %s -- %s", state.verdict.value, state.rationale)
    return state
