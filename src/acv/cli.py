"""Command-line entry point.

Subcommands map strictly to pipeline stages.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .settings import GENERATED, LOCAL_MPI_RANKS


# =============================================================================
#                  ********* LOGGING & CONFIGURATION *********                 
# =============================================================================

def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )
    for noisy in ("httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# =============================================================================
#                     ********* COMMAND HANDLERS *********                    
# =============================================================================

def cmd_corpus(args) -> int:
    from .pipeline import corpus

    try:
        papers = corpus.build(from_year=args.from_year, max_papers=args.max_papers)
    except corpus.BudgetExhausted as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    print(f"\ncorpus: {len(papers)} papers, {sum(1 for p in papers if p.retrievable)} retrievable")
    return 0


def cmd_fetch(args) -> int:
    from .pipeline import corpus, fetch

    results = fetch.fetch_all(corpus.load(), limit=args.limit, force=args.force)
    print(f"\nfetched {sum(1 for r in results if r.ok)}/{len(results)}")
    return 0


def cmd_extract(args) -> int:
    from .pipeline import corpus, extract

    stats = extract.extract_passes(corpus.load(), limit=args.limit, force=args.force)
    recs = extract.load()
    ok = sum(1 for r in recs if r.status.value == "ok")
    print(f"\nextracted {ok}/{len(recs)} usable  ({stats['n_passes_requested']} pass(es))")
    if stats.get("stability") is not None:
        print(f"  pass agreement : {stats['stability']:.1%} "
              f"({stats['fields_in_all_passes']}/{stats['fields_in_any_pass']} fields)")
    print(f"  reported fields: {stats['reported_fields']}"
          f"   ungrounded cut: {stats['ungrounded_fields_cut']}"
          f"   collisions: {stats['collisions_resolved']} resolved,"
          f" {stats['collisions_unresolved']} unresolved")
    return 0


def cmd_audit(args) -> int:
    from .pipeline import report

    stats = report.run()
    n = stats["n_usable"]
    if not n:
        print("no usable extractions; run `acv extract` first", file=sys.stderr)
        return 1

    print(f"\n=== Tier 0 reportability · n = {n} ===\n")
    print(f"reproducible in principle : {stats['reproducible_in_principle']}/{n} "
          f"({100 * stats['reproducible_in_principle'] / n:.0f}%)")
    print(f"mean fraction reported    : {100 * stats['mean_fraction_reported']:.0f}%\n")
    print("not reported:")
    for field, count in stats["most_often_missing"]:
        print(f"  {field:<28} {count:3d}/{n}  ({100 * count / n:3.0f}%)")
    v = stats.get("validation", {})
    print(f"\nphysics flags: {v.get('n_flags', 0)} across "
          f"{v.get('n_flagged_records', 0)} papers  {v.get('by_field', {})}")
    print(f"k-sampling style: {stats.get('k_sampling_style', {})}")
    print(f"\ncodes: {stats['codes']}")
    print(f"elastic units: {stats['elastic_units']}")
    print(f"GPa without stated thickness: {stats['gpa_without_stated_thickness']}")

    if args.plot:
        print(f"\nfigure -> {report.plot(stats)}")
    return 0


def cmd_crossref(args) -> int:
    from .pipeline import crossref

    stats = crossref.run()
    v = stats["verdicts"]
    print(f"\n=== Tier 0.5 cross-reference · {stats['n_claims_checked']} claims ===\n")
    print(f"  corroborated  {v.get('corroborated', 0):3d}   matches a known 2D structure")
    print(f"  unmatched     {v.get('unmatched', 0):3d}   no counterpart within tolerance")
    print(f"  def-mismatch  {v.get('definition_mismatch', 0):3d}   same name, different quantity")
    print(f"  no_reference  {v.get('no_reference', 0):3d}   composition absent from databases")
    print(f"  skipped       {v.get('skipped', 0):3d}   property not comparable via OPTIMADE")
    if stats["median_rel_error_pct"] is not None:
        print(f"\n  median error where a reference exists: {stats['median_rel_error_pct']}%")
    return 0


def cmd_screen(args) -> int:
    from .pipeline import screen

    stats = screen.run(limit_per_formula=args.limit_per_formula)
    print(f"\n=== Tier 1 MLIP screening ({stats['model']}) ===\n")
    print(f"  screened   {stats['n_screened']} structures in {stats['total_seconds']}s")
    print(f"  median lattice error  {stats['median_lattice_error_pct']}%")
    print(f"  worst  lattice error  {stats['worst_lattice_error_pct']}%")
    print("\n  TRIAGE ONLY -- an MLIP disagreeing with a paper most likely means the")
    print("  potential is out of its training domain, not that the paper is wrong.")
    return 0


def cmd_generate(args) -> int:
    """Build penta structures from the PdSe2 template."""
    from .pipeline import generate

    out_dir = Path(args.out_dir) if args.out_dir else GENERATED
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for spec in args.structures:
        parts = spec.split(":")
        elements, target_a = parts[0].split(","), (float(parts[1]) if len(parts) > 1 else None)
        if len(elements) == 1:                       # unitary: A6
            m, x1, x2 = elements[0], elements[0], elements[0]
        elif len(elements) == 2:                     # binary: M=A, X=B
            m, x1, x2 = elements[0], elements[1], elements[1]
        elif len(elements) == 3:                     # ternary: M=A, X=2B+2C
            m, x1, x2 = elements
        else:
            print(f"  skip {spec!r}: expected 1-3 elements")
            continue
        atoms = generate.build(m, x1, x2, target_a=target_a)
        name = f"penta-{atoms.get_chemical_formula()}.vasp"
        atoms.write(out_dir / name, format="vasp")
        lengths = atoms.cell.lengths()
        written.append(name)
        print(f"  {name:28} a={lengths[0]:.3f}  b/a={lengths[1]/lengths[0]:.3f}")
    print(f"\n  {len(written)} structure(s) -> {out_dir}")
    return 0


def cmd_calibrate(args) -> int:
    """Measure the SIESTA-vs-plane-wave offset on pentagonal structures."""
    from .pipeline import calibrate

    stats = calibrate.run(ranks=args.ranks, max_structures=args.max_structures)
    print("\n=== Tier 2 calibration ===\n")
    print(f"  points     {stats.get('n_points', 0)}  "
          f"(usable {stats.get('n_usable', 0)})")
    print(f"  structures {stats.get('distinct_structures')}")
    lattice = stats.get("lattice")
    if not lattice:
        print(f"\n  NO THRESHOLD: {stats.get('note', 'insufficient data')}")
        print("  Tier 2 verdicts will be INCONCLUSIVE until this is resolved.")
        return 0
    print(f"  median {100 * lattice['median_abs_offset']:.2f}%   "
          f"p90 {100 * lattice['p90_abs_offset']:.2f}%   "
          f"max {100 * lattice['max_abs_offset']:.2f}%")
    return 0


def cmd_verify(args) -> int:
    """Run the Tier 2 verification flow and write verdicts."""
    from collections import Counter

    from .flows import verification_flow

    states = verification_flow.run_all(
        formulas=args.formulas or None,
        dry_run=not args.execute,
        ranks=args.ranks,
        critic_review=not args.no_critic,
    )
    print("\n=== Tier 2 verdicts ===\n")
    for st in states:
        held = " [HELD]" if st.provisional_verdict else ""
        print(f"  {st.formula:10} {st.verdict.value:14}{held}  {st.rationale[:88]}")
    counts = Counter(st.verdict.value for st in states)
    print(f"\n  {dict(counts)}")
    if not args.execute:
        print("\n  DRY RUN: inputs written, nothing executed. Pass --execute to spend compute.")
    return 0


def cmd_evaluate(args) -> int:
    """Extraction accuracy against the source text, with no hand labels."""
    from .pipeline import evaluate

    stats = evaluate.run()
    g, v = stats["grounding"], stats["value_support"]
    print(f"\n=== Extraction evaluation · {stats['n_reported_fields']} reported fields "
          f"across {stats['n_with_text']} papers ===\n")
    print("GROUNDING  is the quoted sentence really in the paper?")
    print(f"  exact        {g['exact']:4d}")
    print(f"  fuzzy        {g['fuzzy']:4d}   (PDF line-wrapping)")
    print(f"  absent       {g['absent']:4d}   <- quoted text not found")
    print(f"  no evidence  {g['no_evidence']:4d}")
    if g["rate"] is not None:
        print(f"  RATE         {100 * g['rate']:.1f}%")
    print("\nVALUE SUPPORT  does the number follow from that sentence?")
    print(f"  supported    {v['supported']:4d}   via {v['via'] or '-'}")
    print(f"  unsupported  {v['unsupported']:4d}")
    if v["rate"] is not None:
        print(f"  RATE         {100 * v['rate']:.1f}%")
    print("\nby field (lowest grounding first):")
    for name, s in sorted(stats["by_field"].items(), key=lambda x: x[1]["rate"])[:8]:
        print(f"  {name:<32} {s['grounded']:3d}/{s['n']:<3d} {100 * s['rate']:5.1f}%")
    print("\n  Note: Measuring missed extractions (false negatives) requires hand-labels.")
    return 0


def cmd_status(args) -> int:
    from .settings import INTERIM, FULLTEXT, PROCESSED, RAW

    corpus_file = RAW / "corpus.jsonl"
    n_corpus = (
        sum(1 for line in corpus_file.read_text().splitlines() if line.strip())
        if corpus_file.exists() else 0
    )
    n_text = len(list(FULLTEXT.glob("*.txt")))
    extracted = INTERIM / "extracted.jsonl"
    n_extracted = (
        sum(1 for line in extracted.read_text().splitlines() if line.strip())
        if extracted.exists() else 0
    )
    summary = PROCESSED / "summary.json"

    print(f"  corpus     {n_corpus:5d} papers")
    print(f"  fulltext   {n_text:5d} cached")
    print(f"  extracted  {n_extracted:5d} records")
    if summary.exists():
        stats = json.loads(summary.read_text())
        print(f"  audited    {stats['n_usable']:5d} usable  "
              f"({stats['reproducible_in_principle']} reproducible in principle)")
    return 0


# =============================================================================
#                   ********* ARGPARSE DEFINITION *********                  
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acv", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("corpus", help="build the paper corpus from OpenAlex")
    p.add_argument("--from-year", type=int, default=None)
    p.add_argument(
        "--max-papers", type=int, default=None,
        help="draw a deterministic sample of this many papers instead of the whole "
             "corpus. Overrides configs/corpus.yaml search.max_papers. The sample is "
             "hash-ordered, so it preserves the year distribution; it is a pilot "
             "subset, not the corpus.",
    )
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("fetch", help="download full text")
    p.add_argument("--limit", type=int, default=None,
                   help="override configs/corpus.yaml fetch.limit for this run only")
    p.add_argument("--force", action="store_true", help="re-fetch cached papers")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("extract", help="extract method parameters")
    p.add_argument("--limit", type=int, default=None,
                   help="override configs/corpus.yaml extract.limit for this run only")
    p.add_argument("--force", action="store_true", help="re-extract existing records")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("audit", help="score reportability and summarise")
    p.add_argument("--plot", action="store_true", help="also write the figure")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("crossref", help="cross-reference claims against 2D databases")
    p.set_defaults(func=cmd_crossref)

    p = sub.add_parser("screen", help="Tier 1: universal MLIP screening (triage only)")
    p.add_argument("--limit-per-formula", type=int, default=None,
                   help="override configs/tier0.yaml screen.limit_per_formula")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("generate", help="build penta structures from the PdSe2 template")
    p.add_argument(
        "structures", nargs="+",
        help="ELEMENTS[:target_a], e.g. 'C' (unitary), 'Pd,Se' (binary), "
             "'C,B,P' (ternary), 'C:3.64' to seed from a measured lattice constant",
    )
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser(
        "calibrate",
        help="Tier 2 prerequisite: measure the SIESTA-vs-plane-wave offset on penta structures",
    )
    p.add_argument("--ranks", type=int, default=LOCAL_MPI_RANKS)
    p.add_argument("--max-structures", type=int, default=None,
                   help="override configs/tier2.yaml calibration.max_structures")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("verify", help="Tier 2: run the verification flow and write verdicts")
    p.add_argument("--formulas", nargs="*", default=None,
                   help="restrict to these reduced formulas (default: all with a prototype)")
    p.add_argument("--ranks", type=int, default=LOCAL_MPI_RANKS)
    p.add_argument("--execute", action="store_true",
                   help="actually run SIESTA; without this only inputs are written")
    p.add_argument("--no-critic", action="store_true", help="Bypass the pre-verdict Critic review.")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser(
        "evaluate",
        help="extraction accuracy against the source text (no hand labels needed)",
    )
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("status", help="what exists on disk")
    p.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
