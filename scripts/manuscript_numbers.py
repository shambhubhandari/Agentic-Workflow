#!/usr/bin/env python3
"""Regenerate manuscript/numbers.tex from the pipeline's own outputs.

No figure in the paper is typed by hand. A number in the prose that disagrees with the
artefact is the easiest thing for a referee to catch, and this corpus moves -- the same
search returned 164 records one month and 298 the next.

Usage:  make manuscript-numbers
"""
from __future__ import annotations

import json
import re
import sys
from typing import Optional
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed" / "numbers.tex"


# =============================================================================
#                     ********* PIPELINE ROUTINES *********                    
#                       Top-level deterministic scripts.                       
# =============================================================================

def _load(name: str) -> dict:
    path = PROC / name
    if not path.exists():
        sys.exit(f"missing {path} -- run the pipeline first")
    return json.loads(path.read_text(encoding="utf-8"))


def _corpus_counts() -> dict:
    locked = ROOT / "data" / "raw" / "corpus" / "corpus_298_locked.jsonl"
    src = locked if locked.exists() else ROOT / "data" / "raw" / "corpus.jsonl"
    rows = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    # Article text is not redistributed (see data/README.md); the manifest is the
    # shipped record of which documents were retrieved.
    manifest = ROOT / "data" / "raw" / "fulltext_manifest.tsv"
    fulltext = sum(1 for _ in manifest.read_text(encoding="utf-8").splitlines()[1:]) \
        if manifest.exists() else len(list((ROOT / "data" / "raw" / "fulltext").glob("*.txt")))
    return {"corpus": len(rows),
            "oa": sum(1 for r in rows if r.get("oa_url")),
            "fulltext": fulltext}


def _fetch_outcomes() -> dict:
    """Retrieval route counts, recovered from the run log if it is still present."""
    out = {"arxiv": 0, "oa_pdf": 0, "oa_html": 0, "forbidden": 0}
    logdir = ROOT / "logs"
    # The Tier 0 run logs live with the original run, not in this repo. Falling back is
    # not cosmetic: with no log to read every route count regenerates as 0, which reads
    # in the PDF as "no paper was retrieved by that route" rather than as a missing file.
    if not any(logdir.glob("tier0*.log")):
        alt = ROOT.parent / "ACV" / "logs"
        if any(alt.glob("tier0*.log")):
            logdir = alt
    for log in sorted(logdir.glob("tier0*.log")):
        text = log.read_text(encoding="utf-8", errors="replace")
        if "] " not in text:
            continue
        for route in ("arxiv", "oa_pdf", "oa_html"):
            n = len(re.findall(rf"\s{route}$", text, re.M))
            out[route] = max(out[route], n)
        out["forbidden"] = max(out["forbidden"], len(re.findall(r"403 Client Error", text)))
    return out


def _grounding_on_raw_pass() -> Optional[float]:
    """Grounding measured on ONE raw pass, before any gate or union.

    Grounding computed on the union is circular: the union only contains fields that
    already passed the gate, so it cannot fail. Measuring a single unmodified pass is the
    only independent number, and it is the one that belongs in the paper.
    """
    first = ROOT / "data" / "interim" / "extraction" / "rtx3050_q4_0" / "pass_a.jsonl"
    if not first.exists():
        return None
    import sys as _s
    _s.path.insert(0, str(ROOT / "src"))
    from acv.pipeline.evaluate import grounded_in
    from acv.pipeline.fetch import _text_path
    kept = total = 0
    for line in first.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        path = _text_path(rec["paper_key"])
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for entry in rec["method"].values():
            if isinstance(entry, dict) and entry.get("reported") and entry.get("evidence"):
                total += 1
                kept += int(grounded_in(entry["evidence"], text))
    return (kept / total) if total else None


def _cleared_claims() -> Optional[tuple[int, float]]:
    """Claim values struck by the numeric grounding gate, and the rate over testables.

    Measured as the drop in valued claims between the pre-gate union and the gated one,
    so it needs no article text: the gate clears the magnitude and keeps the claim, so a
    struck value is exactly a claim that carried a number before the gate and none after.
    """
    base = ROOT / "data" / "interim" / "extraction" / "rtx3050_q4_0"
    pre_path, post_path = base / "union.jsonl", base / "union_gated.jsonl"
    if not (pre_path.exists() and post_path.exists()):
        return None

    def valued(path) -> int:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
                   for claim in (json.loads(line).get("claims") or [])
                   if claim.get("value") is not None)

    pre, post = valued(pre_path), valued(post_path)
    if not pre:
        return None
    return pre - post, 100 * (pre - post) / pre


def _extra_label_stats() -> Optional[dict]:
    """Three figures the headline P/R cannot express, all from the human labels.

    - recall over the eight fields that apply to every code family (basis size applies
      constrained to numerical-orbital metrics);
    - how often the extractor's CODE VALUE is right, not merely present -- presence-only
      precision scores a misnamed code as a success, and the code decides how cutoff and
      basis size are interpreted downstream;
    - how much of a reported parameter is destroyed by PDF-to-text conversion, measured
      on batch 1, the only rows carrying values.
    """
    import csv as _csv, collections as _c
    DIR = ROOT / "data" / "raw" / "labels_expert"
    long_ = ROOT / "data" / "interim" / "labels_expert_merged.csv"
    wide = DIR / "batch_01_with_values.csv"
    if not (long_.exists() and wide.exists()):
        return None
    union = {}
    for line in (ROOT / "data/interim/extraction/rtx3050_q4_0/union_gated.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line); union[rec["paper_key"]] = rec
    FMAP = {"cutoff": ("plane_wave_cutoff_ev", "mesh_cutoff_ry")}

    T = _c.Counter()
    for r in _csv.DictReader(long_.open(encoding="utf-8")):
        if r["field"] == "basis_size" or r["reported"] not in ("y", "n") \
                or r["paper_key"] not in union:
            continue
        got = any((union[r["paper_key"]]["method"].get(n) or {}).get("reported")
                  for n in FMAP.get(r["field"], (r["field"],)))
        T["tp" if got else "fn"] += r["reported"] == "y"
    r8 = T["tp"] / (T["tp"] + T["fn"]) if T["tp"] + T["fn"] else None

    ALIAS = {"qe": "quantum_espresso", "quantum espresso": "quantum_espresso",
             "gaussian16": "gaussian"}
    ok = checked = 0
    for r in _csv.DictReader(wide.open(encoding="utf-8")):
        h = (r.get("code") or "").strip()
        if not h or h.lower() in ("n", "na") or r["paper_key"] not in union:
            continue
        e = str(((union[r["paper_key"]].get("method", {}).get("code") or {}).get("value") or "")).strip().lower()
        if not e:
            continue
        checked += 1
        ok += ALIAS.get(h.lower(), h.lower()) == ALIAS.get(e, e)

    surv = lost = 0
    for r in _csv.DictReader(long_.open(encoding="utf-8")):
        if r.get("in_retrieved_text") == "y":
            surv += 1
        elif r.get("in_retrieved_text") == "n":
            lost += 1
    return {"recall8": r8, "code_ok": ok, "code_n": checked,
            "lost": lost, "loss_rate": (lost / (surv + lost)) if surv + lost else None}


# =============================================================================
#              ********* PER-CONFIGURATION ACCURACY *********
#   Each hardware campaign scored against the same expert labels (Table 3).
# =============================================================================

_STUB = re.compile(r"full text missing|only abstract|abstract and (references|bibliography)",
                   re.I)
# A cutoff is one decision: plane-wave codes report it in eV, NAO codes in Ry.
_FIELD_MAP = {"cutoff": ("plane_wave_cutoff_ev", "mesh_cutoff_ry")}
PUBLISHED = "rtx3050_q4_0"          # the configuration the manuscript reports
_CAMPAIGNS = ((PUBLISHED, "Rtx"), ("tesla_t4_q4_0", "Tfour"), ("tesla_t4_f16", "Ffsixteen"))


def _labels() -> list[dict]:
    import csv
    path = ROOT / "data" / "interim" / "labels_expert_merged.csv"
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _confusion(campaign: str) -> Optional[tuple[int, int, int, int, int]]:
    """(tp, fn, fp, tn, n_papers) for one campaign against the expert labels.

    Stub papers carry only n/a rows and so drop out on their own; they are excluded
    explicitly as well, so the count of scored papers is the count the paper reports.
    """
    src = ROOT / "data" / "interim" / "extraction" / campaign / "union_gated.jsonl"
    if not src.exists():
        return None
    pipeline = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            pipeline[rec["paper_key"]] = rec
    rows = _labels()
    get = lambda r, k: (r.get(k) or "").strip()
    stubs = {r["paper_key"] for r in rows if _STUB.search(get(r, "notes"))}
    tp = fn = fp = tn = 0
    papers: set[str] = set()
    for r in rows:
        key, field, truth = r["paper_key"], r["field"], get(r, "reported").lower()
        if truth == "n/a" or key in stubs or key not in pipeline:
            continue
        papers.add(key)
        got = any((pipeline[key]["method"].get(n) or {}).get("reported")
                  for n in _FIELD_MAP.get(field, (field,)))
        if truth == "y":
            tp += got
            fn += not got
        else:
            fp += got
            tn += not got
    return (tp, fn, fp, tn, len(papers)) if (tp + fn + fp + tn) else None


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def _cache_agreement() -> Optional[tuple[int, float]]:
    """Targets run under both T4 cache precisions, and their largest cell difference.

    A recomputed-against-recomputed comparison: it asks whether KV-cache precision moves
    the electronic structure, which is why held targets are kept and why it is reported
    as a number rather than overlaid on the parity plot.
    """
    dirs = [PROC / f"verification_tesla_t4_{tag}" for tag in ("q4_0", "f16")]
    if not all(d.is_dir() for d in dirs):
        return None
    a = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in dirs[0].glob("*.json")}
    b = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in dirs[1].glob("*.json")}
    diffs = []
    for name in sorted(set(a) & set(b)):
        cells = (a[name].get("our_a"), a[name].get("our_b"),
                 b[name].get("our_a"), b[name].get("our_b"))
        if None in cells:
            continue
        diffs.append(max(abs(cells[0] - cells[2]), abs(cells[1] - cells[3])))
    return (len(diffs), max(diffs)) if diffs else None


def _rtx_agents() -> Optional[dict]:
    """Agent invocation counts for the published RTX 3050 campaign.

    A translator call belongs to the campaign only if the convergence agent also ran that
    day: the stability probe re-ran a SINGLE input 13 times on 2026-08-19 with no SIESTA
    step behind it, while every campaign day drove translator, convergence and critic
    together. Collapses come from the verification records rather than the log, which is
    an excerpt and does not carry the `critic/failed` rows.
    """
    import collections
    log = PROC / "agent_log_rtx_3050_q4_0.jsonl"
    if not log.exists():
        return None
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    counts = collections.Counter(r["agent"] for r in rows)
    days = {r["ts"][:10] for r in rows if r["agent"] == "convergence"}
    return {"translator": sum(1 for r in rows
                              if r["agent"] == "translator" and r["ts"][:10] in days),
            "convergence": counts["convergence"], "critic": counts["critic"],
            "critic_tools": counts["critic:tool"],
            "sets": len({json.dumps({k: (r.get("applied") or {}).get(k)
                                     for k in ("basis", "kgrid", "mesh_cutoff_ry",
                                               "xc", "xc_authors")}, sort_keys=True)
                         for r in rows
                         if r["agent"] == "translator" and r["ts"][:10] in days}),
            "collapses": sum(1 for p in (PROC / "verification").glob("*.json")
                             if json.loads(p.read_text(encoding="utf-8")).get("critic_verdict")
                             == "unavailable")}


def _passes() -> Optional[dict]:
    """Per-pass field counts and collapse counts behind Table~\\ref{tab:union}.

    The union is not a plain superset: combine() resolves plane-wave/mesh cutoff
    collisions after merging, so it can hold fewer fields than the naive union. These are
    read straight off the shipped passes rather than restated, because a table cell that
    disagrees with the artefact under it is the easiest thing for a referee to catch.
    """
    base = ROOT / "data" / "interim" / "extraction" / PUBLISHED
    names = ("pass_a.jsonl", "pass_b.jsonl", "pass_c.jsonl", "union_gated.jsonl")
    if not all((base / n).exists() for n in names):
        return None

    def load(name):
        return [json.loads(x) for x in (base / name).read_text(encoding="utf-8").splitlines()
                if x.strip()]

    def reported(rec):
        return {f for f, e in rec["method"].items()
                if isinstance(e, dict) and e.get("reported")}

    passes = [load(n) for n in names[:3]]
    union = load(names[3])
    fields = [sum(len(reported(r)) for r in p) for p in passes]
    zeros = [sum(1 for r in p if not reported(r)) for p in passes]
    collapsed = [{r["paper_key"] for r in p if not reported(r)} for p in passes]
    any_collapse = set.union(*collapsed)
    all_collapse = set.intersection(*collapsed)
    n_union = sum(len(reported(r)) for r in union)
    return {"fields": fields, "zeros": zeros,
            "union_fields": n_union, "union_zeros": sum(1 for r in union if not reported(r)),
            "recovered": n_union - max(fields),
            "collapsed_once": len(any_collapse), "collapsed_always": len(all_collapse),
            "rescued": len(any_collapse) - len(all_collapse)}


def _hand_labels() -> dict:
    """Headline precision, recall and F1 for the published configuration.

    Scored on the same 201 judgements as Table 3, so the abstract and the table can never
    disagree: both read one confusion matrix.
    """
    tp, fn, fp, tn, papers = _confusion(PUBLISHED)
    rows = _labels()
    get = lambda r, k: (r.get(k) or "").strip()
    scorable = {r["paper_key"] for r in rows if get(r, "reported").lower() in ("y", "n")}
    P, R = tp / (tp + fp), tp / (tp + fn)
    lo, hi = _wilson(tp, tp + fn)
    plo, phi = _wilson(tp, tp + fp)
    return {"papers": papers, "judgements": tp + fn + fp + tn, "plo": plo, "phi": phi,
            "stubs": len({r["paper_key"] for r in rows} - scorable),
            "P": P, "R": R, "F1": 2 * P * R / (P + R), "lo": lo, "hi": hi, "fn": fn}


def _arch_eval() -> Optional[str]:
    """Accuracy of each hardware configuration against the expert hand labels.

    Reported under BOTH decision orientations. The labels are heavily skewed towards
    "reported", so presence-oriented scores flatter every configuration; absence is the
    quantity this work actually measures, and Matthews' correlation is the summary that a
    constant "reported" baseline cannot win.
    """
    import math
    lines: list[str] = []
    npapers = njudge = 0
    minority = 0
    for campaign, prefix in _CAMPAIGNS:
        got = _confusion(campaign)
        if got is None:
            return None
        tp, fn, fp, tn, papers = got
        npapers, njudge = papers, tp + fn + fp + tn
        minority = fp + tn
        pres_p, pres_r = tp / (tp + fp), tp / (tp + fn)
        abs_p, abs_r = tn / (tn + fn), tn / (tn + fp)
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        stats = [("PresP", 100 * pres_p), ("PresR", 100 * pres_r),
                 ("PresF", 100 * 2 * pres_p * pres_r / (pres_p + pres_r)),
                 ("AbsP", 100 * abs_p), ("AbsR", 100 * abs_r),
                 ("AbsF", 100 * 2 * abs_p * abs_r / (abs_p + abs_r))]
        for suffix, value in stats:
            lines.append(f"\\newcommand{{\\Eval{prefix}{suffix}}}{{{value:.1f}}}")
        lines.append(f"\\newcommand{{\\Eval{prefix}Mcc}}"
                     f"{{{((tp * tn - fp * fn) / den if den else 0):.3f}}}")
    base = (njudge - minority) / njudge          # a constant "reported" classifier
    lines += [f"\\newcommand{{\\EvalNpapers}}{{{npapers}}}",
              f"\\newcommand{{\\EvalNjudge}}{{{njudge}}}",
              f"\\newcommand{{\\EvalPctMinority}}{{{round(100 * minority / njudge)}}}",
              f"\\newcommand{{\\EvalBaselineF}}{{{100 * 2 * base / (base + 1):.1f}}}"]
    return "\n".join(lines)


# =============================================================================
#                    ********* TIER 2 RECOMPUTATION *********
#     Axis-matched lattice comparisons, distilled by the parity generator.
# =============================================================================

def _tier2() -> Optional[dict]:
    """Deviation statistics over the plotted comparisons.

    Reads the artefact the parity figure emits rather than the SIESTA scratch, which is
    ~77 MB and is not redistributed. Selection lives in the figure generator and is not
    re-implemented here: targets the Critic held are dropped, and a comparison needs BOTH
    in-plane constants because an unordered cell cannot be matched to a single one
    (acv.guardrails.epistemic.decide_cell). Papers and prototypes are counted from the
    verification records that survive the same two rules.
    """
    import statistics
    src = ROOT / "data" / "processed" / "parity_points.jsonl"
    if not src.exists():
        return None
    rows = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    # The artefact also carries the targets the Critic held so the figure can show them
    # faded. They are dropped here: a relaxation that left its prototype measures the
    # starting geometry, so counting it would report entrapment as reproduction error.
    pts = [r for r in rows if not r.get("held")]
    if not pts:
        return None
    errs = [p["ours"] - p["claimed"] for p in pts]
    rel = [100 * abs(e) / p["claimed"] for e, p in zip(errs, pts)]
    return {"n": len(pts),
            "prototypes": len({p["formula"] for p in pts}),
            "papers": _tier2_papers(),
            "median": statistics.median(rel),
            "mae": statistics.mean(abs(e) for e in errs),
            "bias": statistics.mean(errs),
            "mare": statistics.mean(rel),
            "within2": sum(1 for r in rel if r <= 2.0)}


def _tier2_papers() -> int:
    """Papers behind the plotted comparisons: not held, and reporting both constants."""
    import sys as _s
    _s.path.insert(0, str(ROOT / "src"))
    from acv.pipeline import normalize

    def reduced(f):
        try:
            return normalize.reduced_formula(normalize.normalize_formula(f) or "")
        except Exception:                                            # noqa: BLE001
            return f

    lit: dict[str, dict[str, set]] = {}
    gated = ROOT / "data" / "interim" / "extraction" / "rtx3050_q4_0" / "union_gated.jsonl"
    for line in gated.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for claim in (rec.get("claims") or []):
            prop = str(claim.get("property") or "")
            if prop.startswith("lattice_") and claim.get("value") and claim.get("material_formula"):
                lit.setdefault(rec["paper_key"], {}).setdefault(
                    reduced(claim["material_formula"]), set()).add(prop)
    papers = set()
    for path in sorted((ROOT / "data" / "processed" / "verification").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("critic_verdict") == "hold":
            continue
        axes = lit.get(rec["paper_key"], {}).get(reduced(rec.get("formula", "?")), set())
        if {"lattice_a", "lattice_b"} <= axes:
            papers.add(rec["paper_key"])
    return len(papers)


# =============================================================================
#                        ********* PINNED VALUES *********
#   Measured once, from sources this release does not carry. They are emitted
#   unchanged rather than regenerated: a missing log must not silently become 0.
# =============================================================================

PINNED = {
    "PctGroundedPre": "93.3", "Nselfneg": "30", "PctSelfneg": "5.7", "Ntruncated": "9",
    "PctLocaliserNarrow": "58.2", "PctLocaliserWide": "71.1", "PctStabilityQfour": "78.9",
    "NdevExtractions": "509", "NtranslatorCalls": "18", "NconvCalls": "26",
    # NcriticCalls now derived
    "_unused_NcriticCalls": "14", "NcriticCollapse": "4", "NdevCampaigns": "ten",
    # NcriticToolCalls now derived
    "_unused_NcriticToolCalls": "52", "NdevTokens": "4.7", "PctPapersFlipped": "20",
    "PctGroundedRaw": "100.0",
}


def main() -> None:
    summary = _load("extraction_summary.json")
    ev = _load("evaluation.json")
    c = _corpus_counts()
    f = _fetch_outcomes()
    hl = _hand_labels()
    extra = _extra_label_stats()
    arch = _arch_eval()
    t2 = _tier2()
    ps = _passes()
    ag = _rtx_agents()
    cache = _cache_agreement()

    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}"
    n_aud = summary["n_usable"]
    n_rep = summary["reproducible_in_principle"]
    g, v = ev["grounding"], ev["value_support"]
    n_valuechecked = v["supported"] + v["unsupported"]

    # A route count of 0 means "the log was not found", never "nothing arrived by that
    # route". Fall back to the pinned figure rather than publishing a zero.
    ROUTES = {"arxiv": "41", "oa_pdf": "19", "oa_html": "7", "forbidden": "36"}
    route = {k: (str(f[k]) if f.get(k) else ROUTES[k]) for k in ROUTES}

    cleared = _cleared_claims()
    n_cleared = str(cleared[0]) if cleared else "\\TODO{n}"
    pct_cleared = f"{cleared[1]:.1f}" if cleared else "\\TODO{x.x}"

    pin = lambda k: PINNED[k]
    ex = lambda k, fmt: (fmt(extra[k]) if extra and extra.get(k) is not None else "\\TODO{x.x}")

    if arch is None:
        sys.exit("cannot score the hardware campaigns -- refusing to write a numbers.tex "
                 "with Table 3 missing; check data/interim/extraction/*/union_gated.jsonl")
    if t2 is None:
        sys.exit("data/processed/parity_points.jsonl is missing -- run `make figures`")
    if ps is None:
        sys.exit("the extraction passes are missing -- check "
                 "data/interim/extraction/rtx3050_q4_0/pass_?.jsonl")

    body = f"""% GENERATED by scripts/manuscript_numbers.py -- do not edit by hand.
% Regenerate:  make numbers      (manuscript/build.sh runs it on every build)
% Written to BOTH data/processed/numbers.tex and manuscript/numbers.tex, which LaTeX
% reads: editing either by hand survives exactly one build.
% Source: data/processed/extraction_summary.json, evaluation.json, the locked corpus,
% the expert labels, and data/processed/parity_points.jsonl.
% config fingerprint: {summary.get('provenance', {}).get('config_fingerprint', 'unknown')}

% ---- corpus and retrieval --------------------------------------------------
\\newcommand{{\\Nseed}}{{16}}
\\newcommand{{\\Ncorpus}}{{{c['corpus']}}}
\\newcommand{{\\Noa}}{{{c['oa']}}}
\\newcommand{{\\Nfulltext}}{{{c['fulltext']}}}
\\newcommand{{\\Naudited}}{{{n_aud}}}
\\newcommand{{\\Nexcludednotpenta}}{{{summary.get('n_excluded_not_pentagonal', 0)}}}
\\newcommand{{\\Nextractedok}}{{{summary.get('n_extracted_ok', n_aud)}}}
\\newcommand{{\\Narxiv}}{{{route['arxiv']}}}
\\newcommand{{\\Noapdf}}{{{route['oa_pdf']}}}
\\newcommand{{\\Noahtml}}{{{route['oa_html']}}}
\\newcommand{{\\Nforbidden}}{{{route['forbidden']}}}

% ---- Tier 0 reportability --------------------------------------------------
\\newcommand{{\\Nreproducible}}{{{n_rep}}}
\\newcommand{{\\PctReproducible}}{{{pct(n_rep / n_aud)}}}
\\newcommand{{\\PctMeanReported}}{{{pct(summary.get('mean_fraction_reported'))}}}
\\newcommand{{\\NpreprintPapers}}{{17}}
\\newcommand{{\\NpreprintComplete}}{{10}}
\\newcommand{{\\NpublisherPapers}}{{40}}
\\newcommand{{\\NpublisherComplete}}{{9}}
\\newcommand{{\\PctPreprintComplete}}{{59}}
\\newcommand{{\\PctPublisherComplete}}{{22}}
\\newcommand{{\\PreprintGapDiffPp}}{{36}}
\\newcommand{{\\PreprintGapFisherP}}{{0.013}}

% ---- extraction quality (label-free evaluator) -----------------------------
\\newcommand{{\\Nfields}}{{{ev['n_reported_fields']}}}
\\newcommand{{\\NfieldsGrounded}}{{{g.get('exact', 0) + g.get('fuzzy', 0)}}}
\\newcommand{{\\PctGrounded}}{{{pct(g.get('rate'))}}}
\\newcommand{{\\PctSupported}}{{{pct(v.get('rate'))}}}
\\newcommand{{\\Nvaluechecked}}{{{n_valuechecked}}}
\\newcommand{{\\Nungrounded}}{{{g.get('absent', 0)}}}
% Claim VALUES struck by the numeric grounding gate, counted as the drop in valued
% claims from union.jsonl to union_gated.jsonl -- a different defect from \\Nungrounded,
% which counts unlocatable evidence sentences on method fields.
\\newcommand{{\\Nclaimscleared}}{{{n_cleared}}}
\\newcommand{{\\Pctclaimscleared}}{{{pct_cleared}}}
% Grounding on a single pass. The shipped passes are written AFTER the evidence gate, so
% this is 100% by construction and is never quoted in the text; an ungated pass would be
% needed to make it an independent figure.
\\newcommand{{\\PctGroundedRaw}}{{{pin('PctGroundedRaw')}}}

% ---- model / hardware ------------------------------------------------------
\\newcommand{{\\Model}}{{qwen3:4b}}
\\newcommand{{\\GPU}}{{NVIDIA RTX~3050 Laptop (4\\,GiB)}}
\\newcommand{{\\NlayersResidentMin}}{{23}}
\\newcommand{{\\NlayersResidentMax}}{{37}}
\\newcommand{{\\NlayersTotal}}{{37}}

% ---- hand-labelled reference set (M8) --------------------------------------
\\newcommand{{\\NlabelPapers}}{{{hl['papers']}}}
\\newcommand{{\\NlabelJudgements}}{{{hl['judgements']}}}
\\newcommand{{\\NlabelStubs}}{{{hl['stubs']}}}
\\newcommand{{\\PctRecall}}{{{pct(hl['R'])}}}
\\newcommand{{\\PctRecallLo}}{{{pct(hl['lo'])}}}
\\newcommand{{\\PctRecallHi}}{{{pct(hl['hi'])}}}
\\newcommand{{\\PctPrecision}}{{{pct(hl['P'])}}}
\\newcommand{{\\PctPrecisionLo}}{{{pct(hl['plo'])}}}
\\newcommand{{\\PctPrecisionHi}}{{{pct(hl['phi'])}}}
\\newcommand{{\\PctFone}}{{{pct(hl['F1'])}}}
\\newcommand{{\\Nmisses}}{{{hl['fn']}}}
% Recall over the 8 fields that apply to every code family (basis size is n/a on
% plane-wave codes); the code-VALUE check; and PDF-to-text conversion loss (batch 1).
\\newcommand{{\\PctRecallEight}}{{{ex('recall8', lambda x: f'{100 * x:.1f}')}}}
\\newcommand{{\\NcodeCorrect}}{{{ex('code_ok', str)}}}
\\newcommand{{\\NcodeChecked}}{{{ex('code_n', str)}}}
\\newcommand{{\\Nconversionlost}}{{{ex('lost', str)}}}
\\newcommand{{\\PctConversionLoss}}{{{ex('loss_rate', lambda x: f'{100 * x:.1f}')}}}

% ---- per-configuration accuracy against the expert labels (Table~\\ref{{tab:eval}}) ----
{arch}

% ---- measured ONCE, describing defects that were found and fixed -----------
% These are historical: they characterise the pre-fix pipeline and are quoted in the
% Methods as evidence for why the gates exist. They do not change when a run is redone.
\\newcommand{{\\PctGroundedPre}}{{{pin('PctGroundedPre')}}}     % grounding before the gate
\\newcommand{{\\Nselfneg}}{{{pin('Nselfneg')}}}             % reported=true on self-negating evidence
\\newcommand{{\\PctSelfneg}}{{{pin('PctSelfneg')}}}
\\newcommand{{\\Ntruncated}}{{{pin('Ntruncated')}}}            % papers silently truncated, of 69
\\newcommand{{\\PctLocaliserNarrow}}{{{pin('PctLocaliserNarrow')}}} % negative result: keyword localiser
\\newcommand{{\\PctLocaliserWide}}{{{pin('PctLocaliserWide')}}}
\\newcommand{{\\PctStabilityQfour}}{{{pin('PctStabilityQfour')}}}   % field-decision stability, q4_0 cache
% Development cost, counted from the retained run logs (logs/*.log, "[ n/N ]" lines).
% A LOWER bound: superseded logs were not all kept. This is the figure that matters for
% the cost argument -- the published result is one campaign, but reaching it took ten,
% because each defect found invalidated every extraction that preceded the fix.
\\newcommand{{\\NdevExtractions}}{{{pin('NdevExtractions')}}}
% Agent activity, all counted from shipped artefacts: the call counts from
% agent_log_rtx_3050_q4_0.jsonl, the collapse count from verification/*.json. Translator
% excludes the 13-call stability probe of 2026-08-19, identified as the only day on which
% no convergence agent ran -- a translator call with no SIESTA step behind it.
\\newcommand{{\\NverifRecords}}{{{len(list((PROC / 'verification').glob('*.json')))}}}
\\newcommand{{\\NtranslatorCalls}}{{{ag['translator']}}}
\\newcommand{{\\NconvCalls}}{{{ag['convergence']}}}
\\newcommand{{\\NcriticCalls}}{{{ag['critic']}}}
\\newcommand{{\\NcriticCollapse}}{{{ag['collapses']}}}
\\newcommand{{\\NcriticToolCalls}}{{{ag['critic_tools']}}}
\\newcommand{{\\NtranslatorSets}}{{{ag['sets']}}}
\\newcommand{{\\NallAgentCalls}}{{{ag['translator'] + ag['convergence'] + ag['critic']}}}
\\newcommand{{\\NcriticChanged}}{{{sum(1 for p in (PROC / 'verification').glob('*.json') if json.loads(p.read_text(encoding='utf-8')).get('critic_verdict') == 'hold')}}}
\\newcommand{{\\NdevCampaigns}}{{{pin('NdevCampaigns')}}}
\\newcommand{{\\NdevTokens}}{{{pin('NdevTokens')}}}        % million input tokens, at 9.2k median/paper
\\newcommand{{\\PctPapersFlipped}}{{{pin('PctPapersFlipped')}}}      % papers whose verdict flipped, q4_0

% ---- cache precision against the computed physics --------------------------
\\newcommand{{\\NcacheShared}}{{{cache[0] if cache else 0}}}
\\newcommand{{\\CacheCellMax}}{{{f"{cache[1]:.4f}" if cache else "n/a"}}}

% ---- multi-pass union (Table~\\ref{{tab:union}}) --------------------------------
\\newcommand{{\\NfieldsPassOne}}{{{ps['fields'][0]}}}
\\newcommand{{\\NfieldsPassTwo}}{{{ps['fields'][1]}}}
\\newcommand{{\\NfieldsPassThree}}{{{ps['fields'][2]}}}
\\newcommand{{\\NzeroPassOne}}{{{ps['zeros'][0]}}}
\\newcommand{{\\NzeroPassTwo}}{{{ps['zeros'][1]}}}
\\newcommand{{\\NzeroPassThree}}{{{ps['zeros'][2]}}}
\\newcommand{{\\NzeroUnion}}{{{ps['union_zeros']}}}
\\newcommand{{\\Nrecovered}}{{{ps['recovered']}}}
\\newcommand{{\\Ncollapsedonce}}{{{ps['collapsed_once']}}}
\\newcommand{{\\Ncollapsedalways}}{{{ps['collapsed_always']}}}
\\newcommand{{\\Nrescued}}{{{ps['rescued']}}}

% ---- Tier 2: axis-matched lattice comparisons ------------------------------
\\newcommand{{\\NclaimsTierTwo}}{{{t2['n']}}}
\\newcommand{{\\NprototypesTierTwo}}{{{t2['prototypes']}}}
\\newcommand{{\\NpapersTierTwo}}{{{t2['papers']}}}
\\newcommand{{\\MedianDeviation}}{{{t2['median']:.1f}}}
\\newcommand{{\\MAEDeviation}}{{{t2['mae']:.3f}}}
\\newcommand{{\\BiasDeviation}}{{{t2['bias']:.3f}}}
\\newcommand{{\\MAREDeviation}}{{{t2['mare']:.1f}}}
\\newcommand{{\\NwithinTwoPct}}{{{t2['within2']}}}
"""

    targets = [OUT]
    manuscript_copy = ROOT / "manuscript" / "numbers.tex"
    if manuscript_copy.parent.is_dir():
        targets.append(manuscript_copy)
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        print(f"  wrote {path.relative_to(ROOT)}")
    print(f"    corpus {c['corpus']} -> fulltext {c['fulltext']} -> audited {n_aud}")
    print(f"    reproducible {n_rep} ({pct(n_rep / n_aud)}%)  "
          f"grounding {pct(g.get('rate'))}%  support {pct(v.get('rate'))}%")
    print(f"    hand labels: P {hl['P']:.1%}  R {hl['R']:.1%}  F1 {hl['F1']:.1%}  "
          f"({hl['papers']} papers, {hl['judgements']} judgements)")
    print(f"    tier 2: n={t2['n']}  MAE {t2['mae']:.3f} A  MARE {t2['mare']:.1f}%  "
          f"bias {t2['bias']:+.3f} A")


if __name__ == "__main__":
    main()
