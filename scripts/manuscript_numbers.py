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

    Recompute systematically from PRE-gate corpus.
    """
    src = ROOT / "data/interim/extraction/rtx3050_q4_0/union.jsonl"
    if not src.exists():
        return None
    import sys as _s
    _s.path.insert(0, str(ROOT / "src"))
    from acv.pipeline.evaluate import value_grounded_in
    kept = cleared = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        path = ROOT / "data" / "raw" / "fulltext" / f"{rec['paper_key']}.txt"
        if not path.exists():
            # Untestable: clearing on absent text would manufacture the absence.
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for claim in (rec.get("claims") or []):
            value = claim.get("value")
            if value is None:
                continue
            if value_grounded_in(float(value), text):
                kept += 1
            else:
                cleared += 1
    if not (kept + cleared):
        return None
    return cleared, 100 * cleared / (kept + cleared)


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


def _arch_eval() -> Optional[str]:
    """Accuracy of each hardware configuration against the expert hand labels.

    Export configuration directly to LaTeX macro definitions.py -- do not edit by hand.
% Regenerate:  make manuscript-numbers
% Source: data/processed/summary.json, evaluation.json, and the locked corpus.
% config fingerprint: {summary.get('provenance', {}).get('config_fingerprint', 'unknown')}

% ---- corpus and retrieval --------------------------------------------------
\\newcommand{{\\Nseed}}{{16}}
\\newcommand{{\\Ncorpus}}{{{c['corpus']}}}
\\newcommand{{\\Noa}}{{{c['oa']}}}
\\newcommand{{\\Nfulltext}}{{{c['fulltext']}}}
\\newcommand{{\\Naudited}}{{{n_aud}}}
\\newcommand{{\\Nexcludednotpenta}}{{{summary.get('n_excluded_not_pentagonal', 0)}}}
\\newcommand{{\\Nextractedok}}{{{summary.get('n_extracted_ok', n_aud)}}}
\\newcommand{{\\Narxiv}}{{{f['arxiv']}}}
\\newcommand{{\\Noapdf}}{{{f['oa_pdf']}}}
\\newcommand{{\\Noahtml}}{{{f['oa_html']}}}
\\newcommand{{\\Nforbidden}}{{{f['forbidden']}}}

% ---- Tier 0 reportability --------------------------------------------------
\\newcommand{{\\Nreproducible}}{{{n_rep}}}
\\newcommand{{\\PctReproducible}}{{{pct(rate)}}}
\\newcommand{{\\PctMeanReported}}{{{pct(summary.get('mean_fraction_reported'))}}}

% ---- extraction quality (label-free evaluator) -----------------------------
\\newcommand{{\\Nfields}}{{{ev['n_reported_fields']}}}
\\newcommand{{\\PctGrounded}}{{{pct(g.get('rate'))}}}
% NOTE: when extract.passes > 1 the shipped extracted.jsonl is a UNION of gated passes,
% so grounding measured on it is 100% BY CONSTRUCTION and must not be published as an
% independent figure. The defensible number is grounding on a SINGLE raw pass, written
% below from data/interim/extraction/rtx3050_q4_0/pass_a.jsonl when that file exists.
\\newcommand{{\\PctSupported}}{{{pct(v.get('rate'))}}}
\\newcommand{{\\Nvaluechecked}}{{{n_valuechecked}}}
\\newcommand{{\\Nungrounded}}{{{g.get('absent', 0)}}}
% Claim VALUES struck by the numeric grounding gate -- a different defect from
% \\Nungrounded, which counts unlocatable evidence sentences on method fields.
\\newcommand{{\\Nclaimscleared}}{{{n_cleared}}}
\\newcommand{{\\Pctclaimscleared}}{{{pct_cleared}}}
% Grounding on a single RAW pass -- the independent, publishable figure.
\\newcommand{{\\PctGroundedRaw}}{{{pct(raw)}}}

% ---- model / hardware ------------------------------------------------------
\\newcommand{{\\Model}}{{qwen3:4b}}
\\newcommand{{\\GPU}}{{NVIDIA RTX~3050 Laptop (4\\,GiB)}}

% ---- hand-labelled reference set (M8) --------------------------------------
\\newcommand{{\\NlabelPapers}}{{{hl['papers'] if hl else 0}}}
\\newcommand{{\\NlabelJudgements}}{{{hl['judgements'] if hl else 0}}}
\\newcommand{{\\NlabelStubs}}{{{hl['stubs'] if hl else 0}}}
\\newcommand{{\\PctRecall}}{{{pct(hl['R']) if hl else 'n/a'}}}
\\newcommand{{\\PctRecallLo}}{{{pct(hl['lo']) if hl else 'n/a'}}}
\\newcommand{{\\PctRecallHi}}{{{pct(hl['hi']) if hl else 'n/a'}}}
\\newcommand{{\\PctPrecision}}{{{pct(hl['P']) if hl else 'n/a'}}}
\\newcommand{{\\PctFone}}{{{pct(hl['F1']) if hl else 'n/a'}}}
\\newcommand{{\\Nmisses}}{{{hl['fn'] if hl else 0}}}
% Recall over the 8 fields that apply to every code family (basis size is n/a on
% plane-wave codes); the code-VALUE check; and PDF-to-text conversion loss (batch 1).
\\newcommand{{\\PctRecallEight}}{{{pct_r8}}}
\\newcommand{{\\NcodeCorrect}}{{{code_ok}}}
\\newcommand{{\\NcodeChecked}}{{{code_n}}}
\\newcommand{{\\Nconversionlost}}{{{conv_n}}}
\\newcommand{{\\PctConversionLoss}}{{{conv_pct}}}

% ---- per-configuration accuracy against the expert labels (Table~\\ref{{tab:eval}}) ----
{arch}


% ---- measured ONCE, describing defects that were found and fixed -----------
% These are historical: they characterise the pre-fix pipeline and are quoted in the
% Methods as evidence for why the gates exist. They do not change when a run is redone.
\\newcommand{{\\PctGroundedPre}}{{93.3}}     % grounding before the gate
\\newcommand{{\\Nselfneg}}{{30}}             % reported=true on self-negating evidence
\\newcommand{{\\PctSelfneg}}{{5.7}}
\\newcommand{{\\Ntruncated}}{{9}}            % papers silently truncated, of 69
\\newcommand{{\\PctLocaliserNarrow}}{{58.2}} % negative result: keyword localiser
\\newcommand{{\\PctLocaliserWide}}{{71.1}}
\\newcommand{{\\PctStabilityQfour}}{{78.9}}   % field-decision stability, q4_0 cache
% Development cost, counted from the retained run logs (logs/*.log, "[ n/N ]" lines).
% A LOWER bound: superseded logs were not all kept. This is the figure that matters for
% Baseline cost evaluations derived via isolated execution parameters.
\\newcommand{{\\NdevExtractions}}{{509}}
% Agent activity, counted from data/processed/verification/*.json and the campaign logs
% (logs/*.log, excluding the translator-stability probe). Static: they characterise one
% campaign and do not change when numbers.tex is regenerated.
\\newcommand{{\\NverifRecords}}{{12}}
\\newcommand{{\\NtranslatorCalls}}{{18}}
\\newcommand{{\\NconvCalls}}{{26}}
\\newcommand{{\\NcriticCalls}}{{14}}
\\newcommand{{\\NcriticCollapse}}{{4}}
\\newcommand{{\\NcriticToolCalls}}{{52}}
\\newcommand{{\\NcriticChanged}}{{4}}
\\newcommand{{\\NdevCampaigns}}{{ten}}
\\newcommand{{\\NdevTokens}}{{4.7}}        % million input tokens, at 9.2k median/paper
\\newcommand{{\\PctPapersFlipped}}{{20}}      % papers whose verdict flipped, q4_0

% ---- Tier 2 (filled when the recomputation campaign completes) -------------
\\newcommand{{\\NclaimsTierTwo}}{{{n_par}}}
\\newcommand{{\\NprototypesTierTwo}}{{{n_par_mat}}}
\\newcommand{{\\NpapersTierTwo}}{{{n_par_pap}}}
\\newcommand{{\\MedianDeviation}}{{{med_dev}}}
\\newcommand{{\\MAEDeviation}}{{{mae_dev}}}
\\newcommand{{\\MAREDeviation}}{{{mare_dev}}}
\\newcommand{{\\NwithinTwoPct}}{{{n_within2}}}
"""
    OUT.write_text(body, encoding="utf-8")
    print(f"  wrote {OUT.relative_to(ROOT)}")
    print(f"    corpus {c['corpus']} -> fulltext {c['fulltext']} -> audited {n_aud}")
    print(f"    reproducible {n_rep} ({pct(rate)}%)  "
          f"grounding {pct(g.get('rate'))}%  support {pct(v.get('rate'))}%")
    if hl:
        print(f"    hand labels: P {hl['P']:.1%}  R {hl['R']:.1%}  F1 {hl['F1']:.1%}  "
              f"({hl['papers']} papers, {hl['judgements']} judgements)")


if __name__ == "__main__":
    main()
