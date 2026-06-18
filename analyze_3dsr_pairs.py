"""
Pair analysis for 3DSR: compare model accuracy on regular vs flipped questions.

For each question the dataset has a mirrored-image 'flip' variant whose correct
answer is the opposite.  A model that genuinely understands spatial relations
should get BOTH the regular and flipped version right; a model that is biased
or guessing will show an asymmetry.

Statistics
----------
- Accuracy + Wilson 95% CI on regular-only and flip-only subsets
- 2x2 contingency table: {both correct, reg-only correct, flip-only correct, both wrong}
- McNemar's test (with continuity correction) on the discordant pairs (b, c)
  H0: P(correct | regular) == P(correct | flipped)

Run from repo root:  python3 analyze_3dsr_pairs.py
"""

import json
import re
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import chi2 as chi2_dist

BASE = "data/evals/3DSR"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_freeform(text):
    """Base model: left XOR right present anywhere in text."""
    t = text.lower()
    has_left = "left" in t
    has_right = "right" in t
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    return None


def parse_direct(text):
    """Fine-tuned direct: output is exactly 'left' or 'right'."""
    t = text.strip().lower()
    return t if t in ("left", "right") else None


def parse_cot(text):
    """CoT: extract conclusion from 'Therefore … is to the (left|right)'."""
    m = re.search(r"Therefore.*?is to the (left|right)", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).lower() if m else None


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def wilson_ci(n_correct, n_total):
    if n_total == 0:
        return float("nan"), float("nan")
    lo, hi = proportion_confint(n_correct, n_total, method="wilson")
    return lo, hi


def mcnemar(b, c):
    """
    McNemar's test with continuity correction (Edwards 1948).
    b = reg_correct & flip_wrong
    c = reg_wrong  & flip_correct
    Returns (chi2_stat, p_value).
    """
    n = b + c
    if n == 0:
        return float("nan"), float("nan")
    stat = (abs(b - c) - 1) ** 2 / n
    p = chi2_dist.sf(stat, df=1)
    return stat, p


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def load_answers(fname):
    with open(f"{BASE}/answers/{fname}") as f:
        return {a["question_id"]: a["text"] for a in (json.loads(line) for line in f)}


def analyze_pairs(ans_map, gt, pairs, parser):
    """
    Classify each pair into the McNemar 2x2 cells.

    Returns
    -------
    reg_correct : bool array
    flip_correct : bool array
    a : both correct
    b : reg correct, flip wrong
    c : reg wrong, flip correct
    d : both wrong
    skipped : pairs missing from ans_map
    """
    reg_correct, flip_correct = [], []
    skipped = 0

    for reg_id, flip_id in pairs:
        if reg_id not in ans_map or flip_id not in ans_map:
            skipped += 1
            continue

        pred_reg = parser(ans_map[reg_id])
        pred_flip = parser(ans_map[flip_id])

        # Treat parse failures as wrong
        rc = pred_reg is not None and pred_reg == gt[reg_id]
        fc = pred_flip is not None and pred_flip == gt[flip_id]

        reg_correct.append(rc)
        flip_correct.append(fc)

    reg_correct = np.array(reg_correct, dtype=bool)
    flip_correct = np.array(flip_correct, dtype=bool)

    a = int((reg_correct & flip_correct).sum())
    b = int((reg_correct & ~flip_correct).sum())
    c = int((~reg_correct & flip_correct).sum())
    d = int((~reg_correct & ~flip_correct).sum())

    return reg_correct, flip_correct, a, b, c, d, skipped


def make_row(model, prompt, reg_correct, flip_correct, a, b, c, d):
    n = len(reg_correct)

    def acc_stats(correct_arr):
        n_c = int(correct_arr.sum())
        acc = n_c / n if n else float("nan")
        lo, hi = wilson_ci(n_c, n)
        return acc, lo, hi

    reg_acc, reg_lo, reg_hi = acc_stats(reg_correct)
    flip_acc, flip_lo, flip_hi = acc_stats(flip_correct)
    stat, p = mcnemar(b, c)

    return {
        "model": model,
        "prompt": prompt if prompt is not None else "",
        "n_pairs": n,
        "reg_acc": reg_acc,
        "reg_ci_lo": reg_lo,
        "reg_ci_hi": reg_hi,
        "flip_acc": flip_acc,
        "flip_ci_lo": flip_lo,
        "flip_ci_hi": flip_hi,
        "both_correct": a,
        "reg_only": b,
        "flip_only": c,
        "both_wrong": d,
        "both_correct_pct": a / n if n else float("nan"),
        "both_wrong_pct": d / n if n else float("nan"),
        "mcnemar_stat": stat,
        "mcnemar_p": p,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Ground truth (regular + flip)
    with open(f"{BASE}/left_right_answers.json") as f:
        gt_all = json.load(f)
    gt = {a["question_id"]: a["answer"] for a in gt_all}

    # Filter: both reg and flip must have keep_filter == 1
    df_meta = pd.read_csv(f"{BASE}/3DSR_left_right_eval_filtered.csv")
    keep_ids = set(df_meta[df_meta["keep_filter"] == 1]["index"].tolist())

    pairs = []
    for qid in gt:
        if "flip" in qid:
            continue
        flip_id = qid + "-flip"
        if qid in keep_ids and flip_id in keep_ids and flip_id in gt:
            pairs.append((qid, flip_id))

    print(f"Valid pairs (both pass keep_filter): {len(pairs)}")

    configs = [
        ("base",            None,      "answers_base.jsonl",     parse_freeform),
        ("rotation-text",   "direct",  "answers_text.jsonl",     parse_direct),
        ("rotation-text",   "cot",     "answers_text_CoT.jsonl", parse_cot),
        ("rotation-tokens", "direct",  "answers_best.jsonl",     parse_direct),
        ("rotation-tokens", "cot",     "answers_cot_best.jsonl", parse_cot),
    ]

    rows = []
    for model, prompt, fname, parser in configs:
        ans_map = load_answers(fname)
        reg_c, flip_c, a, b, c, d, skipped = analyze_pairs(ans_map, gt, pairs, parser)
        if skipped:
            print(f"  [{model}/{prompt or 'none'}] {skipped} pairs missing from answer file")
        rows.append(make_row(model, prompt, reg_c, flip_c, a, b, c, d))

    df = pd.DataFrame(rows)

    out_path = "data/evals/3DSR/pair_analysis.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows → {out_path}\n")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.3f}".format)

    print("=== Accuracy: regular vs flipped (Wilson 95% CI) ===")
    acc_cols = ["model", "prompt", "n_pairs",
                "reg_acc", "reg_ci_lo", "reg_ci_hi",
                "flip_acc", "flip_ci_lo", "flip_ci_hi"]
    print(df[acc_cols].to_string(index=False))

    print("\n=== McNemar 2×2 contingency table ===")
    print("  (b = reg✓ flip✗,  c = reg✗ flip✓ — these drive the test)")
    ct_cols = ["model", "prompt", "n_pairs",
               "both_correct", "reg_only", "flip_only", "both_wrong",
               "both_correct_pct", "both_wrong_pct",
               "mcnemar_stat", "mcnemar_p"]
    print(df[ct_cols].to_string(index=False))


if __name__ == "__main__":
    main()
