"""
Compute accuracy + Wilson CIs for all benchmarks and write to data/evals/results.csv.
Benchmarks: 3DSR, cocoval2017, isle_bricks_v2, perspective_taking.
"""

import json
import re
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

BASE = "data/evals"
RESULTS_CSV = f"{BASE}/results.csv"


def wilson_ci(n_correct, n_total):
    lo, hi = proportion_confint(n_correct, n_total, method="wilson")
    return hi, lo  # (upper, lower)


# ── parsers ───────────────────────────────────────────────────────────────────

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def parse_freeform(text):
    """Base-model free-form: left XOR right present in text."""
    t = text.lower()
    has_left = "left" in t
    has_right = "right" in t
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    return None

def parse_direct(text):
    """Fine-tuned direct: expected output is exactly 'left' or 'right'."""
    t = text.strip().lower()
    return t if t in ("left", "right") else None

def parse_cot(text):
    """CoT: extract conclusion from 'Therefore ... is to the (left|right)'."""
    m = re.search(r'Therefore.*?is to the (left|right)', text, re.IGNORECASE | re.DOTALL)
    return m.group(1).lower() if m else None

def manual_check(text, ground_truth):
    print(f"\n[Parse failed]  ground truth: {ground_truth}")
    print(f"  model output: {text}")
    while True:
        ans = input("  Correct? (y/n): ").strip().lower()
        if ans in ("y", "n"):
            return ans == "y"

def is_correct(pred, ground_truth, text):
    if pred is not None:
        return pred == ground_truth
    return manual_check(text, ground_truth)


# ── 3DSR ─────────────────────────────────────────────────────────────────────

def process_3dsr():
    base = f"{BASE}/3DSR"
    df_meta = pd.read_csv(f"{base}/3DSR_left_right_eval_filtered.csv")
    keep_ids = set(df_meta[df_meta["keep_filter"] == 1]["index"].tolist())

    with open(f"{base}/left_right_answers.json") as f:
        gt_all = json.load(f)
    gt = {
        a["question_id"]: a["answer"]
        for a in gt_all
        if a["question_id"] in keep_ids and "flip" not in a["question_id"]
    }
    qids = list(gt.keys())

    def load_ans(fname):
        raw = read_jsonl(f"{base}/answers/{fname}")
        return {a["question_id"]: a["text"] for a in raw}

    def correct_freeform(ans_map):
        return [is_correct(parse_freeform(ans_map[q]), gt[q], ans_map[q]) for q in qids if q in ans_map]

    def correct_direct(ans_map):
        return [is_correct(parse_direct(ans_map[q]), gt[q], ans_map[q]) for q in qids if q in ans_map]

    def correct_cot(ans_map):
        return [is_correct(parse_cot(ans_map[q]), gt[q], ans_map[q]) for q in qids if q in ans_map]

    rows = []
    rows.append(compute_row("3DSR", "base", None,
                            correct_freeform(load_ans("answers_base.jsonl")), None))
    rows.append(compute_row("3DSR", "embodiment-text", "direct",
                            correct_direct(load_ans("answers_vit_text.jsonl")), None))
    rows.append(compute_row("3DSR", "embodiment-text", "cot",
                            correct_cot(load_ans("answers_vit_text_CoT.jsonl")), None))
    rows.append(compute_row("3DSR", "embodiment-coco", "direct",
                            correct_direct(load_ans("answers_coco.jsonl")), None))
    rows.append(compute_row("3DSR", "embodiment-coco", "cot",
                            correct_cot(load_ans("answers_coco_CoT.jsonl")), None))
    rows.append(compute_row("3DSR", "embodiment-vit", "direct",
                            correct_direct(load_ans("answers_vit.jsonl")), None))
    rows.append(compute_row("3DSR", "embodiment-vit", "cot",
                            correct_cot(load_ans("answers_vit_CoT.jsonl")), None))
    return rows


def compute_row(benchmark, model, prompt, correct_list, _unused):
    correct = np.array(correct_list, dtype=bool)
    n_total = len(correct)
    n_correct = int(correct.sum())
    total = n_correct / n_total
    total_upper, total_lower = wilson_ci(n_correct, n_total)
    return {
        "benchmark": benchmark, "model": model, "prompt": prompt,
        "align": None, "align_upper": None, "align_lower": None,
        "unalign": None, "unalign_upper": None, "unalign_lower": None,
        "total": total, "total_upper": total_upper, "total_lower": total_lower,
    }


# ── cocoval2017 ───────────────────────────────────────────────────────────────

def process_cocoval2017():
    filt_items = open(f"{BASE}/cocoval2017/eval_check_items_filtered.txt").read()
    filt_items_lines = [l.strip() for l in filt_items.splitlines() if l.strip()]

    filt_specs = []
    for item in filt_items_lines:
        item = item.replace(".png", "")
        coco, ref, qry = item.split("_")
        filt_specs.append((coco, ref.lower(), qry.lower()))

    question_ids = []
    with open(f"{BASE}/cocoval2017/questions/direct_pose.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            img_name = obj.get("image", "")
            text = obj.get("text", "").lower()
            qid = obj.get("question_id")
            coco_id = img_name.split(".")[0]
            for cid, ref, qry in filt_specs:
                if cid == coco_id and ref in text and qry in text:
                    question_ids.append(qid)
                    break

    answers_data = read_jsonl(f"{BASE}/cocoval2017/answers.jsonl")
    gt = {a["question_id"]: a["text"] for a in answers_data}
    azimuths = {a["question_id"]: a["azimuth"] for a in answers_data}

    left_right_answers = [gt[qid] for qid in question_ids]
    angles = [azimuths[qid] for qid in question_ids]

    ALIGNED_ANGLES = {0.0, 36.0, 72.0, 288.0, 324.0}
    UNALIGNED_ANGLES = {108.0, 144.0, 180.0, 216.0, 252.0}

    alignment = []
    for a in angles:
        if a in ALIGNED_ANGLES:
            alignment.append(0)
        elif a in UNALIGNED_ANGLES:
            alignment.append(1)
        else:
            alignment.append(2)

    alignment = np.array(alignment)

    def compute_row(benchmark, model, prompt, correct_arr):
        correct_arr = np.array(correct_arr, dtype=bool)
        n_total = len(correct_arr)
        n_correct = int(correct_arr.sum())

        total = n_correct / n_total
        total_upper, total_lower = wilson_ci(n_correct, n_total)

        al = correct_arr[alignment == 0]
        un = correct_arr[alignment == 1]

        if len(al) > 0:
            align_val = al.mean()
            align_upper, align_lower = wilson_ci(int(al.sum()), len(al))
        else:
            align_val = align_upper = align_lower = None

        if len(un) > 0:
            unalign_val = un.mean()
            unalign_upper, unalign_lower = wilson_ci(int(un.sum()), len(un))
        else:
            unalign_val = unalign_upper = unalign_lower = None

        return {
            "benchmark": benchmark,
            "model": model,
            "prompt": prompt,
            "align": align_val,
            "align_upper": align_upper,
            "align_lower": align_lower,
            "unalign": unalign_val,
            "unalign_upper": unalign_upper,
            "unalign_lower": unalign_lower,
            "total": total,
            "total_upper": total_upper,
            "total_lower": total_lower,
        }

    rows = []
    base_dir = f"{BASE}/cocoval2017/answers"
    question_ids_set = set(question_ids)

    def lookup(raw, key="text"):
        """Return answers keyed by question_id, in question_ids order."""
        d = {a["question_id"]: a[key] for a in raw if a["question_id"] in question_ids_set}
        return [d[qid] for qid in question_ids]

    # base
    base_texts = lookup(read_jsonl(f"{base_dir}/answers_base.jsonl"))
    base_correct = [is_correct(parse_freeform(t), gt, t) for t, gt in zip(base_texts, left_right_answers)]
    rows.append(compute_row("cocoval2017", "base", None, base_correct))

    # embodiment-text direct
    t_resp = lookup(read_jsonl(f"{base_dir}/answers_vit_text.jsonl"))
    t_correct = [is_correct(parse_direct(t), gt, t) for t, gt in zip(t_resp, left_right_answers)]
    rows.append(compute_row("cocoval2017", "embodiment-text", "direct", t_correct))

    # embodiment-text cot
    tcot_texts = lookup(read_jsonl(f"{base_dir}/answers_vit_text_CoT.jsonl"))
    tcot_correct = [is_correct(parse_cot(t), gt, t) for t, gt in zip(tcot_texts, left_right_answers)]
    rows.append(compute_row("cocoval2017", "embodiment-text", "cot", tcot_correct))

    # embodiment-coco direct
    c_resp = lookup(read_jsonl(f"{base_dir}/answers_coco.jsonl"))
    c_correct = [is_correct(parse_direct(t), gt, t) for t, gt in zip(c_resp, left_right_answers)]
    rows.append(compute_row("cocoval2017", "embodiment-coco", "direct", c_correct))

    # embodiment-coco cot
    ccot_texts = lookup(read_jsonl(f"{base_dir}/answers_coco_CoT.jsonl"))
    ccot_correct = [is_correct(parse_cot(t), gt, t) for t, gt in zip(ccot_texts, left_right_answers)]
    rows.append(compute_row("cocoval2017", "embodiment-coco", "cot", ccot_correct))

    # embodiment-vit direct
    v_resp = lookup(read_jsonl(f"{base_dir}/answers_vit.jsonl"))
    v_correct = [is_correct(parse_direct(t), gt, t) for t, gt in zip(v_resp, left_right_answers)]
    rows.append(compute_row("cocoval2017", "embodiment-vit", "direct", v_correct))

    # embodiment-vit cot
    vcot_texts = lookup(read_jsonl(f"{base_dir}/answers_vit_CoT.jsonl"))
    vcot_correct = [is_correct(parse_cot(t), gt, t) for t, gt in zip(vcot_texts, left_right_answers)]
    rows.append(compute_row("cocoval2017", "embodiment-vit", "cot", vcot_correct))

    return rows


# ── isle_bricks_v2 ────────────────────────────────────────────────────────────

def process_isle_bricks():
    ib_dir = f"{BASE}/isle_bricks_v2"

    with open(f"{ib_dir}/answers.txt") as f:
        answers = [l.strip().split(",") for l in f if l.strip()]
    left_right_answers = [a[1].strip().lower() for a in answers]

    with open(f"{ib_dir}/alignment.txt") as f:
        alignment = np.array([l.strip() for l in f if l.strip()])

    # exclude perpendicular (alignment == '2')
    mask = alignment != "2"
    lr_excl = [lr for lr, m in zip(left_right_answers, mask) if m]
    al_excl = alignment[mask]  # '0' aligned, '1' unaligned

    def eval_masked(texts, parser):
        """Parse and evaluate items, returning only those where mask is True."""
        return [
            is_correct(parser(text), gt, text)
            for text, gt, keep in zip(texts, left_right_answers, mask)
            if keep
        ]

    def compute_row(benchmark, model, prompt, correct_excl):
        correct = np.array(correct_excl, dtype=bool)
        n_total = len(correct)
        n_correct = int(correct.sum())
        total = n_correct / n_total
        total_upper, total_lower = wilson_ci(n_correct, n_total)

        al = correct[al_excl == "0"]
        un = correct[al_excl == "1"]

        align_val = al.mean() if len(al) > 0 else None
        align_upper, align_lower = (wilson_ci(int(al.sum()), len(al)) if len(al) > 0 else (None, None))

        unalign_val = un.mean() if len(un) > 0 else None
        unalign_upper, unalign_lower = (wilson_ci(int(un.sum()), len(un)) if len(un) > 0 else (None, None))

        return {
            "benchmark": benchmark,
            "model": model,
            "prompt": prompt,
            "align": align_val,
            "align_upper": align_upper,
            "align_lower": align_lower,
            "unalign": unalign_val,
            "unalign_upper": unalign_upper,
            "unalign_lower": unalign_lower,
            "total": total,
            "total_upper": total_upper,
            "total_lower": total_lower,
        }

    rows = []
    ans_dir = f"{ib_dir}/answers"

    # base
    base_raw = read_jsonl(f"{ans_dir}/vpt_ib_answers_base.jsonl")
    rows.append(compute_row("isle_bricks_v2", "base", None,
                            eval_masked([a["text"] for a in base_raw], parse_freeform)))

    # embodiment-text direct
    t_raw = read_jsonl(f"{ans_dir}/vpt_ib_answers_vit_text.jsonl")
    rows.append(compute_row("isle_bricks_v2", "embodiment-text", "direct",
                            eval_masked([a["text"] for a in t_raw], parse_direct)))

    # embodiment-text cot
    tcot_raw = read_jsonl(f"{ans_dir}/vpt_ib_answers_vit_text_CoT.jsonl")
    rows.append(compute_row("isle_bricks_v2", "embodiment-text", "cot",
                            eval_masked([a["text"] for a in tcot_raw], parse_cot)))

    # embodiment-coco direct
    c_raw = read_jsonl(f"{ans_dir}/vpt_ib_answers_coco.jsonl")
    rows.append(compute_row("isle_bricks_v2", "embodiment-coco", "direct",
                            eval_masked([a["text"] for a in c_raw], parse_direct)))

    # embodiment-coco cot
    ccot_raw = read_jsonl(f"{ans_dir}/vpt_ib_answers_coco_CoT.jsonl")
    rows.append(compute_row("isle_bricks_v2", "embodiment-coco", "cot",
                            eval_masked([a["text"] for a in ccot_raw], parse_cot)))

    # embodiment-vit direct
    v_raw = read_jsonl(f"{ans_dir}/vpt_ib_answers_vit.jsonl")
    rows.append(compute_row("isle_bricks_v2", "embodiment-vit", "direct",
                            eval_masked([a["text"] for a in v_raw], parse_direct)))

    # embodiment-vit cot
    vcot_raw = read_jsonl(f"{ans_dir}/vpt_ib_answers_vit_CoT.jsonl")
    rows.append(compute_row("isle_bricks_v2", "embodiment-vit", "cot",
                            eval_masked([a["text"] for a in vcot_raw], parse_cot)))

    return rows


# ── perspective_taking ────────────────────────────────────────────────────────

def process_perspective_taking():
    pt_dir = "data/evals/perspective_taking/answers"

    true_answers = ["left"] * 12 + ["right"] * 12  # 24 items
    perp_index = {3, 9, 15, 21}
    angles_all = [0, 45, 60, 90, 120, 135, 180, 225, 240, 270, 300, 315] * 2

    # indices to keep (excluding perp)
    keep_idx = [i for i in range(24) if i not in perp_index]
    angles_plot = [angles_all[i] for i in keep_idx]
    true_excl = [true_answers[i] for i in keep_idx]  # 20 items

    ALIGNED_ANGLES = {0, 45, 60, 300, 315}
    alignment = np.array(["aligned" if a in ALIGNED_ANGLES else "unaligned" for a in angles_plot])

    def eval_kept(data, parser):
        return np.array([
            is_correct(parser(data[i]["text"]), true_excl[j], data[i]["text"])
            for j, i in enumerate(keep_idx)
        ], dtype=float)

    def compute_row_from_correct(benchmark, model, prompt, correct_arr):
        correct_arr = np.array(correct_arr)
        n_items = len(correct_arr)

        # For averaged repeats: correct_arr may be floats 0..1 per item
        n_correct_total = int(round(correct_arr.sum()))
        n_total = n_items
        total = correct_arr.mean()
        total_upper, total_lower = wilson_ci(n_correct_total, n_total)

        al = correct_arr[alignment == "aligned"]
        un = correct_arr[alignment == "unaligned"]

        n_al_correct = int(round(al.sum()))
        n_un_correct = int(round(un.sum()))

        align_val = al.mean()
        align_upper, align_lower = wilson_ci(n_al_correct, len(al))

        unalign_val = un.mean()
        unalign_upper, unalign_lower = wilson_ci(n_un_correct, len(un))

        return {
            "benchmark": benchmark,
            "model": model,
            "prompt": prompt,
            "align": align_val,
            "align_upper": align_upper,
            "align_lower": align_lower,
            "unalign": unalign_val,
            "unalign_upper": unalign_upper,
            "unalign_lower": unalign_lower,
            "total": total,
            "total_upper": total_upper,
            "total_lower": total_lower,
        }

    rows = []

    base_data = read_jsonl(f"{pt_dir}/vpt_avatar_answers_base.jsonl")
    rows.append(compute_row_from_correct("perspective_taking", "base", None,
                                        eval_kept(base_data, parse_freeform)))

    vit_data = read_jsonl(f"{pt_dir}/vpt_avatar_answers_vitpose.jsonl")
    rows.append(compute_row_from_correct("perspective_taking", "embodiment-vit", "direct",
                                        eval_kept(vit_data, parse_direct)))

    coco_data = read_jsonl(f"{pt_dir}/vpt_avatar_answers_COCO.jsonl")
    rows.append(compute_row_from_correct("perspective_taking", "embodiment-coco", "direct",
                                        eval_kept(coco_data, parse_direct)))

    vcot_data = read_jsonl(f"{pt_dir}/vpt_avatar_answers_vitpose_cot.jsonl")
    rows.append(compute_row_from_correct("perspective_taking", "embodiment-vit", "cot",
                                        eval_kept(vcot_data, parse_cot)))

    ccot_data = read_jsonl(f"{pt_dir}/vpt_avatar_answers_COCO_CoT.jsonl")
    rows.append(compute_row_from_correct("perspective_taking", "embodiment-coco", "cot",
                                        eval_kept(ccot_data, parse_cot)))

    text_data = read_jsonl(f"{pt_dir}/vpt_avatar_answers_text_vitpose.jsonl")
    rows.append(compute_row_from_correct("perspective_taking", "embodiment-text", "direct",
                                        eval_kept(text_data, parse_direct)))

    tcot_data = read_jsonl(f"{pt_dir}/vpt_avatar_answers_text_vitpose_CoT.jsonl")
    rows.append(compute_row_from_correct("perspective_taking", "embodiment-text", "cot",
                                        eval_kept(tcot_data, parse_cot)))

    return rows


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Computing 3DSR...")
    dsr_rows = process_3dsr()

    print("Computing cocoval2017...")
    coco_rows = process_cocoval2017()

    print("Computing isle_bricks_v2...")
    ib_rows = process_isle_bricks()

    print("Computing perspective_taking...")
    pt_rows = process_perspective_taking()

    all_rows = dsr_rows + coco_rows + ib_rows + pt_rows

    df = pd.DataFrame(all_rows, columns=[
        "benchmark", "model", "prompt",
        "align", "align_upper", "align_lower",
        "unalign", "unalign_upper", "unalign_lower",
        "total", "total_upper", "total_lower",
    ])

    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {RESULTS_CSV}")
    print(df.to_string())
