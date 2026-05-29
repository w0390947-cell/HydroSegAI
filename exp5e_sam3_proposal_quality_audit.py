#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


EXPERIMENT_NAME = "Exp5-E: SAM3 proposal quality audit"
CLASS_NAMES = {
    0: "impervious surface",
    1: "building",
    2: "low vegetation",
    3: "tree",
    4: "car",
    5: "clutter/background",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value, default=0.0):
    if value in (None, ""):
        return default
    return float(value)


def safe_int(value, default=0):
    if value in (None, ""):
        return default
    return int(float(value))


def bool_value(value):
    return str(value).lower() == "true"


def mask_key(row):
    return (
        row["image_name"],
        row["patch_key"],
        str(row["global_index"]),
        str(row["mask_index"]),
        str(row["class_id"]),
    )


def score_bin(score):
    if score < 0.50:
        return "<0.50"
    if score < 0.55:
        return "0.50-0.55"
    if score < 0.60:
        return "0.55-0.60"
    if score < 0.70:
        return "0.60-0.70"
    if score < 0.80:
        return "0.70-0.80"
    return ">=0.80"


def area_bin(area):
    if area <= 0:
        return "0"
    if area < 16:
        return "1-15"
    if area < 64:
        return "16-63"
    if area < 256:
        return "64-255"
    if area < 1024:
        return "256-1023"
    if area < 10000:
        return "1k-10k"
    if area < 100000:
        return "10k-100k"
    return ">=100k"


def quality_tier(row):
    gt_precision = safe_float(row["gt_class_precision"])
    error_precision = safe_float(row["eval_error_precision"])
    clutter_precision = safe_float(row["eval_clutter_fn_precision"])
    eval_area = safe_int(row["eval_area"])
    class_id = safe_int(row["class_id"])

    if eval_area <= 0:
        return "empty_or_unusable"
    if gt_precision >= 0.80 and 16 <= eval_area <= 10000:
        return "object_proposal_high_precision"
    if gt_precision >= 0.50 and 16 <= eval_area <= 100000:
        return "object_proposal_medium_precision"
    if error_precision >= 0.35 and gt_precision < 0.50:
        return "error_region_explanation"
    if clutter_precision >= 0.20:
        return "clutter_risk_explanation"
    if class_id == 0 and eval_area >= 100000:
        return "large_surface_risky"
    return "low_confidence_or_ambiguous"


def aggregate(rows, group_keys):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups[key].append(row)

    out = []
    for key, items in sorted(groups.items()):
        eval_area = sum(safe_int(r["eval_area"]) for r in items)
        gt_overlap = sum(safe_int(r["gt_class_overlap"]) for r in items)
        error_overlap = sum(safe_int(r["error_overlap"]) for r in items)
        clutter_overlap = sum(safe_int(r["clutter_fn_overlap"]) for r in items)
        accepted = sum(1 for r in items if bool_value(r.get("accepted", "False")))
        changed_pixels = sum(safe_int(r.get("changed_pixels", 0)) for r in items)
        row = {k: v for k, v in zip(group_keys, key)}
        row.update(
            {
                "mask_count": len(items),
                "eval_area": eval_area,
                "gt_class_overlap": gt_overlap,
                "error_overlap": error_overlap,
                "clutter_fn_overlap": clutter_overlap,
                "gt_precision_weighted": gt_overlap / eval_area if eval_area else 0.0,
                "error_precision_weighted": error_overlap / eval_area if eval_area else 0.0,
                "clutter_precision_weighted": clutter_overlap / eval_area if eval_area else 0.0,
                "accepted_mask_count": accepted,
                "changed_pixels": changed_pixels,
                "mean_score": sum(safe_float(r["score"]) for r in items) / len(items),
            }
        )
        out.append(row)
    return out


def merge_mask_rows(exp5c_rows, exp5d_rows):
    exp5d_by_key = {mask_key(row): row for row in exp5d_rows}
    merged = []
    for row in exp5c_rows:
        item = dict(row)
        d_row = exp5d_by_key.get(mask_key(row), {})
        item["gated_area"] = d_row.get("gated_area", "")
        item["accepted"] = d_row.get("accepted", "False")
        item["reject_reason"] = d_row.get("reject_reason", "")
        item["changed_pixels"] = d_row.get("changed_pixels", "0")
        item["class_name"] = CLASS_NAMES.get(safe_int(item["class_id"]), f"class_{item['class_id']}")
        item["score_bin"] = score_bin(safe_float(item["score"]))
        item["eval_area_bin"] = area_bin(safe_int(item["eval_area"]))
        item["gated_area_bin"] = area_bin(safe_int(item["gated_area"])) if item["gated_area"] != "" else "missing"
        item["quality_tier"] = quality_tier(item)
        merged.append(item)
    return merged


def build_recommendations(class_rows, tier_rows, reject_rows):
    recommendations = []

    for row in class_rows:
        class_id = safe_int(row["class_id"])
        gt_precision = safe_float(row["gt_precision_weighted"])
        eval_area = safe_int(row["eval_area"])
        mask_count = safe_int(row["mask_count"])
        if gt_precision >= 0.80 and eval_area < 200000:
            recommendations.append(
                {
                    "scope": f"class_{class_id}",
                    "class_name": CLASS_NAMES.get(class_id, ""),
                    "recommendation": "candidate_object_layer",
                    "reason": "High weighted GT precision and limited total area; suitable for explanation or review candidates.",
                    "mask_count": mask_count,
                    "gt_precision_weighted": gt_precision,
                }
            )
        elif class_id == 0 and eval_area >= 100000:
            recommendations.append(
                {
                    "scope": f"class_{class_id}",
                    "class_name": CLASS_NAMES.get(class_id, ""),
                    "recommendation": "do_not_auto_refine",
                    "reason": "Large surface masks dominate area and have weak semantic precision; keep as risk/explanation only.",
                    "mask_count": mask_count,
                    "gt_precision_weighted": gt_precision,
                }
            )
        elif gt_precision < 0.50:
            recommendations.append(
                {
                    "scope": f"class_{class_id}",
                    "class_name": CLASS_NAMES.get(class_id, ""),
                    "recommendation": "needs_additional_prior",
                    "reason": "Proposal precision is not high enough for automatic semantic overwrite.",
                    "mask_count": mask_count,
                    "gt_precision_weighted": gt_precision,
                }
            )

    for row in tier_rows:
        if row["quality_tier"] == "object_proposal_high_precision":
            recommendations.append(
                {
                    "scope": row["quality_tier"],
                    "class_name": "mixed",
                    "recommendation": "use_as_review_or_interactive_proposals",
                    "reason": "High precision proposals exist, but Exp5-D shows automatic writes still need stronger gates.",
                    "mask_count": safe_int(row["mask_count"]),
                    "gt_precision_weighted": safe_float(row["gt_precision_weighted"]),
                }
            )

    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Exp5-E: audit SAM3 proposal quality without new inference")
    parser.add_argument("--exp5c-dir", default="results_exp5c_sam3_candidate_patch_validation_heldout")
    parser.add_argument("--exp5d-dir", default="results_exp5d_sam3_conservative_refinement_heldout")
    parser.add_argument("--output-dir", default="results_exp5e_sam3_proposal_quality_audit_heldout")
    args = parser.parse_args()

    exp5c_dir = Path(args.exp5c_dir)
    exp5d_dir = Path(args.exp5d_dir)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    exp5c_rows = read_csv(exp5c_dir / "metrics" / "sam_mask_metrics.csv")
    exp5d_rows = read_csv(exp5d_dir / "metrics" / "refinement_mask_metrics.csv")
    merged = merge_mask_rows(exp5c_rows, exp5d_rows)

    by_class = aggregate(merged, ["class_id", "class_name"])
    by_class_score = aggregate(merged, ["class_id", "class_name", "score_bin"])
    by_class_area = aggregate(merged, ["class_id", "class_name", "eval_area_bin"])
    by_tier = aggregate(merged, ["quality_tier"])
    by_reject = aggregate(merged, ["reject_reason"])
    recommendations = build_recommendations(by_class, by_tier, by_reject)

    write_csv(metrics_dir / "proposal_mask_merged.csv", merged)
    write_csv(metrics_dir / "proposal_quality_by_class.csv", by_class)
    write_csv(metrics_dir / "proposal_quality_by_class_score.csv", by_class_score)
    write_csv(metrics_dir / "proposal_quality_by_class_area.csv", by_class_area)
    write_csv(metrics_dir / "proposal_quality_by_tier.csv", by_tier)
    write_csv(metrics_dir / "proposal_quality_by_reject_reason.csv", by_reject)
    write_csv(metrics_dir / "proposal_recommendations.csv", recommendations)

    total_area = sum(safe_int(r["eval_area"]) for r in merged)
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exp5c_dir": str(exp5c_dir),
        "exp5d_dir": str(exp5d_dir),
        "output_dir": str(output_dir),
        "mask_count": len(merged),
        "accepted_mask_count": sum(1 for r in merged if bool_value(r["accepted"])),
        "changed_pixels": sum(safe_int(r["changed_pixels"]) for r in merged),
        "eval_area": total_area,
        "gt_precision_weighted": (
            sum(safe_int(r["gt_class_overlap"]) for r in merged) / total_area if total_area else 0.0
        ),
        "error_precision_weighted": (
            sum(safe_int(r["error_overlap"]) for r in merged) / total_area if total_area else 0.0
        ),
        "clutter_precision_weighted": (
            sum(safe_int(r["clutter_fn_overlap"]) for r in merged) / total_area if total_area else 0.0
        ),
        "class_summary": by_class,
        "tier_summary": by_tier,
        "recommendations": recommendations,
    }
    with open(metrics_dir / "proposal_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        "Exp5-E complete: "
        f"masks={summary['mask_count']}, "
        f"accepted={summary['accepted_mask_count']}, "
        f"gt_precision={summary['gt_precision_weighted']:.4f}, "
        f"error_precision={summary['error_precision_weighted']:.4f}"
    )


if __name__ == "__main__":
    main()
