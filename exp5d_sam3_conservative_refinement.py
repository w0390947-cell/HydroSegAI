#!/usr/bin/env python3

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_exp5d")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from exp1_sam3_potsdam_zeroshot_baseline import SAM3_AVAILABLE, load_yaml_config, save_config_snapshot
from exp2a_sam3_potsdam_no_background_prompt import raise_sam3_unavailable_error
from exp5c_sam3_candidate_patch_validation import (
    build_evaluator,
    evaluate_sam_masks,
    load_eval_arrays,
    resize_mask,
    resolve_image_ids,
    select_patches,
    generate_candidate_patches,
)
from segformer_potsdam_common import (
    CLASS_INFO,
    IGNORE_INDEX,
    NUM_CLASSES,
    calculate_metrics,
    colorize_label,
    compute_metrics_from_confusion_matrix,
    crop_with_padding,
)


EXPERIMENT_NAME = "Exp5-D: conservative SAM3 refinement"
DEFAULT_CONFIG_PATH = "exp5d_sam3_conservative_refinement.yaml"


def ensure_dirs(output_dir):
    output_dir = Path(output_dir)
    for subdir in ["metrics", "predictions", "change_maps", "patch_visualizations"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    return output_dir


def load_gray_float(path):
    return np.asarray(Image.open(path), dtype=np.float32) / 255.0


def image_id_to_name(image_id):
    return f"{image_id}_RGB"


def load_uncertainty_maps(segformer_eval_dir, image_name):
    uncertainty_dir = Path(segformer_eval_dir) / "uncertainty"
    return {
        "confidence": load_gray_float(uncertainty_dir / f"{image_name}_confidence.png"),
        "entropy": load_gray_float(uncertainty_dir / f"{image_name}_entropy.png"),
        "margin": load_gray_float(uncertainty_dir / f"{image_name}_margin.png"),
    }


def uncertainty_gate(maps, rule):
    entropy_ok = maps["entropy"] >= float(rule.get("entropy_min", 0.15))
    margin_ok = maps["margin"] <= float(rule.get("margin_max", 0.80))
    confidence_ok = maps["confidence"] <= float(rule.get("confidence_max", 0.98))
    mode = str(rule.get("mode", "any")).lower()
    if mode == "all":
        return entropy_ok & margin_ok & confidence_ok
    if mode != "any":
        raise ValueError(f"Unsupported uncertainty_gate.mode={mode!r}")
    return entropy_ok | margin_ok | confidence_ok


def confusion_matrix_from_metrics(metrics):
    return np.asarray(metrics["confusion_matrix"], dtype=np.int64)


def summarize_delta(before, after):
    return {
        "delta_overall_accuracy": after["overall_accuracy"] - before["overall_accuracy"],
        "delta_mean_iou": after["mean_iou"] - before["mean_iou"],
        "delta_mean_f1": after["mean_f1"] - before["mean_f1"],
        "delta_frequency_weighted_iou": after["frequency_weighted_iou"] - before["frequency_weighted_iou"],
        "delta_iou_per_class": [
            after_value - before_value
            for before_value, after_value in zip(before["iou_per_class"], after["iou_per_class"])
        ],
        "delta_recall_per_class": [
            after_value - before_value
            for before_value, after_value in zip(before["recall_per_class"], after["recall_per_class"])
        ],
        "delta_precision_per_class": [
            after_value - before_value
            for before_value, after_value in zip(before["precision_per_class"], after["precision_per_class"])
        ],
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_change_map(path, base_prediction, refined_prediction, label):
    valid = label != IGNORE_INDEX
    changed = valid & (base_prediction != refined_prediction)
    good = changed & (base_prediction != label) & (refined_prediction == label)
    bad = changed & (base_prediction == label) & (refined_prediction != label)
    neutral = changed & ~(good | bad)

    image = np.zeros((*label.shape, 3), dtype=np.uint8)
    image[good] = [0, 200, 0]
    image[bad] = [220, 0, 0]
    image[neutral] = [255, 210, 0]
    image[valid & ~changed] = [30, 30, 30]
    image[~valid] = [0, 0, 0]
    Image.fromarray(image).save(path)


def save_patch_visualization(output_dir, patch_key, image_patch, base_patch, refined_patch, accepted_pixels):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image_patch[:, :, :3])
    axes[0].set_title("RGB patch")
    axes[1].imshow(colorize_label(base_patch))
    axes[1].set_title("SegFormer")
    axes[2].imshow(colorize_label(refined_patch))
    axes[2].set_title("Refined")
    overlay = colorize_label(base_patch).astype(np.float32)
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255
    overlay[accepted_pixels] = 0.35 * overlay[accepted_pixels] + 0.65 * green[accepted_pixels]
    axes[3].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    axes[3].set_title("Accepted SAM3 pixels")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "patch_visualizations" / f"{patch_key}.png", dpi=130)
    plt.close(fig)


def build_selected_patch_records(exp5c_config, max_total_patches):
    evaluation = exp5c_config.get("evaluation", {})
    image_processing = exp5c_config.get("image_processing", {})
    patch_size = int(image_processing.get("patch_size", 1008))
    stride = int(image_processing.get("stride", 672))
    min_candidate_ratio = float(evaluation.get("min_candidate_ratio", 0.08))
    max_patches_per_image = int(evaluation.get("max_patches_per_image", 6))
    sort_by = str(evaluation.get("sort_patches_by", "candidate_ratio"))

    selected_records = []
    proposal_rows = []
    for image_id in tqdm(resolve_image_ids(exp5c_config), desc="Build patch proposals"):
        image_name, _, seg_prediction, label, candidate = load_eval_arrays(exp5c_config, image_id)
        patches = generate_candidate_patches(candidate, seg_prediction, label, patch_size, stride, min_candidate_ratio)
        selected = select_patches(patches, sort_by, max_patches_per_image)
        for rank, patch_info in enumerate(selected):
            row = dict(patch_info)
            row.update({"image_id": image_id, "image_name": image_name, "rank": rank})
            selected_records.append((image_id, image_name, row))

    if max_total_patches is not None and max_total_patches >= 0:
        selected_records = sorted(
            selected_records,
            key=lambda item: item[2].get(sort_by, item[2]["candidate_ratio"]),
            reverse=True,
        )[:max_total_patches]

    for _, _, row in selected_records:
        proposal_rows.append(row)
    return selected_records, proposal_rows, patch_size


def apply_predictions_to_patch(
    predictions,
    patch_context,
    refinement_config,
    patch_refined,
    patch_priority,
):
    accepted_class_ids = {int(x) for x in refinement_config.get("accepted_class_ids", [])}
    score_threshold = float(refinement_config.get("score_threshold", 0.55))
    min_mask_area = int(refinement_config.get("min_mask_area", 16))
    max_mask_area_fraction = float(refinement_config.get("max_mask_area_fraction", 0.08))
    require_candidate_region = bool(refinement_config.get("require_candidate_region", True))
    require_uncertainty_gate = bool(refinement_config.get("require_uncertainty_gate", True))
    apply_only_diff = bool(refinement_config.get("apply_only_where_prediction_differs", True))
    priority_mode = str(refinement_config.get("mask_priority", "score")).lower()

    label = patch_context["label"]
    base_prediction = patch_context["seg_prediction"]
    candidate = patch_context["candidate"]
    uncertain = patch_context["uncertain"]
    valid = label != IGNORE_INDEX
    valid_pixels = int(valid.sum())
    max_mask_area = max(min_mask_area, int(valid_pixels * max_mask_area_fraction))

    masks = predictions.get("masks", [])
    scores = predictions.get("scores", [])
    classes = predictions.get("classes", [])
    accepted_union = np.zeros(label.shape, dtype=bool)
    rows = []

    for idx, raw_mask in enumerate(masks):
        class_id = int(classes[idx])
        score = float(scores[idx])
        resized = resize_mask(raw_mask, label.shape) & valid
        raw_area = int(resized.sum())
        gated = resized.copy()
        if require_candidate_region:
            gated &= candidate
        if require_uncertainty_gate:
            gated &= uncertain
        if apply_only_diff:
            gated &= base_prediction != class_id
        eval_area = int(gated.sum())

        reject_reason = ""
        if class_id not in accepted_class_ids:
            reject_reason = "class_not_accepted"
        elif score < score_threshold:
            reject_reason = "score_below_threshold"
        elif eval_area < min_mask_area:
            reject_reason = "area_too_small"
        elif eval_area > max_mask_area:
            reject_reason = "area_too_large"

        accepted = reject_reason == ""
        changed_pixels = 0
        if accepted:
            priority = score if priority_mode == "score" else 1.0
            update = gated & (priority >= patch_priority)
            patch_refined[update] = class_id
            patch_priority[update] = priority
            accepted_union |= update
            changed_pixels = int(update.sum())

        rows.append(
            {
                "mask_index": idx,
                "class_id": class_id,
                "score": score,
                "raw_area": raw_area,
                "gated_area": eval_area,
                "accepted": accepted,
                "reject_reason": reject_reason,
                "changed_pixels": changed_pixels,
            }
        )
    return accepted_union, rows


def evaluate_change(base_prediction, refined_prediction, label):
    valid = label != IGNORE_INDEX
    changed = valid & (base_prediction != refined_prediction)
    improved = changed & (base_prediction != label) & (refined_prediction == label)
    degraded = changed & (base_prediction == label) & (refined_prediction != label)
    still_wrong_changed = changed & (base_prediction != label) & (refined_prediction != label)
    return {
        "changed_pixels": int(changed.sum()),
        "improved_pixels": int(improved.sum()),
        "degraded_pixels": int(degraded.sum()),
        "still_wrong_changed_pixels": int(still_wrong_changed.sum()),
        "change_precision": float(improved.sum() / changed.sum()) if changed.any() else 0.0,
        "degradation_rate": float(degraded.sum() / changed.sum()) if changed.any() else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Exp5-D: conservative SAM3 refinement validation")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-total-patches", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    paths = config.get("paths", {})
    output_dir = ensure_dirs(args.output_dir or paths.get("output_dir", "results_exp5d_sam3_conservative_refinement_heldout"))
    save_config_snapshot(args.config, output_dir)

    exp5c_config_path = paths.get("exp5c_config", "exp5c_sam3_candidate_patch_validation.yaml")
    exp5c_config = load_yaml_config(exp5c_config_path)
    max_total_patches = args.max_total_patches
    if max_total_patches is None:
        max_total_patches = int(config.get("evaluation", {}).get("max_total_patches", 60))
    selected_records, proposal_rows, patch_size = build_selected_patch_records(exp5c_config, max_total_patches)
    write_csv(output_dir / "metrics" / "candidate_patch_proposals.csv", proposal_rows)

    run_context = {
        "experiment_name": EXPERIMENT_NAME,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": args.config,
        "exp5c_config_path": exp5c_config_path,
        "output_dir": str(output_dir),
        "num_selected_patches": len(selected_records),
        "patch_size": patch_size,
        "max_total_patches": max_total_patches,
        "dry_run": bool(args.dry_run or config.get("evaluation", {}).get("dry_run", False)),
        "refinement": config.get("refinement", {}),
    }
    with open(output_dir / "metrics" / "run_context.json", "w", encoding="utf-8") as f:
        json.dump(run_context, f, indent=2, ensure_ascii=False)

    if run_context["dry_run"]:
        print(f"Dry run complete: selected_patches={len(selected_records)}")
        return

    if not SAM3_AVAILABLE:
        raise_sam3_unavailable_error(EXPERIMENT_NAME)

    evaluator, text_prompts, _ = build_evaluator(exp5c_config, output_dir)
    refinement_config = config.get("refinement", {})
    uncertainty_rule = refinement_config.get("uncertainty_gate", {})
    segformer_eval_dir = paths.get("segformer_eval_dir", exp5c_config.get("paths", {}).get("segformer_eval_dir"))
    output_config = config.get("output", {})
    image_format = output_config.get("image_format", "png")

    image_cache = {}
    refined_by_image = {}
    priority_by_image = {}
    uncertainty_cache = {}
    patch_rows = []
    mask_rows = []

    for global_index, (image_id, image_name, patch_info) in enumerate(tqdm(selected_records, desc="Run SAM3 refinement")):
        if image_id not in image_cache:
            image_cache[image_id] = load_eval_arrays(exp5c_config, image_id)
            _, _, base_prediction, _, _ = image_cache[image_id]
            refined_by_image[image_id] = base_prediction.copy()
            priority_by_image[image_id] = np.full(base_prediction.shape, -np.inf, dtype=np.float32)
            uncertainty_cache[image_id] = load_uncertainty_maps(segformer_eval_dir, image_name)

        _, image, base_prediction, label, candidate = image_cache[image_id]
        refined = refined_by_image[image_id]
        priority = priority_by_image[image_id]
        uncertainty = uncertainty_cache[image_id]

        x, y = int(patch_info["x"]), int(patch_info["y"])
        image_patch = crop_with_padding(image, x, y, patch_size, 0)
        base_patch = crop_with_padding(base_prediction, x, y, patch_size, 5)
        label_patch = crop_with_padding(label, x, y, patch_size, IGNORE_INDEX)
        candidate_patch = crop_with_padding(candidate.astype(np.uint8), x, y, patch_size, 0).astype(bool)
        uncertainty_patch = {
            key: crop_with_padding(value, x, y, patch_size, 0.0)
            for key, value in uncertainty.items()
        }
        uncertain_patch = uncertainty_gate(uncertainty_patch, uncertainty_rule)

        predictions = evaluator.predict_patch(
            image_patch,
            text_prompts,
            patch_index=global_index,
            patch_position=(x, y),
        )

        patch_h = int(patch_info["patch_h"])
        patch_w = int(patch_info["patch_w"])
        refined_view = crop_with_padding(refined, x, y, patch_size, 5)
        priority_view = crop_with_padding(priority, x, y, patch_size, -np.inf)
        refined_view_before = refined_view.copy()
        patch_context = {
            "label": label_patch,
            "seg_prediction": base_patch,
            "candidate": candidate_patch,
            "uncertain": uncertain_patch,
        }
        accepted_union, local_mask_rows = apply_predictions_to_patch(
            predictions,
            patch_context,
            refinement_config,
            refined_view,
            priority_view,
        )

        refined[y : y + patch_h, x : x + patch_w] = refined_view[:patch_h, :patch_w]
        priority[y : y + patch_h, x : x + patch_w] = priority_view[:patch_h, :patch_w]
        refined_patch_after = refined_view
        patch_change = evaluate_change(refined_view_before, refined_patch_after, label_patch)
        patch_sam_summary, _ = evaluate_sam_masks(predictions, patch_context, restrict_to_candidate=True)
        patch_key = f"{image_name}_patch_{global_index:04d}_x{x}_y{y}"

        row = dict(patch_info)
        row.update(
            {
                "image_id": image_id,
                "image_name": image_name,
                "global_index": global_index,
                "patch_key": patch_key,
                "num_sam_masks": len(predictions.get("masks", [])),
                "accepted_pixels": int(accepted_union.sum()),
                "accepted_mask_count": sum(1 for r in local_mask_rows if r["accepted"]),
            }
        )
        row.update(patch_change)
        row.update(
            {
                "sam_error_coverage_before_filter": patch_sam_summary["sam_error_coverage"],
                "sam_error_precision_before_filter": patch_sam_summary["sam_error_precision"],
            }
        )
        patch_rows.append(row)

        for mask_row in local_mask_rows:
            out_row = dict(mask_row)
            out_row.update({"image_id": image_id, "image_name": image_name, "patch_key": patch_key, "global_index": global_index})
            mask_rows.append(out_row)

        if output_config.get("save_patch_visualizations", True):
            save_patch_visualization(output_dir, patch_key, image_patch, base_patch, refined_patch_after, accepted_union)

    for image_id in resolve_image_ids(exp5c_config):
        if image_id in image_cache:
            continue
        image_cache[image_id] = load_eval_arrays(exp5c_config, image_id)
        _, _, base_prediction, _, _ = image_cache[image_id]
        refined_by_image[image_id] = base_prediction.copy()

    image_rows = []
    baseline_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    refined_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    change_totals = {
        "changed_pixels": 0,
        "improved_pixels": 0,
        "degraded_pixels": 0,
        "still_wrong_changed_pixels": 0,
    }

    for image_id, (image_name, _, base_prediction, label, _) in image_cache.items():
        refined = refined_by_image[image_id]
        baseline_metrics = calculate_metrics(base_prediction, label)
        refined_metrics = calculate_metrics(refined, label)
        baseline_cm += confusion_matrix_from_metrics(baseline_metrics)
        refined_cm += confusion_matrix_from_metrics(refined_metrics)
        change = evaluate_change(base_prediction, refined, label)
        for key in change_totals:
            change_totals[key] += change[key]

        row = {
            "image_id": image_id,
            "image_name": image_name,
            "baseline_oa": baseline_metrics["overall_accuracy"],
            "baseline_miou": baseline_metrics["mean_iou"],
            "refined_oa": refined_metrics["overall_accuracy"],
            "refined_miou": refined_metrics["mean_iou"],
            "delta_oa": refined_metrics["overall_accuracy"] - baseline_metrics["overall_accuracy"],
            "delta_miou": refined_metrics["mean_iou"] - baseline_metrics["mean_iou"],
        }
        row.update(change)
        image_rows.append(row)

        if output_config.get("save_refined_predictions", True):
            Image.fromarray(refined).save(output_dir / "predictions" / f"{image_name}_refined_id.{image_format}")
            Image.fromarray(colorize_label(refined)).save(output_dir / "predictions" / f"{image_name}_refined.{image_format}")
        if output_config.get("save_change_maps", True):
            save_change_map(output_dir / "change_maps" / f"{image_name}_change_map.{image_format}", base_prediction, refined, label)

    baseline_dataset = compute_metrics_from_confusion_matrix(baseline_cm)
    refined_dataset = compute_metrics_from_confusion_matrix(refined_cm)
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "run_context": run_context,
        "class_info": CLASS_INFO,
        "baseline_metrics_on_processed_images": baseline_dataset,
        "refined_metrics_on_processed_images": refined_dataset,
        "delta": summarize_delta(baseline_dataset, refined_dataset),
        "change_totals": {
            **change_totals,
            "change_precision": (
                change_totals["improved_pixels"] / change_totals["changed_pixels"]
                if change_totals["changed_pixels"]
                else 0.0
            ),
            "degradation_rate": (
                change_totals["degraded_pixels"] / change_totals["changed_pixels"]
                if change_totals["changed_pixels"]
                else 0.0
            ),
        },
        "accepted_mask_count": sum(1 for row in mask_rows if row["accepted"]),
        "total_mask_count": len(mask_rows),
    }

    write_csv(output_dir / "metrics" / "refinement_patch_metrics.csv", patch_rows)
    write_csv(output_dir / "metrics" / "refinement_mask_metrics.csv", mask_rows)
    write_csv(output_dir / "metrics" / "refinement_per_image_metrics.csv", image_rows)
    with open(output_dir / "metrics" / "refinement_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        "Exp5-D complete: "
        f"baseline_mIoU={baseline_dataset['mean_iou']:.4f}, "
        f"refined_mIoU={refined_dataset['mean_iou']:.4f}, "
        f"delta_mIoU={summary['delta']['delta_mean_iou']:.4f}, "
        f"changed_pixels={change_totals['changed_pixels']}"
    )


if __name__ == "__main__":
    main()
