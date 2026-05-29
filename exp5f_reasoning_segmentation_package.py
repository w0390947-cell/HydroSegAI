#!/usr/bin/env python3

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from segformer_potsdam_common import calculate_metrics, colorize_label, load_yaml_config, save_config_snapshot


EXPERIMENT_NAME = "Exp5-F: reasoning segmentation package"
DEFAULT_CONFIG_PATH = "exp5f_reasoning_segmentation_package.yaml"


def ensure_dirs(output_dir):
    output_dir = Path(output_dir)
    for subdir in [
        "final_predictions",
        "review_priority",
        "candidate_components",
        "overlays",
        "manifests",
        "metrics",
    ]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    return output_dir


def read_id_png(path):
    return np.asarray(Image.open(path)).astype(np.uint8)


def read_float_png(path):
    return np.asarray(Image.open(path), dtype=np.float32) / 255.0


def read_mask(path):
    return np.asarray(Image.open(path)) > 0


def save_uint8(path, array):
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(path)


def find_image_names(segformer_eval_dir):
    prediction_dir = Path(segformer_eval_dir) / "predictions"
    return sorted(path.name.replace("_prediction_id.png", "") for path in prediction_dir.glob("*_prediction_id.png"))


def load_recommendations(proposal_audit_dir):
    path = Path(proposal_audit_dir) / "metrics" / "proposal_recommendations.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_review_priority(candidate, entropy, margin, boundary, config):
    review_config = config.get("review_priority", {})
    candidate_weight = float(review_config.get("candidate_weight", 0.35))
    entropy_weight = float(review_config.get("entropy_weight", 0.30))
    low_margin_weight = float(review_config.get("low_margin_weight", 0.20))
    boundary_weight = float(review_config.get("boundary_weight", 0.15))
    total_weight = max(candidate_weight + entropy_weight + low_margin_weight + boundary_weight, 1e-6)

    low_margin = 1.0 - margin
    priority = (
        candidate_weight * candidate.astype(np.float32)
        + entropy_weight * entropy
        + low_margin_weight * low_margin
        + boundary_weight * boundary.astype(np.float32)
    ) / total_weight
    priority = np.clip(priority, 0.0, 1.0)
    high = priority >= float(review_config.get("high_priority_threshold", 0.60))
    medium = priority >= float(review_config.get("medium_priority_threshold", 0.35))
    return priority, high, medium


def create_priority_overlay(prediction, priority):
    base = colorize_label(prediction).astype(np.float32)
    heat = np.zeros_like(base)
    heat[:, :, 0] = 255
    heat[:, :, 1] = np.clip(255 * (1.0 - priority), 0, 255)
    alpha = np.clip(priority * 0.75, 0, 0.75)[:, :, None]
    overlay = base * (1.0 - alpha) + heat * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_ratio(num, den):
    return float(num / den) if den else 0.0


def main():
    parser = argparse.ArgumentParser(description="Exp5-F: package reasoning segmentation outputs")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    paths = config.get("paths", {})
    segformer_eval_dir = Path(paths.get("segformer_eval_dir", "results_segformer_potsdam_rgb/eval_heldout_uncertainty"))
    candidate_dir = Path(paths.get("candidate_dir", "results_exp5b_segformer_candidate_regions_heldout"))
    proposal_audit_dir = Path(paths.get("proposal_audit_dir", "results_exp5e_sam3_proposal_quality_audit_heldout"))
    output_dir = ensure_dirs(args.output_dir or paths.get("output_dir", "results_exp5f_reasoning_segmentation_package_heldout"))
    save_config_snapshot(args.config, output_dir)

    image_format = config.get("package", {}).get("image_format", "png")
    include_gt = bool(config.get("package", {}).get("include_ground_truth_metrics", True))
    output_config = config.get("output", {})
    recommendations = load_recommendations(proposal_audit_dir)
    image_names = find_image_names(segformer_eval_dir)
    if not image_names:
        raise RuntimeError(f"No prediction_id files found under {segformer_eval_dir / 'predictions'}")

    per_image_rows = []
    package_manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "segformer_eval_dir": str(segformer_eval_dir),
        "candidate_dir": str(candidate_dir),
        "proposal_audit_dir": str(proposal_audit_dir),
        "output_dir": str(output_dir),
        "final_prediction_source": config.get("package", {}).get("final_prediction_source", "segformer"),
        "proposal_recommendations": recommendations,
        "images": [],
    }

    totals = {
        "valid_pixels": 0,
        "candidate_pixels": 0,
        "high_priority_pixels": 0,
        "medium_priority_pixels": 0,
        "error_pixels": 0,
        "high_priority_error_pixels": 0,
        "medium_priority_error_pixels": 0,
    }

    for image_name in tqdm(image_names, desc="Package reasoning outputs"):
        pred_path = segformer_eval_dir / "predictions" / f"{image_name}_prediction_id.png"
        label_path = segformer_eval_dir / "predictions" / f"{image_name}_ground_truth_id.png"
        prediction = read_id_png(pred_path)
        entropy = read_float_png(segformer_eval_dir / "uncertainty" / f"{image_name}_entropy.png")
        margin = read_float_png(segformer_eval_dir / "uncertainty" / f"{image_name}_margin.png")
        confidence = read_float_png(segformer_eval_dir / "uncertainty" / f"{image_name}_confidence.png")
        candidate = read_mask(candidate_dir / "masks" / f"{image_name}_candidate.png")
        high_entropy = read_mask(candidate_dir / "masks" / f"{image_name}_high_entropy.png")
        low_margin = read_mask(candidate_dir / "masks" / f"{image_name}_low_margin.png")
        boundary = read_mask(candidate_dir / "masks" / f"{image_name}_boundary_band.png")

        priority, high_priority, medium_priority = build_review_priority(candidate, entropy, margin, boundary, config)
        final_prediction = prediction.copy()

        save_uint8(output_dir / "final_predictions" / f"{image_name}_final_id.{image_format}", final_prediction)
        save_uint8(output_dir / "final_predictions" / f"{image_name}_final.{image_format}", colorize_label(final_prediction))
        save_uint8(output_dir / "review_priority" / f"{image_name}_priority.{image_format}", priority * 255.0)
        save_uint8(output_dir / "review_priority" / f"{image_name}_high_priority.{image_format}", high_priority.astype(np.uint8) * 255)
        save_uint8(output_dir / "review_priority" / f"{image_name}_medium_priority.{image_format}", medium_priority.astype(np.uint8) * 255)

        if output_config.get("save_component_masks", True):
            save_uint8(output_dir / "candidate_components" / f"{image_name}_candidate.{image_format}", candidate.astype(np.uint8) * 255)
            save_uint8(output_dir / "candidate_components" / f"{image_name}_high_entropy.{image_format}", high_entropy.astype(np.uint8) * 255)
            save_uint8(output_dir / "candidate_components" / f"{image_name}_low_margin.{image_format}", low_margin.astype(np.uint8) * 255)
            save_uint8(output_dir / "candidate_components" / f"{image_name}_boundary_band.{image_format}", boundary.astype(np.uint8) * 255)
        if output_config.get("save_overlays", True):
            save_uint8(output_dir / "overlays" / f"{image_name}_priority_overlay.{image_format}", create_priority_overlay(final_prediction, priority))

        valid = np.ones(prediction.shape, dtype=bool)
        image_metrics = {}
        if include_gt and label_path.exists():
            label = read_id_png(label_path)
            valid = label != 255
            metrics = calculate_metrics(final_prediction, label)
            error = valid & (final_prediction != label)
            image_metrics = {
                "overall_accuracy": metrics["overall_accuracy"],
                "mean_iou": metrics["mean_iou"],
                "mean_f1": metrics["mean_f1"],
                "error_pixels": int(error.sum()),
                "high_priority_error_pixels": int((high_priority & error).sum()),
                "medium_priority_error_pixels": int((medium_priority & error).sum()),
                "high_priority_error_coverage": safe_ratio(int((high_priority & error).sum()), int(error.sum())),
                "medium_priority_error_coverage": safe_ratio(int((medium_priority & error).sum()), int(error.sum())),
            }
            totals["error_pixels"] += int(error.sum())
            totals["high_priority_error_pixels"] += int((high_priority & error).sum())
            totals["medium_priority_error_pixels"] += int((medium_priority & error).sum())

        valid_pixels = int(valid.sum())
        candidate_pixels = int((candidate & valid).sum())
        high_priority_pixels = int((high_priority & valid).sum())
        medium_priority_pixels = int((medium_priority & valid).sum())
        row = {
            "image_name": image_name,
            "valid_pixels": valid_pixels,
            "candidate_pixels": candidate_pixels,
            "candidate_ratio": safe_ratio(candidate_pixels, valid_pixels),
            "high_priority_pixels": high_priority_pixels,
            "high_priority_ratio": safe_ratio(high_priority_pixels, valid_pixels),
            "medium_priority_pixels": medium_priority_pixels,
            "medium_priority_ratio": safe_ratio(medium_priority_pixels, valid_pixels),
            "entropy_mean": float(entropy[valid].mean()) if valid_pixels else 0.0,
            "margin_mean": float(margin[valid].mean()) if valid_pixels else 0.0,
            "confidence_mean": float(confidence[valid].mean()) if valid_pixels else 0.0,
        }
        row.update(image_metrics)
        per_image_rows.append(row)

        totals["valid_pixels"] += valid_pixels
        totals["candidate_pixels"] += candidate_pixels
        totals["high_priority_pixels"] += high_priority_pixels
        totals["medium_priority_pixels"] += medium_priority_pixels

        image_manifest = {
            "image_name": image_name,
            "final_prediction": str(output_dir / "final_predictions" / f"{image_name}_final_id.{image_format}"),
            "review_priority": str(output_dir / "review_priority" / f"{image_name}_priority.{image_format}"),
            "priority_overlay": str(output_dir / "overlays" / f"{image_name}_priority_overlay.{image_format}"),
            "metrics": row,
            "interpretation": {
                "final_prediction_role": "stable coarse semantic result",
                "review_priority_role": "areas recommended for review, explanation, or optional refinement",
                "sam3_proposal_role": "candidate object/risk explanation according to Exp5-E audit",
            },
        }
        package_manifest["images"].append(image_manifest)
        if output_config.get("save_manifests", True):
            with open(output_dir / "manifests" / f"{image_name}_manifest.json", "w", encoding="utf-8") as f:
                json.dump(image_manifest, f, indent=2, ensure_ascii=False)

    dataset_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "date": package_manifest["date"],
        "num_images": len(image_names),
        "totals": totals,
        "ratios": {
            "candidate_ratio": safe_ratio(totals["candidate_pixels"], totals["valid_pixels"]),
            "high_priority_ratio": safe_ratio(totals["high_priority_pixels"], totals["valid_pixels"]),
            "medium_priority_ratio": safe_ratio(totals["medium_priority_pixels"], totals["valid_pixels"]),
            "high_priority_error_coverage": safe_ratio(totals["high_priority_error_pixels"], totals["error_pixels"]),
            "medium_priority_error_coverage": safe_ratio(totals["medium_priority_error_pixels"], totals["error_pixels"]),
        },
        "proposal_recommendations": recommendations,
        "note": (
            "Final prediction is not overwritten by SAM3. SAM3 audit outputs are used as "
            "object proposal and risk explanation guidance."
        ),
    }
    package_manifest["dataset_summary"] = dataset_summary

    write_csv(output_dir / "metrics" / "reasoning_package_per_image.csv", per_image_rows)
    with open(output_dir / "metrics" / "reasoning_package_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2, ensure_ascii=False)
    with open(output_dir / "manifests" / "package_manifest.json", "w", encoding="utf-8") as f:
        json.dump(package_manifest, f, indent=2, ensure_ascii=False)

    print(
        "Exp5-F complete: "
        f"images={len(image_names)}, "
        f"candidate_ratio={dataset_summary['ratios']['candidate_ratio']:.4f}, "
        f"high_priority_ratio={dataset_summary['ratios']['high_priority_ratio']:.4f}, "
        f"high_priority_error_coverage={dataset_summary['ratios']['high_priority_error_coverage']:.4f}"
    )


if __name__ == "__main__":
    main()
