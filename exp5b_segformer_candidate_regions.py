#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from segformer_potsdam_common import CLASS_INFO, CLASS_IDS, IGNORE_INDEX, colorize_label


DEFAULT_CONFUSION_CLASSES = [0, 1, 2, 5]


def load_id_png(path):
    return np.asarray(Image.open(path)).astype(np.uint8)


def save_mask(path, mask):
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def top_fraction_mask(values, valid_mask, fraction, largest=True):
    if fraction <= 0:
        return np.zeros(valid_mask.shape, dtype=bool)
    flat_valid = np.flatnonzero(valid_mask.reshape(-1))
    if flat_valid.size == 0:
        return np.zeros(valid_mask.shape, dtype=bool)
    k = max(1, int(round(flat_valid.size * fraction)))
    flat_values = values.reshape(-1)
    valid_values = flat_values[flat_valid]
    if largest:
        selected_local = np.argpartition(valid_values, -k)[-k:]
    else:
        selected_local = np.argpartition(valid_values, k - 1)[:k]
    selected = flat_valid[selected_local]
    mask = np.zeros(valid_mask.size, dtype=bool)
    mask[selected] = True
    return mask.reshape(valid_mask.shape)


def boundary_band_from_prediction(prediction, valid_mask, class_ids, radius):
    selected = np.isin(prediction, np.asarray(class_ids, dtype=np.uint8)) & valid_mask
    if not np.any(selected):
        return np.zeros(prediction.shape, dtype=bool)

    boundary = np.zeros(prediction.shape, dtype=bool)
    for class_id in class_ids:
        class_mask = (prediction == class_id) & valid_mask
        if not np.any(class_mask):
            continue
        class_uint8 = class_mask.astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        grad = cv2.morphologyEx(class_uint8, cv2.MORPH_GRADIENT, kernel) > 0
        boundary |= grad

    if radius > 0:
        kernel_size = 2 * int(radius) + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        boundary = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1) > 0
    return boundary & valid_mask & selected


def calculate_group_counts(candidate, prediction, label):
    valid = label != IGNORE_INDEX
    error = valid & (prediction != label)
    clutter = valid & (label == 5)
    clutter_fn = clutter & (prediction != 5)
    clutter_to_impervious = clutter & (prediction == 0)
    clutter_to_lowveg = clutter & (prediction == 2)
    clutter_to_building = clutter & (prediction == 1)

    groups = {
        "valid": valid,
        "error": error,
        "clutter_fn": clutter_fn,
        "clutter_to_impervious": clutter_to_impervious,
        "clutter_to_lowveg": clutter_to_lowveg,
        "clutter_to_building": clutter_to_building,
    }
    counts = {}
    for name, mask in groups.items():
        total = int(mask.sum())
        covered = int((candidate & mask).sum())
        counts[f"{name}_total"] = total
        counts[f"{name}_covered"] = covered
        counts[f"{name}_coverage"] = covered / total if total else 0.0
    counts["candidate_pixels"] = int(candidate.sum())
    counts["candidate_ratio"] = counts["candidate_pixels"] / counts["valid_total"] if counts["valid_total"] else 0.0
    counts["candidate_error_precision"] = (
        counts["error_covered"] / counts["candidate_pixels"] if counts["candidate_pixels"] else 0.0
    )
    counts["candidate_clutter_fn_precision"] = (
        counts["clutter_fn_covered"] / counts["candidate_pixels"] if counts["candidate_pixels"] else 0.0
    )
    return counts


def create_candidate_overlay(candidate, prediction):
    overlay = colorize_label(prediction).astype(np.float32)
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    overlay[candidate] = 0.35 * overlay[candidate] + 0.65 * red[candidate]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def find_image_ids(prediction_dir):
    return sorted(p.name.replace("_prediction_id.png", "") for p in prediction_dir.glob("*_prediction_id.png"))


def main():
    parser = argparse.ArgumentParser(description="Generate SegFormer candidate regions for SAM3 local refinement")
    parser.add_argument("--eval-dir", default="results_segformer_potsdam_rgb/eval_heldout_uncertainty")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--entropy-fraction", type=float, default=0.20)
    parser.add_argument("--margin-fraction", type=float, default=0.20)
    parser.add_argument("--boundary-radius", type=int, default=8)
    parser.add_argument(
        "--boundary-classes",
        default="0,1,2,5",
        help="Comma-separated predicted class ids used for boundary band generation",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir) if args.output_dir else eval_dir / "candidate_regions"
    prediction_dir = eval_dir / "predictions"
    uncertainty_dir = eval_dir / "uncertainty"
    metrics_dir = output_dir / "metrics"
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    boundary_classes = [int(x) for x in args.boundary_classes.split(",") if x.strip()]
    image_ids = find_image_ids(prediction_dir)
    if not image_ids:
        raise RuntimeError(f"No prediction_id PNG files found in {prediction_dir}")

    per_image_rows = []
    dataset_counts = {}

    for image_name in tqdm(image_ids, desc="Generate candidate regions"):
        prediction = load_id_png(prediction_dir / f"{image_name}_prediction_id.png")
        label = load_id_png(prediction_dir / f"{image_name}_ground_truth_id.png")
        entropy = np.asarray(Image.open(uncertainty_dir / f"{image_name}_entropy.png"), dtype=np.float32) / 255.0
        margin = np.asarray(Image.open(uncertainty_dir / f"{image_name}_margin.png"), dtype=np.float32) / 255.0
        valid = label != IGNORE_INDEX

        high_entropy = top_fraction_mask(entropy, valid, args.entropy_fraction, largest=True)
        low_margin = top_fraction_mask(margin, valid, args.margin_fraction, largest=False)
        boundary_band = boundary_band_from_prediction(prediction, valid, boundary_classes, args.boundary_radius)
        candidate = (high_entropy | low_margin | boundary_band) & valid

        save_mask(mask_dir / f"{image_name}_candidate.png", candidate)
        save_mask(mask_dir / f"{image_name}_high_entropy.png", high_entropy)
        save_mask(mask_dir / f"{image_name}_low_margin.png", low_margin)
        save_mask(mask_dir / f"{image_name}_boundary_band.png", boundary_band)
        Image.fromarray(create_candidate_overlay(candidate, prediction)).save(
            overlay_dir / f"{image_name}_candidate_overlay.png"
        )

        counts = calculate_group_counts(candidate, prediction, label)
        counts.update(
            {
                "image_name": image_name,
                "high_entropy_pixels": int(high_entropy.sum()),
                "low_margin_pixels": int(low_margin.sum()),
                "boundary_band_pixels": int(boundary_band.sum()),
                "boundary_classes": boundary_classes,
            }
        )
        per_image_rows.append(counts)
        for key, value in counts.items():
            if key.endswith("_total") or key.endswith("_covered") or key in (
                "candidate_pixels",
                "high_entropy_pixels",
                "low_margin_pixels",
                "boundary_band_pixels",
            ):
                dataset_counts[key] = dataset_counts.get(key, 0) + int(value)

    for group in [
        "valid",
        "error",
        "clutter_fn",
        "clutter_to_impervious",
        "clutter_to_lowveg",
        "clutter_to_building",
    ]:
        total = dataset_counts.get(f"{group}_total", 0)
        covered = dataset_counts.get(f"{group}_covered", 0)
        dataset_counts[f"{group}_coverage"] = covered / total if total else 0.0
    dataset_counts["candidate_ratio"] = (
        dataset_counts["candidate_pixels"] / dataset_counts["valid_total"] if dataset_counts.get("valid_total") else 0.0
    )
    dataset_counts["candidate_error_precision"] = (
        dataset_counts["error_covered"] / dataset_counts["candidate_pixels"]
        if dataset_counts.get("candidate_pixels")
        else 0.0
    )
    dataset_counts["candidate_clutter_fn_precision"] = (
        dataset_counts["clutter_fn_covered"] / dataset_counts["candidate_pixels"]
        if dataset_counts.get("candidate_pixels")
        else 0.0
    )

    summary = {
        "eval_dir": str(eval_dir),
        "output_dir": str(output_dir),
        "num_images": len(image_ids),
        "image_names": image_ids,
        "entropy_fraction": args.entropy_fraction,
        "margin_fraction": args.margin_fraction,
        "boundary_radius": args.boundary_radius,
        "boundary_classes": boundary_classes,
        "class_names": [CLASS_INFO[i]["name"] for i in CLASS_IDS],
        "dataset_counts": dataset_counts,
    }
    with open(metrics_dir / "candidate_region_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fieldnames = list(per_image_rows[0].keys())
    with open(metrics_dir / "candidate_region_per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

    print(
        "Candidate regions complete: "
        f"candidate_ratio={dataset_counts['candidate_ratio']:.4f}, "
        f"error_coverage={dataset_counts['error_coverage']:.4f}, "
        f"clutter_fn_coverage={dataset_counts['clutter_fn_coverage']:.4f}"
    )


if __name__ == "__main__":
    main()
