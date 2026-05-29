#!/usr/bin/env python3

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_exp5c")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from exp1_sam3_potsdam_zeroshot_baseline import (
    SAM3_AVAILABLE,
    build_class_info,
    load_yaml_config,
    save_config_snapshot,
)
from exp2a_sam3_potsdam_no_background_prompt import (
    load_prompt_class_config,
    raise_sam3_unavailable_error,
)
from exp2b_sam3_potsdam_visual_prompt_ensemble import (
    Exp2BVisualPromptEnsembleEvaluator,
    load_prompt_variants,
)
from segformer_potsdam_common import (
    IGNORE_INDEX,
    colorize_label,
    crop_with_padding,
    generate_patch_starts,
    read_potsdam_image,
)


EXPERIMENT_NAME = "Exp5-C: SAM3 on SegFormer candidate patches"
DEFAULT_CONFIG_PATH = "exp5c_sam3_candidate_patch_validation.yaml"
POTSDAM_HELDOUT_IMAGE_IDS = [
    "top_potsdam_2_13",
    "top_potsdam_2_14",
    "top_potsdam_3_13",
    "top_potsdam_3_14",
    "top_potsdam_4_13",
    "top_potsdam_4_14",
    "top_potsdam_4_15",
    "top_potsdam_5_13",
    "top_potsdam_5_14",
    "top_potsdam_5_15",
    "top_potsdam_6_13",
    "top_potsdam_6_14",
    "top_potsdam_6_15",
    "top_potsdam_7_13",
]


def ensure_dirs(output_dir):
    output_dir = Path(output_dir)
    for subdir in ["metrics", "logs", "patch_visualizations", "mask_overlays"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    return output_dir


def load_id_png(path):
    return np.asarray(Image.open(path)).astype(np.uint8)


def image_id_to_name(image_id):
    return f"{image_id}_RGB"


def resolve_image_ids(config):
    evaluation = config.get("evaluation", {})
    explicit = evaluation.get("image_ids") or []
    if explicit:
        return list(explicit)
    split = str(evaluation.get("split", "heldout")).lower()
    if split in ("heldout", "held-out", "test"):
        return list(POTSDAM_HELDOUT_IMAGE_IDS)
    raise ValueError(f"Exp5-C 当前仅内置 heldout split，实际 split={split!r}")


def image_path_for_id(config, image_id):
    paths = config.get("paths", {})
    base_dir = Path(paths.get("base_dir", "/home/anjou/PythonENV/Test_11/Potsdam"))
    image_subdir = paths.get("image_subdir", "2_Ortho_RGB/2_Ortho_RGB")
    image_pattern = paths.get("image_pattern", "{image_id}_RGB.tif")
    return base_dir / image_subdir / image_pattern.format(image_id=image_id)


def load_eval_arrays(config, image_id):
    paths = config.get("paths", {})
    seg_eval_dir = Path(paths.get("segformer_eval_dir", "results_segformer_potsdam_rgb/eval_heldout_uncertainty"))
    candidate_dir = Path(paths.get("candidate_dir", "results_exp5b_segformer_candidate_regions_heldout"))
    image_name = image_id_to_name(image_id)
    prediction = load_id_png(seg_eval_dir / "predictions" / f"{image_name}_prediction_id.png")
    label = load_id_png(seg_eval_dir / "predictions" / f"{image_name}_ground_truth_id.png")
    candidate = np.asarray(Image.open(candidate_dir / "masks" / f"{image_name}_candidate.png")) > 0
    image = read_potsdam_image(image_path_for_id(config, image_id))
    return image_name, image, prediction, label, candidate


def generate_candidate_patches(candidate, prediction, label, patch_size, stride, min_candidate_ratio):
    h, w = candidate.shape
    patches = []
    for y in generate_patch_starts(h, patch_size, stride):
        for x in generate_patch_starts(w, patch_size, stride):
            patch_h = min(patch_size, h - y)
            patch_w = min(patch_size, w - x)
            candidate_patch = candidate[y : y + patch_h, x : x + patch_w]
            valid_patch = label[y : y + patch_h, x : x + patch_w] != IGNORE_INDEX
            valid_pixels = int(valid_patch.sum())
            if valid_pixels == 0:
                continue
            candidate_pixels = int((candidate_patch & valid_patch).sum())
            candidate_ratio = candidate_pixels / valid_pixels
            if candidate_ratio < min_candidate_ratio:
                continue
            error_patch = valid_patch & (prediction[y : y + patch_h, x : x + patch_w] != label[y : y + patch_h, x : x + patch_w])
            clutter_fn_patch = valid_patch & (label[y : y + patch_h, x : x + patch_w] == 5) & (
                prediction[y : y + patch_h, x : x + patch_w] != 5
            )
            patches.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "patch_h": int(patch_h),
                    "patch_w": int(patch_w),
                    "valid_pixels": valid_pixels,
                    "candidate_pixels": candidate_pixels,
                    "candidate_ratio": float(candidate_ratio),
                    "error_pixels": int(error_patch.sum()),
                    "error_ratio": float(error_patch.sum() / valid_pixels),
                    "clutter_fn_pixels": int(clutter_fn_patch.sum()),
                    "clutter_fn_ratio": float(clutter_fn_patch.sum() / valid_pixels),
                }
            )
    return patches


def select_patches(patches, sort_by, max_patches_per_image):
    reverse = True
    selected = sorted(patches, key=lambda item: item.get(sort_by, item["candidate_ratio"]), reverse=reverse)
    if max_patches_per_image is not None and max_patches_per_image >= 0:
        selected = selected[:max_patches_per_image]
    return selected


def resize_mask(mask, shape):
    h, w = shape
    mask = np.squeeze(mask)
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return mask > 0.5


def safe_div(num, den):
    return float(num / den) if den else 0.0


def evaluate_sam_masks(predictions, patch_context, restrict_to_candidate=True):
    masks = predictions.get("masks", [])
    scores = predictions.get("scores", [])
    classes = predictions.get("classes", [])
    label = patch_context["label"]
    seg_prediction = patch_context["seg_prediction"]
    candidate = patch_context["candidate"]
    valid = label != IGNORE_INDEX
    error = valid & (seg_prediction != label)
    clutter_fn = valid & (label == 5) & (seg_prediction != 5)
    h, w = label.shape

    any_union = np.zeros((h, w), dtype=bool)
    class_unions = {class_id: np.zeros((h, w), dtype=bool) for class_id in range(5)}
    mask_rows = []

    for idx, raw_mask in enumerate(masks):
        class_id = int(classes[idx])
        score = float(scores[idx])
        mask = resize_mask(raw_mask, (h, w)) & valid
        raw_area = int(mask.sum())
        if restrict_to_candidate:
            eval_mask = mask & candidate
        else:
            eval_mask = mask
        eval_area = int(eval_mask.sum())
        gt_class = valid & (label == class_id)
        row = {
            "mask_index": idx,
            "class_id": class_id,
            "score": score,
            "raw_area": raw_area,
            "eval_area": eval_area,
            "candidate_overlap": int((eval_mask & candidate).sum()),
            "error_overlap": int((eval_mask & error).sum()),
            "clutter_fn_overlap": int((eval_mask & clutter_fn).sum()),
            "gt_class_overlap": int((eval_mask & gt_class).sum()),
            "eval_error_precision": safe_div(int((eval_mask & error).sum()), eval_area),
            "eval_clutter_fn_precision": safe_div(int((eval_mask & clutter_fn).sum()), eval_area),
            "gt_class_precision": safe_div(int((eval_mask & gt_class).sum()), eval_area),
            "gt_class_recall_in_candidate": safe_div(
                int((eval_mask & gt_class & candidate).sum()),
                int((gt_class & candidate).sum()),
            ),
        }
        mask_rows.append(row)
        if eval_area > 0:
            any_union |= eval_mask
            if class_id in class_unions:
                class_unions[class_id] |= eval_mask

    patch_summary = {
        "num_sam_masks": len(masks),
        "any_sam_pixels": int(any_union.sum()),
        "sam_error_overlap": int((any_union & error).sum()),
        "sam_clutter_fn_overlap": int((any_union & clutter_fn).sum()),
        "sam_candidate_overlap": int((any_union & candidate).sum()),
        "candidate_pixels": int((candidate & valid).sum()),
        "error_pixels": int(error.sum()),
        "clutter_fn_pixels": int(clutter_fn.sum()),
        "sam_error_coverage": safe_div(int((any_union & error).sum()), int(error.sum())),
        "sam_clutter_fn_coverage": safe_div(int((any_union & clutter_fn).sum()), int(clutter_fn.sum())),
        "sam_error_precision": safe_div(int((any_union & error).sum()), int(any_union.sum())),
        "sam_clutter_fn_precision": safe_div(int((any_union & clutter_fn).sum()), int(any_union.sum())),
    }
    for class_id, union in class_unions.items():
        gt_class = valid & (label == class_id)
        patch_summary[f"class_{class_id}_sam_pixels"] = int(union.sum())
        patch_summary[f"class_{class_id}_gt_precision"] = safe_div(int((union & gt_class).sum()), int(union.sum()))
        patch_summary[f"class_{class_id}_gt_recall_in_candidate"] = safe_div(
            int((union & gt_class & candidate).sum()),
            int((gt_class & candidate).sum()),
        )
    return patch_summary, mask_rows


def save_patch_visualization(output_dir, patch_key, image_patch, label_patch, seg_patch, candidate_patch):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image_patch[:, :, :3])
    axes[0].set_title("RGB patch")
    axes[1].imshow(colorize_label(label_patch))
    axes[1].set_title("Ground truth")
    axes[2].imshow(colorize_label(seg_patch))
    axes[2].set_title("SegFormer")
    overlay = colorize_label(seg_patch).astype(np.float32)
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    overlay[candidate_patch] = 0.35 * overlay[candidate_patch] + 0.65 * red[candidate_patch]
    axes[3].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    axes[3].set_title("Candidate region")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "patch_visualizations" / f"{patch_key}.png", dpi=130)
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_evaluator(config, output_dir):
    paths = config.get("paths", {})
    image_processing = config.get("image_processing", {})
    device = config.get("device", {})
    output_config = config.get("output", {})
    class_info = build_class_info(config)
    prompt_class_ids, excluded_prompt_class_ids = load_prompt_class_config(config, class_info)
    prompt_variants = load_prompt_variants(config, class_info, prompt_class_ids)
    evaluator = Exp2BVisualPromptEnsembleEvaluator(
        base_dir=paths.get("base_dir", "/home/anjou/PythonENV/Test_11/Potsdam"),
        output_dir=output_dir,
        checkpoint_path=paths.get("checkpoint_path"),
        patch_size=image_processing.get("patch_size", 1008),
        stride=image_processing.get("stride", 672),
        score_threshold=image_processing.get("score_threshold", 0.5),
        class_info=class_info,
        gpu_dtype=device.get("gpu_dtype", "float32"),
        device_type=device.get("type", "auto"),
        allow_hf_fallback=config.get("model", {}).get("allow_hf_fallback", False),
        output_config=output_config,
        metrics_config={},
        prompt_variants=prompt_variants,
        prompt_class_ids=prompt_class_ids,
        excluded_prompt_class_ids=excluded_prompt_class_ids,
    )
    text_prompts = [(class_id, class_info[class_id]["prompt"]) for class_id in prompt_class_ids]
    return evaluator, text_prompts, prompt_variants


def main():
    parser = argparse.ArgumentParser(description="Exp5-C: validate SAM3 masks on SegFormer candidate patches")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only generate candidate patch proposal statistics")
    parser.add_argument("--max-total-patches", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    paths = config.get("paths", {})
    evaluation = config.get("evaluation", {})
    image_processing = config.get("image_processing", {})
    output_config = config.get("output", {})
    output_dir = ensure_dirs(args.output_dir or paths.get("output_dir", "results_exp5c_sam3_candidate_patch_validation_heldout"))
    save_config_snapshot(args.config, output_dir)

    patch_size = int(image_processing.get("patch_size", 1008))
    stride = int(image_processing.get("stride", 672))
    min_candidate_ratio = float(evaluation.get("min_candidate_ratio", 0.08))
    max_patches_per_image = int(evaluation.get("max_patches_per_image", 6))
    max_total_patches = args.max_total_patches
    if max_total_patches is None:
        max_total_patches = int(evaluation.get("max_total_patches", 60))
    max_images = int(evaluation.get("max_images", -1))
    sort_by = str(evaluation.get("sort_patches_by", "candidate_ratio"))
    dry_run = bool(args.dry_run or evaluation.get("dry_run", False))
    restrict_to_candidate = bool(image_processing.get("restrict_masks_to_candidate", True))

    image_ids = resolve_image_ids(config)
    if max_images >= 0:
        image_ids = image_ids[:max_images]

    proposal_rows = []
    selected_patch_records = []
    for image_id in tqdm(image_ids, desc="Build patch proposals"):
        image_name, image, seg_prediction, label, candidate = load_eval_arrays(config, image_id)
        patches = generate_candidate_patches(candidate, seg_prediction, label, patch_size, stride, min_candidate_ratio)
        selected = select_patches(patches, sort_by, max_patches_per_image)
        for rank, patch_info in enumerate(selected):
            row = dict(patch_info)
            row.update({"image_id": image_id, "image_name": image_name, "rank": rank})
            proposal_rows.append(row)
            selected_patch_records.append((image_id, image_name, row))

    if max_total_patches is not None and max_total_patches >= 0:
        selected_patch_records = sorted(
            selected_patch_records,
            key=lambda item: item[2].get(sort_by, item[2]["candidate_ratio"]),
            reverse=True,
        )
        selected_patch_records = selected_patch_records[:max_total_patches]
        proposal_rows = [record[2] for record in selected_patch_records]

    write_csv(output_dir / "metrics" / "candidate_patch_proposals.csv", proposal_rows)

    run_context = {
        "experiment_name": EXPERIMENT_NAME,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": args.config,
        "output_dir": str(output_dir),
        "image_ids": image_ids,
        "num_selected_patches": len(selected_patch_records),
        "patch_size": patch_size,
        "stride": stride,
        "min_candidate_ratio": min_candidate_ratio,
        "max_patches_per_image": max_patches_per_image,
        "max_total_patches": max_total_patches,
        "sort_by": sort_by,
        "dry_run": dry_run,
        "restrict_masks_to_candidate": restrict_to_candidate,
    }
    with open(output_dir / "metrics" / "run_context.json", "w", encoding="utf-8") as f:
        json.dump(run_context, f, indent=2, ensure_ascii=False)

    if dry_run:
        print(f"Dry run complete: selected_patches={len(selected_patch_records)}")
        return

    if not SAM3_AVAILABLE:
        raise_sam3_unavailable_error(EXPERIMENT_NAME)

    evaluator, text_prompts, prompt_variants = build_evaluator(config, output_dir)
    patch_rows = []
    mask_rows = []
    dataset_counts = {
        "patches": 0,
        "sam_masks": 0,
        "candidate_pixels": 0,
        "error_pixels": 0,
        "clutter_fn_pixels": 0,
        "any_sam_pixels": 0,
        "sam_error_overlap": 0,
        "sam_clutter_fn_overlap": 0,
        "sam_candidate_overlap": 0,
    }

    cache = {}
    for global_index, (image_id, image_name, patch_info) in enumerate(
        tqdm(selected_patch_records, desc="Run SAM3 on candidate patches")
    ):
        if image_id not in cache:
            cache[image_id] = load_eval_arrays(config, image_id)
        _, image, seg_prediction, label, candidate = cache[image_id]
        x, y = int(patch_info["x"]), int(patch_info["y"])
        image_patch = crop_with_padding(image, x, y, patch_size, 0)
        seg_patch = crop_with_padding(seg_prediction, x, y, patch_size, 5)
        label_patch = crop_with_padding(label, x, y, patch_size, IGNORE_INDEX)
        candidate_patch = crop_with_padding(candidate.astype(np.uint8), x, y, patch_size, 0).astype(bool)
        patch_key = f"{image_name}_patch_{global_index:04d}_x{x}_y{y}"

        predictions = evaluator.predict_patch(
            image_patch,
            text_prompts,
            patch_index=global_index,
            patch_position=(x, y),
        )
        patch_context = {
            "label": label_patch,
            "seg_prediction": seg_patch,
            "candidate": candidate_patch,
        }
        patch_summary, patch_mask_rows = evaluate_sam_masks(
            predictions,
            patch_context,
            restrict_to_candidate=restrict_to_candidate,
        )
        row = dict(patch_info)
        row.update(patch_summary)
        row.update({"image_id": image_id, "image_name": image_name, "patch_key": patch_key, "global_index": global_index})
        patch_rows.append(row)
        for mask_row in patch_mask_rows:
            mask_out = dict(mask_row)
            mask_out.update({"image_id": image_id, "image_name": image_name, "patch_key": patch_key, "global_index": global_index})
            mask_rows.append(mask_out)

        for key in dataset_counts:
            if key == "patches":
                continue
            dataset_counts[key] += int(patch_summary.get(key, 0))
        dataset_counts["patches"] += 1
        dataset_counts["sam_masks"] += int(patch_summary.get("num_sam_masks", 0))

        if output_config.get("save_patch_visualizations", True):
            save_patch_visualization(output_dir, patch_key, image_patch, label_patch, seg_patch, candidate_patch)

    dataset_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "prompt_variants": prompt_variants,
        "counts": dataset_counts,
        "metrics": {
            "sam_error_coverage": safe_div(dataset_counts["sam_error_overlap"], dataset_counts["error_pixels"]),
            "sam_clutter_fn_coverage": safe_div(
                dataset_counts["sam_clutter_fn_overlap"],
                dataset_counts["clutter_fn_pixels"],
            ),
            "sam_error_precision": safe_div(dataset_counts["sam_error_overlap"], dataset_counts["any_sam_pixels"]),
            "sam_clutter_fn_precision": safe_div(
                dataset_counts["sam_clutter_fn_overlap"],
                dataset_counts["any_sam_pixels"],
            ),
            "sam_candidate_overlap_ratio": safe_div(
                dataset_counts["sam_candidate_overlap"],
                dataset_counts["candidate_pixels"],
            ),
        },
        "run_context": run_context,
    }
    write_csv(output_dir / "metrics" / "sam_patch_metrics.csv", patch_rows)
    write_csv(output_dir / "metrics" / "sam_mask_metrics.csv", mask_rows)
    with open(output_dir / "metrics" / "sam_candidate_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2, ensure_ascii=False)

    print(
        "Exp5-C complete: "
        f"patches={dataset_counts['patches']}, "
        f"sam_masks={dataset_counts['sam_masks']}, "
        f"error_coverage={dataset_summary['metrics']['sam_error_coverage']:.4f}, "
        f"clutter_fn_coverage={dataset_summary['metrics']['sam_clutter_fn_coverage']:.4f}"
    )


if __name__ == "__main__":
    main()
