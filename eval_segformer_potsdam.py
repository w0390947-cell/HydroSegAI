#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from segformer_potsdam_common import (
    NUM_CLASSES,
    build_image_ids,
    calculate_metrics,
    crop_with_padding,
    generate_patch_starts,
    image_and_label_paths,
    load_yaml_config,
    normalize_image,
    prepare_output_dir,
    read_potsdam_image,
    read_potsdam_label,
    save_overall_metrics,
    save_prediction_outputs,
)


PARTICIPANT_LABEL_SUBDIR = "5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary"
ALL_LABEL_SUBDIR = "5_Labels_all_noBoundary"


def resolve_device(config):
    device_config = str(config.get("device", {}).get("type", "auto")).lower()
    if device_config == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device.type=cuda but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if device_config == "cpu":
        return torch.device("cpu")
    if device_config != "auto":
        raise ValueError(f"Unknown device.type={device_config!r}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_for_eval(config, checkpoint_override=None):
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError as exc:
        raise RuntimeError("transformers is required for SegFormer evaluation") from exc

    model_config = config.get("model", {})
    checkpoint = (
        checkpoint_override
        or model_config.get("checkpoint_path")
        or str(Path(config.get("paths", {}).get("output_dir", "results_segformer_potsdam_rgb")) / "checkpoints" / "best")
    )
    return SegformerForSemanticSegmentation.from_pretrained(checkpoint)


def adapt_label_dir_for_split(config, split_name):
    split_name = str(split_name or "").lower()
    if split_name not in ("heldout", "held-out", "test"):
        return
    paths = config.setdefault("paths", {})
    if paths.get("label_subdir", PARTICIPANT_LABEL_SUBDIR) == PARTICIPANT_LABEL_SUBDIR:
        paths["label_subdir"] = ALL_LABEL_SUBDIR


@torch.no_grad()
def predict_full_image(model, image, config, device, return_logits=False):
    image_processing = config.get("image_processing", {})
    mean = image_processing.get("mean", [0.485, 0.456, 0.406])
    std = image_processing.get("std", [0.229, 0.224, 0.225])
    patch_size = int(image_processing.get("eval_patch_size", image_processing.get("patch_size", 512)))
    stride = int(image_processing.get("eval_stride", image_processing.get("val_stride", patch_size)))
    batch_size = int(config.get("evaluation", {}).get("batch_size", 2))
    amp_enabled = bool(config.get("evaluation", {}).get("amp", True)) and device.type == "cuda"

    h, w = image.shape[:2]
    y_starts = generate_patch_starts(h, patch_size, stride)
    x_starts = generate_patch_starts(w, patch_size, stride)
    coords = [(x, y) for y in y_starts for x in x_starts]

    score_sum = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.uint16)

    model.eval()
    for start in tqdm(range(0, len(coords), batch_size), desc="Infer patches", leave=False):
        batch_coords = coords[start : start + batch_size]
        batch_arrays = []
        for x, y in batch_coords:
            patch = crop_with_padding(image, x, y, patch_size, 0)
            batch_arrays.append(normalize_image(patch, mean, std))
        pixel_values = torch.from_numpy(np.stack(batch_arrays, axis=0)).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            outputs = model(pixel_values=pixel_values)
            logits = F.interpolate(
                outputs.logits,
                size=(patch_size, patch_size),
                mode="bilinear",
                align_corners=False,
            )
        logits_np = logits.detach().float().cpu().numpy()

        for patch_logits, (x, y) in zip(logits_np, batch_coords):
            patch_h = min(patch_size, h - y)
            patch_w = min(patch_size, w - x)
            score_sum[:, y : y + patch_h, x : x + patch_w] += patch_logits[:, :patch_h, :patch_w]
            count[y : y + patch_h, x : x + patch_w] += 1

    if np.any(count == 0):
        raise RuntimeError("Sliding-window inference left uncovered pixels")
    score_sum /= np.maximum(count[None, :, :], 1)
    prediction = np.argmax(score_sum, axis=0).astype(np.uint8)
    if return_logits:
        return prediction, score_sum
    return prediction


def uncertainty_from_logits(logits):
    logits = logits.astype(np.float32, copy=False)
    logits = logits - np.max(logits, axis=0, keepdims=True)
    probs = np.exp(logits)
    probs /= np.maximum(np.sum(probs, axis=0, keepdims=True), 1e-12)

    sorted_probs = np.sort(probs, axis=0)
    top1 = sorted_probs[-1]
    top2 = sorted_probs[-2]
    confidence = top1
    margin = top1 - top2
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=0)
    entropy /= np.log(NUM_CLASSES)
    return {
        "confidence": confidence.astype(np.float32),
        "margin": margin.astype(np.float32),
        "entropy": entropy.astype(np.float32),
    }


def save_uncertainty_outputs(output_dir, image_name, logits, prediction, label, image_format="png"):
    output_dir = Path(output_dir)
    uncertainty_dir = output_dir / "uncertainty"
    uncertainty_dir.mkdir(parents=True, exist_ok=True)

    maps = uncertainty_from_logits(logits)
    for name, values in maps.items():
        values_uint8 = np.clip(values * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(values_uint8).save(uncertainty_dir / f"{image_name}_{name}.{image_format}")

    valid = label != 255
    error_map = np.full(label.shape, 127, dtype=np.uint8)
    error_map[valid & (prediction == label)] = 0
    error_map[valid & (prediction != label)] = 255
    Image.fromarray(error_map).save(uncertainty_dir / f"{image_name}_error.{image_format}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SegFormer on Potsdam full tiles")
    parser.add_argument("--config", default="segformer_potsdam_rgb.yaml")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint dir; defaults to paths.output_dir/checkpoints/best")
    parser.add_argument("--split", default=None, help="Override evaluation split: val/dev/heldout/all")
    parser.add_argument("--output-dir", default=None, help="Optional evaluation output dir")
    parser.add_argument("--save-uncertainty", action="store_true", help="Save confidence, entropy, margin, and error maps")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    train_output_dir = Path(config.get("paths", {}).get("output_dir", "results_segformer_potsdam_rgb"))
    if args.split:
        config.setdefault("splits", {})["eval"] = args.split
        config.setdefault("splits", {})["eval_images"] = []
    adapt_label_dir_for_split(config, config.get("splits", {}).get("eval", "val"))
    if args.output_dir:
        config.setdefault("paths", {})["output_dir"] = args.output_dir
    else:
        eval_name = str(config.get("splits", {}).get("eval", "val"))
        config.setdefault("paths", {})["output_dir"] = str(train_output_dir / f"eval_{eval_name}")
    if not args.checkpoint and not config.get("model", {}).get("checkpoint_path"):
        config.setdefault("model", {})["checkpoint_path"] = str(train_output_dir / "checkpoints" / "best")

    output_dir = prepare_output_dir(config, args.config)
    device = resolve_device(config)
    model = load_model_for_eval(config, checkpoint_override=args.checkpoint).to(device)

    eval_ids = build_image_ids(config.get("splits", {}), "eval")
    all_results = []
    output_config = config.get("output", {})
    image_format = output_config.get("image_format", "png")
    dpi = int(output_config.get("visualization_dpi", 150))

    for image_id in tqdm(eval_ids, desc="Evaluate images"):
        image_path, label_path = image_and_label_paths(config, image_id)
        image = read_potsdam_image(image_path)
        label = read_potsdam_label(label_path)
        if image.shape[:2] != label.shape:
            raise ValueError(f"Image/label shape mismatch: {image_path}, {label_path}")

        prediction_result = predict_full_image(model, image, config, device, return_logits=args.save_uncertainty)
        if args.save_uncertainty:
            prediction, logits = prediction_result
        else:
            prediction = prediction_result
            logits = None
        metrics = calculate_metrics(prediction, label)
        image_name = Path(image_path).stem
        if output_config.get("save_predictions", True) or output_config.get("save_visualizations", True):
            save_prediction_outputs(
                output_dir,
                image_name,
                image,
                label,
                prediction,
                metrics,
                image_format=image_format,
                dpi=dpi,
            )
        if args.save_uncertainty:
            save_uncertainty_outputs(
                output_dir,
                image_name,
                logits,
                prediction,
                label,
                image_format=image_format,
            )
        all_results.append({"image_name": image_name, "metrics": metrics})

    run_context = {
        "config_path": args.config,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "eval_images": eval_ids,
        "num_eval_images": len(eval_ids),
        "save_uncertainty": bool(args.save_uncertainty),
    }
    with open(output_dir / "metrics" / "run_context.json", "w", encoding="utf-8") as f:
        json.dump(run_context, f, indent=2, ensure_ascii=False)
    overall = save_overall_metrics(output_dir, all_results, config, run_context=run_context)
    print(
        "Evaluation complete: "
        f"mIoU={overall['dataset_mean_iou']:.4f}, "
        f"MeanF1={overall['dataset_mean_f1']:.4f}, "
        f"OA={overall['dataset_overall_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
