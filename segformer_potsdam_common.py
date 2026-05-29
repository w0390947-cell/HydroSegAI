#!/usr/bin/env python3

import csv
import json
import os
from datetime import datetime
from pathlib import Path

_MPL_CACHE_DIR = "/tmp/matplotlib_segformer"
os.makedirs(_MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE_DIR)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from osgeo import gdal

    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False


IGNORE_INDEX = 255
CLASS_INFO = {
    0: {"name": "impervious surface", "color": [255, 255, 255]},
    1: {"name": "building", "color": [0, 0, 255]},
    2: {"name": "low vegetation", "color": [0, 255, 255]},
    3: {"name": "tree", "color": [0, 255, 0]},
    4: {"name": "car", "color": [255, 255, 0]},
    5: {"name": "clutter/background", "color": [255, 0, 0]},
}
COLOR_TO_CLASS = {
    (255, 255, 255): 0,
    (0, 0, 255): 1,
    (0, 255, 255): 2,
    (0, 255, 0): 3,
    (255, 255, 0): 4,
    (255, 0, 0): 5,
    (0, 0, 0): IGNORE_INDEX,
}
CLASS_IDS = sorted(CLASS_INFO.keys())
NUM_CLASSES = len(CLASS_IDS)


POTSDAM_DEV_IMAGE_IDS = [
    "top_potsdam_2_10",
    "top_potsdam_2_11",
    "top_potsdam_2_12",
    "top_potsdam_3_10",
    "top_potsdam_3_11",
    "top_potsdam_3_12",
    "top_potsdam_4_10",
    "top_potsdam_4_11",
    "top_potsdam_4_12",
    "top_potsdam_5_10",
    "top_potsdam_5_11",
    "top_potsdam_5_12",
    "top_potsdam_6_7",
    "top_potsdam_6_8",
    "top_potsdam_6_9",
    "top_potsdam_6_10",
    "top_potsdam_6_11",
    "top_potsdam_6_12",
    "top_potsdam_7_7",
    "top_potsdam_7_8",
    "top_potsdam_7_9",
    "top_potsdam_7_10",
    "top_potsdam_7_11",
    "top_potsdam_7_12",
]

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


DEFAULT_TRAIN_IMAGE_IDS = [
    "top_potsdam_2_10",
    "top_potsdam_2_11",
    "top_potsdam_2_12",
    "top_potsdam_3_10",
    "top_potsdam_3_11",
    "top_potsdam_3_12",
    "top_potsdam_4_10",
    "top_potsdam_4_11",
    "top_potsdam_4_12",
    "top_potsdam_5_10",
    "top_potsdam_5_11",
    "top_potsdam_5_12",
    "top_potsdam_6_7",
    "top_potsdam_6_8",
    "top_potsdam_6_9",
    "top_potsdam_6_10",
    "top_potsdam_6_11",
    "top_potsdam_6_12",
]

DEFAULT_VAL_IMAGE_IDS = [
    "top_potsdam_7_7",
    "top_potsdam_7_8",
    "top_potsdam_7_9",
    "top_potsdam_7_10",
    "top_potsdam_7_11",
    "top_potsdam_7_12",
]


def load_yaml_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read config files") from exc
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config_snapshot(config_path, output_dir):
    source_path = Path(config_path)
    if not source_path.exists():
        return None
    snapshot_path = Path(output_dir) / "metrics" / "config_snapshot.yaml"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    return str(snapshot_path)


def read_geotiff(path):
    path = Path(path)
    if GDAL_AVAILABLE:
        dataset = gdal.Open(str(path))
        if dataset is None:
            raise FileNotFoundError(f"Cannot open file: {path}")
        array = dataset.ReadAsArray()
        dataset = None
        array = np.asarray(array)
        if array.ndim == 3 and array.shape[0] in (3, 4):
            array = np.transpose(array, (1, 2, 0))
        return array

    image = Image.open(path)
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[:, :, :3]
    return array


def read_potsdam_image(path, channels="rgb"):
    image = read_geotiff(path)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected HWC image with at least 3 channels: {path}, shape={image.shape}")
    image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def read_potsdam_label(path):
    label_rgb = read_geotiff(path)
    if label_rgb.ndim == 3 and label_rgb.shape[-1] == 4:
        label_rgb = label_rgb[:, :, :3]
    if label_rgb.ndim == 2:
        return label_rgb.astype(np.uint8, copy=False)
    if label_rgb.ndim != 3 or label_rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB label or class-id label: {path}, shape={label_rgb.shape}")

    h, w = label_rgb.shape[:2]
    label = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
    for color, class_id in COLOR_TO_CLASS.items():
        mask = (
            (label_rgb[:, :, 0] == color[0])
            & (label_rgb[:, :, 1] == color[1])
            & (label_rgb[:, :, 2] == color[2])
        )
        label[mask] = class_id
    return label


def colorize_label(label):
    colored = np.zeros((*label.shape, 3), dtype=np.uint8)
    colored[label == IGNORE_INDEX] = [0, 0, 0]
    for class_id, info in CLASS_INFO.items():
        colored[label == class_id] = info["color"]
    return colored


def normalize_image(image, mean, std):
    image = image.astype(np.float32) / 255.0
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    return np.transpose(image, (2, 0, 1)).astype(np.float32)


def generate_patch_starts(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    edge_start = length - patch_size
    if starts[-1] != edge_start:
        starts.append(edge_start)
    return starts


def crop_with_padding(array, x, y, patch_size, fill_value):
    h, w = array.shape[:2]
    patch_h = min(patch_size, h - y)
    patch_w = min(patch_size, w - x)
    patch = array[y : y + patch_h, x : x + patch_w]
    if patch_h == patch_size and patch_w == patch_size:
        return patch

    if array.ndim == 3:
        padded = np.full((patch_size, patch_size, array.shape[2]), fill_value, dtype=array.dtype)
        padded[:patch_h, :patch_w, :] = patch
    else:
        padded = np.full((patch_size, patch_size), fill_value, dtype=array.dtype)
        padded[:patch_h, :patch_w] = patch
    return padded


def build_image_ids(split_config, stage):
    explicit_key = f"{stage}_images"
    explicit = split_config.get(explicit_key)
    if explicit:
        return list(explicit)

    split = str(split_config.get(stage, "") or "").lower()
    if split in ("train", "default_train"):
        return list(DEFAULT_TRAIN_IMAGE_IDS)
    if split in ("val", "valid", "validation", "default_val"):
        return list(DEFAULT_VAL_IMAGE_IDS)
    if split in ("dev", "development"):
        return list(POTSDAM_DEV_IMAGE_IDS)
    if split in ("heldout", "held-out", "test"):
        return list(POTSDAM_HELDOUT_IMAGE_IDS)
    if split == "all":
        return list(POTSDAM_DEV_IMAGE_IDS) + list(POTSDAM_HELDOUT_IMAGE_IDS)

    if stage == "train":
        return list(DEFAULT_TRAIN_IMAGE_IDS)
    if stage in ("val", "eval"):
        return list(DEFAULT_VAL_IMAGE_IDS)
    raise ValueError(f"Cannot resolve image ids for stage={stage}, split={split!r}")


def image_and_label_paths(config, image_id):
    paths = config.get("paths", {})
    base_dir = Path(paths.get("base_dir", "/home/anjou/PythonENV/Test_11/Potsdam"))
    image_subdir = paths.get("image_subdir", "2_Ortho_RGB/2_Ortho_RGB")
    label_subdir = paths.get(
        "label_subdir",
        "5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary",
    )
    image_pattern = paths.get("image_pattern", "{image_id}_RGB.tif")
    label_pattern = paths.get("label_pattern", "{image_id}_label_noBoundary.tif")
    return (
        base_dir / image_subdir / image_pattern.format(image_id=image_id),
        base_dir / label_subdir / label_pattern.format(image_id=image_id),
    )


def build_patch_index(config, image_ids, patch_size, stride, min_valid_ratio=0.05):
    samples = []
    for image_id in image_ids:
        image_path, label_path = image_and_label_paths(config, image_id)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Label not found: {label_path}")
        label = read_potsdam_label(label_path)
        image = read_potsdam_image(image_path)
        if image.shape[:2] != label.shape:
            raise ValueError(f"Image/label shape mismatch: {image_path}, {label_path}")
        y_starts = generate_patch_starts(label.shape[0], patch_size, stride)
        x_starts = generate_patch_starts(label.shape[1], patch_size, stride)
        for y in y_starts:
            for x in x_starts:
                patch = crop_with_padding(label, x, y, patch_size, IGNORE_INDEX)
                valid_ratio = float(np.mean(patch != IGNORE_INDEX))
                if valid_ratio >= min_valid_ratio:
                    samples.append(
                        {
                            "image_id": image_id,
                            "image_path": str(image_path),
                            "label_path": str(label_path),
                            "x": int(x),
                            "y": int(y),
                            "valid_ratio": valid_ratio,
                        }
                    )
    return samples


class PotsdamPatchDataset(Dataset):
    def __init__(
        self,
        config,
        image_ids,
        patch_size,
        stride,
        mean,
        std,
        min_valid_ratio=0.05,
        augment=False,
        cache_tiles=False,
    ):
        self.config = config
        self.image_ids = list(image_ids)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.mean = mean
        self.std = std
        self.augment = bool(augment)
        self.cache_tiles = bool(cache_tiles)
        self.samples = build_patch_index(
            config,
            self.image_ids,
            self.patch_size,
            self.stride,
            min_valid_ratio=min_valid_ratio,
        )
        self._image_cache = {}
        self._label_cache = {}

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        if self.cache_tiles and path in self._image_cache:
            return self._image_cache[path]
        image = read_potsdam_image(path)
        if self.cache_tiles:
            self._image_cache[path] = image
        return image

    def _load_label(self, path):
        if self.cache_tiles and path in self._label_cache:
            return self._label_cache[path]
        label = read_potsdam_label(path)
        if self.cache_tiles:
            self._label_cache[path] = label
        return label

    def __getitem__(self, index):
        sample = self.samples[index]
        image = self._load_image(sample["image_path"])
        label = self._load_label(sample["label_path"])
        x, y = sample["x"], sample["y"]
        image_patch = crop_with_padding(image, x, y, self.patch_size, 0)
        label_patch = crop_with_padding(label, x, y, self.patch_size, IGNORE_INDEX)

        if self.augment:
            if np.random.rand() < 0.5:
                image_patch = np.ascontiguousarray(image_patch[:, ::-1, :])
                label_patch = np.ascontiguousarray(label_patch[:, ::-1])
            if np.random.rand() < 0.5:
                image_patch = np.ascontiguousarray(image_patch[::-1, :, :])
                label_patch = np.ascontiguousarray(label_patch[::-1, :])

        pixel_values = normalize_image(image_patch, self.mean, self.std)
        return {
            "pixel_values": torch.from_numpy(pixel_values),
            "labels": torch.from_numpy(label_patch.astype(np.int64, copy=False)),
            "image_id": sample["image_id"],
            "x": x,
            "y": y,
        }


def compute_metrics_from_confusion_matrix(cm):
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    true_positive = np.diag(cm)
    pred_count = cm.sum(axis=0)
    gt_count = cm.sum(axis=1)

    overall_accuracy = true_positive.sum() / total if total > 0 else 0.0
    union = gt_count + pred_count - true_positive
    iou_per_class = np.divide(
        true_positive,
        union,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=union > 0,
    )
    precision_per_class = np.divide(
        true_positive,
        pred_count,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=pred_count > 0,
    )
    recall_per_class = np.divide(
        true_positive,
        gt_count,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=gt_count > 0,
    )
    f1_denominator = precision_per_class + recall_per_class
    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        f1_denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=f1_denominator > 0,
    )
    frequency = np.divide(
        gt_count,
        total,
        out=np.zeros_like(gt_count, dtype=np.float64),
        where=total > 0,
    )
    frequency_weighted_iou = np.sum(frequency * iou_per_class)

    return {
        "overall_accuracy": float(overall_accuracy),
        "mean_iou": float(np.mean(iou_per_class)),
        "mean_f1": float(np.mean(f1_per_class)),
        "mean_precision": float(np.mean(precision_per_class)),
        "mean_recall": float(np.mean(recall_per_class)),
        "frequency_weighted_iou": float(frequency_weighted_iou),
        "iou_per_class": [float(x) for x in iou_per_class],
        "f1_per_class": [float(x) for x in f1_per_class],
        "precision_per_class": [float(x) for x in precision_per_class],
        "recall_per_class": [float(x) for x in recall_per_class],
    }


def calculate_metrics(prediction, ground_truth):
    valid_mask = ground_truth != IGNORE_INDEX
    pred_flat = prediction[valid_mask].reshape(-1)
    gt_flat = ground_truth[valid_mask].reshape(-1)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    if pred_flat.size > 0:
        valid_pred = (pred_flat >= 0) & (pred_flat < NUM_CLASSES)
        gt_flat = gt_flat[valid_pred]
        pred_flat = pred_flat[valid_pred]
        np.add.at(cm, (gt_flat.astype(np.int64), pred_flat.astype(np.int64)), 1)
    metrics = compute_metrics_from_confusion_matrix(cm)
    metrics.update(
        {
            "confusion_matrix": cm.tolist(),
            "valid_pixels": int(valid_mask.sum()),
            "total_pixels": int(ground_truth.size),
            "ignored_pixels": int(ground_truth.size - valid_mask.sum()),
        }
    )
    return metrics


def save_prediction_outputs(output_dir, image_name, image, label, prediction, metrics, image_format="png", dpi=150):
    output_dir = Path(output_dir)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)

    Image.fromarray(prediction.astype(np.uint8)).save(
        output_dir / "predictions" / f"{image_name}_prediction_id.{image_format}"
    )
    Image.fromarray(label.astype(np.uint8)).save(
        output_dir / "predictions" / f"{image_name}_ground_truth_id.{image_format}"
    )
    Image.fromarray(colorize_label(prediction)).save(
        output_dir / "predictions" / f"{image_name}_prediction.{image_format}"
    )
    Image.fromarray(colorize_label(label)).save(
        output_dir / "predictions" / f"{image_name}_ground_truth.{image_format}"
    )
    create_visualization(
        image,
        label,
        prediction,
        output_dir / "visualizations" / f"{image_name}_comparison.{image_format}",
        metrics,
        dpi=dpi,
    )
    with open(output_dir / "metrics" / f"{image_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_name": image_name,
                "class_ids": CLASS_IDS,
                "class_names": [CLASS_INFO[i]["name"] for i in CLASS_IDS],
                "metrics": metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def create_visualization(image, ground_truth, prediction, save_path, metrics, dpi=150):
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes[0, 0].imshow(image[:, :, :3])
    axes[0, 0].set_title("Input Image", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(colorize_label(ground_truth))
    axes[0, 1].set_title("Ground Truth", fontsize=14, fontweight="bold")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(colorize_label(prediction))
    axes[1, 0].set_title("SegFormer Prediction", fontsize=14, fontweight="bold")
    axes[1, 0].axis("off")

    overlay = 0.6 * (image[:, :, :3].astype(np.float32) / 255.0)
    overlay += 0.4 * (colorize_label(prediction).astype(np.float32) / 255.0)
    axes[1, 1].imshow(np.clip(overlay * 255, 0, 255).astype(np.uint8))
    axes[1, 1].set_title("Prediction Overlay", fontsize=14, fontweight="bold")
    axes[1, 1].axis("off")

    metrics_text = (
        f"OA: {metrics['overall_accuracy']:.4f}\n"
        f"mIoU: {metrics['mean_iou']:.4f}\n"
        f"Mean F1: {metrics['mean_f1']:.4f}\n"
        f"FWIoU: {metrics['frequency_weighted_iou']:.4f}"
    )
    fig.text(
        0.02,
        0.02,
        metrics_text,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85),
        verticalalignment="bottom",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_overall_metrics(output_dir, all_results, config, run_context=None):
    output_dir = Path(output_dir)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    dataset_cm = np.sum(
        [np.asarray(r["metrics"]["confusion_matrix"], dtype=np.int64) for r in all_results],
        axis=0,
    )
    dataset_metrics = compute_metrics_from_confusion_matrix(dataset_cm)
    overall = {
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_name": config.get("experiment", {}).get("name", "SegFormer Potsdam baseline"),
        "num_images": len(all_results),
        "image_names": [r["image_name"] for r in all_results],
        "class_ids": CLASS_IDS,
        "class_names": [CLASS_INFO[i]["name"] for i in CLASS_IDS],
        "class_colors": [CLASS_INFO[i]["color"] for i in CLASS_IDS],
        "ignore_index": IGNORE_INDEX,
        "model": config.get("model", {}),
        "run_context": run_context or {},
        "primary_metrics": [
            "dataset_mean_iou",
            "dataset_mean_f1",
            "dataset_overall_accuracy",
            "dataset_frequency_weighted_iou",
        ],
        "dataset_overall_accuracy": dataset_metrics["overall_accuracy"],
        "dataset_mean_iou": dataset_metrics["mean_iou"],
        "dataset_mean_f1": dataset_metrics["mean_f1"],
        "dataset_mean_precision": dataset_metrics["mean_precision"],
        "dataset_mean_recall": dataset_metrics["mean_recall"],
        "dataset_frequency_weighted_iou": dataset_metrics["frequency_weighted_iou"],
        "dataset_iou_per_class": dataset_metrics["iou_per_class"],
        "dataset_f1_per_class": dataset_metrics["f1_per_class"],
        "dataset_precision_per_class": dataset_metrics["precision_per_class"],
        "dataset_recall_per_class": dataset_metrics["recall_per_class"],
        "dataset_confusion_matrix": dataset_cm.tolist(),
        "dataset_valid_pixels": int(sum(r["metrics"]["valid_pixels"] for r in all_results)),
        "dataset_ignored_pixels": int(sum(r["metrics"]["ignored_pixels"] for r in all_results)),
    }
    if all_results:
        for key in [
            "overall_accuracy",
            "mean_iou",
            "mean_f1",
            "mean_precision",
            "mean_recall",
            "frequency_weighted_iou",
        ]:
            overall[f"average_{key}"] = float(np.mean([r["metrics"][key] for r in all_results]))
        for class_index, class_id in enumerate(CLASS_IDS):
            overall[f"dataset_iou_class_{class_id}"] = dataset_metrics["iou_per_class"][class_index]
            overall[f"dataset_f1_class_{class_id}"] = dataset_metrics["f1_per_class"][class_index]
            overall[f"dataset_precision_class_{class_id}"] = dataset_metrics["precision_per_class"][class_index]
            overall[f"dataset_recall_class_{class_id}"] = dataset_metrics["recall_per_class"][class_index]

    with open(output_dir / "metrics" / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    rows = []
    for result in all_results:
        row = {
            "image_name": result["image_name"],
            "overall_accuracy": result["metrics"]["overall_accuracy"],
            "mean_iou": result["metrics"]["mean_iou"],
            "mean_f1": result["metrics"]["mean_f1"],
            "mean_precision": result["metrics"]["mean_precision"],
            "mean_recall": result["metrics"]["mean_recall"],
            "frequency_weighted_iou": result["metrics"]["frequency_weighted_iou"],
        }
        for class_index, class_id in enumerate(CLASS_IDS):
            row[f"iou_class_{class_id}"] = result["metrics"]["iou_per_class"][class_index]
            row[f"f1_class_{class_id}"] = result["metrics"]["f1_per_class"][class_index]
            row[f"precision_class_{class_id}"] = result["metrics"]["precision_per_class"][class_index]
            row[f"recall_class_{class_id}"] = result["metrics"]["recall_per_class"][class_index]
        rows.append(row)

    with open(output_dir / "metrics" / "per_image_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image_name"])
        writer.writeheader()
        writer.writerows(rows)
    return overall


def prepare_output_dir(config, config_path=None):
    output_dir = Path(config.get("paths", {}).get("output_dir", "results_segformer_potsdam_rgb"))
    for subdir in ["checkpoints", "logs", "metrics", "predictions", "visualizations"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    if config_path:
        save_config_snapshot(config_path, output_dir)
    return output_dir


def id2label():
    return {i: CLASS_INFO[i]["name"] for i in CLASS_IDS}


def label2id():
    return {CLASS_INFO[i]["name"]: i for i in CLASS_IDS}
