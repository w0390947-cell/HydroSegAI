#!/usr/bin/env python3
"""
Exp2-C: SAM3 + Potsdam prompt ensemble with mask quality filtering.

实验变量：
- 继承 Exp2-B 的 prompt ensemble。
- 不主动输入 clutter/background prompt。
- 在 patch 内多类别融合前，加入固定的类别面积先验过滤。
- class 5 clutter/background 仍作为 Potsdam 标准评估类别和 fallback 类别。

运行示例：
  python exp2c_sam3_potsdam_prompt_ensemble_mask_filter.py --dry-run
  python exp2c_sam3_potsdam_prompt_ensemble_mask_filter.py
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from exp1_sam3_potsdam_zeroshot_baseline import (
    PotsdamSAM3Evaluator,
    SAM3_AVAILABLE,
)
from exp2a_sam3_potsdam_no_background_prompt import (
    PROMPT_CLASS_IDS,
    count_patches_with_exp1_logic,
    image_and_label_paths,
    load_runtime_settings,
)
from exp2b_sam3_potsdam_visual_prompt_ensemble import (
    Exp2BVisualPromptEnsembleEvaluator,
    load_prompt_variants,
    print_prompt_variants,
)


EXPERIMENT_NAME = "Exp2-C: SAM3 Potsdam prompt ensemble with mask filter"
DEFAULT_CONFIG_PATH = "exp2c_sam3_potsdam_prompt_ensemble_mask_filter.yaml"
DEFAULT_OUTPUT_DIR = "/home/anjou/PythonENV/Test_11/results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter"

DEFAULT_AREA_RULES = {
    0: {"min": 0.001, "max": 0.80},
    1: {"min": 0.0005, "max": 0.60},
    2: {"min": 0.001, "max": 0.80},
    3: {"min": 0.0003, "max": 0.50},
    4: {"min": 0.00001, "max": 0.05},
}


def load_mask_filter_config(config):
    """读取 Exp2-C mask 过滤规则。"""
    filter_config = config.get("mask_filter", {})
    area_config = filter_config.get("class_area_ratio", {})

    area_rules = {}
    for class_id, default_rule in DEFAULT_AREA_RULES.items():
        raw_rule = area_config.get(f"class_{class_id}", {})
        min_ratio = float(raw_rule.get("min", default_rule["min"]))
        max_ratio = float(raw_rule.get("max", default_rule["max"]))
        if min_ratio < 0 or max_ratio <= 0 or min_ratio > max_ratio:
            raise ValueError(
                f"class_{class_id} 面积过滤阈值无效: min={min_ratio}, max={max_ratio}"
            )
        area_rules[class_id] = {"min": min_ratio, "max": max_ratio}

    return {
        "enabled": bool(filter_config.get("enabled", True)),
        "area_rules": area_rules,
    }


class Exp2CPromptEnsembleMaskFilterEvaluator(Exp2BVisualPromptEnsembleEvaluator):
    """实验二 C：Exp2-B 加固定 mask 面积过滤。"""

    def __init__(self, *args, mask_filter_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_filter_enabled = bool((mask_filter_config or {}).get("enabled", True))
        self.area_rules = (mask_filter_config or {}).get("area_rules", DEFAULT_AREA_RULES)
        self.mask_filter_stats = {
            class_id: {
                "input_masks": 0,
                "kept_masks": 0,
                "filtered_small": 0,
                "filtered_large": 0,
                "filtered_invalid": 0,
            }
            for class_id in self.prompt_class_ids
        }

    def predict_patch(self, patch, text_prompts):
        predictions = super().predict_patch(patch, text_prompts)
        if not self.mask_filter_enabled:
            return predictions
        return self._filter_predictions_by_area(predictions)

    def _filter_predictions_by_area(self, predictions):
        masks = predictions.get("masks", [])
        boxes = predictions.get("boxes", [])
        scores = predictions.get("scores", [])
        classes = predictions.get("classes", [])

        kept_masks = []
        kept_boxes = []
        kept_scores = []
        kept_classes = []

        for idx, mask in enumerate(masks):
            if idx >= len(scores) or idx >= len(classes):
                continue

            class_id = int(classes[idx])
            if class_id not in self.area_rules:
                kept_masks.append(mask)
                kept_boxes.append(boxes[idx] if idx < len(boxes) else None)
                kept_scores.append(scores[idx])
                kept_classes.append(class_id)
                continue

            stats = self.mask_filter_stats[class_id]
            stats["input_masks"] += 1

            mask_array = np.asarray(mask)
            if mask_array.size == 0:
                stats["filtered_invalid"] += 1
                continue

            mask_area_ratio = float(np.count_nonzero(mask_array > 0.5) / mask_array.size)
            rule = self.area_rules[class_id]

            if mask_area_ratio < rule["min"]:
                stats["filtered_small"] += 1
                continue
            if mask_area_ratio > rule["max"]:
                stats["filtered_large"] += 1
                continue

            stats["kept_masks"] += 1
            kept_masks.append(mask)
            kept_boxes.append(boxes[idx] if idx < len(boxes) else None)
            kept_scores.append(scores[idx])
            kept_classes.append(class_id)

        return {
            "masks": kept_masks,
            "boxes": kept_boxes,
            "scores": kept_scores,
            "classes": kept_classes,
        }

    def save_overall_metrics(self, all_results):
        """保存总体指标，并补充 Exp2-C 的实验元数据。"""
        PotsdamSAM3Evaluator.save_overall_metrics(self, all_results)

        prompt_calls_per_patch = sum(
            len(self.prompt_variants[class_id]) for class_id in self.prompt_class_ids
        )
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "variable_under_test": (
                "visual prompt ensemble with class-aware mask area filtering"
            ),
            "prompt_class_ids": self.prompt_class_ids,
            "prompt_class_names": [
                self.class_info[class_id]["name"] for class_id in self.prompt_class_ids
            ],
            "excluded_prompt_class_ids": self.excluded_prompt_class_ids,
            "excluded_prompt_class_names": [
                self.class_info[class_id]["name"]
                for class_id in self.excluded_prompt_class_ids
            ],
            "fallback_class_id": self.default_class_id,
            "fallback_class_name": self.class_info[self.default_class_id]["name"],
            "score_threshold": self.SCORE_THRESHOLD,
            "prompt_variants": self.prompt_variants,
            "prompt_calls_per_patch": prompt_calls_per_patch,
            "mask_filter_enabled": self.mask_filter_enabled,
            "mask_filter_area_rules": self.area_rules,
            "mask_filter_stats": self.mask_filter_stats,
            "note": (
                "Masks are filtered by fixed class-aware area-ratio thresholds before "
                "confidence-based patch fusion. Thresholds are not tuned per image."
            ),
        }

        metadata_path = self.output_dir / "metrics" / "experiment_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        overall_path = self.output_dir / "metrics" / "overall_metrics.json"
        if overall_path.exists():
            with open(overall_path, "r", encoding="utf-8") as f:
                overall_metrics = json.load(f)
            overall_metrics.update(metadata)
            with open(overall_path, "w", encoding="utf-8") as f:
                json.dump(overall_metrics, f, indent=2, ensure_ascii=False)


def print_filter_rules(class_info, mask_filter_config, score_threshold):
    print(f"mask score_threshold: {score_threshold}")
    print(f"mask area filtering: {mask_filter_config['enabled']}")
    prompt_class_ids = mask_filter_config.get("prompt_class_ids", PROMPT_CLASS_IDS)
    for class_id in prompt_class_ids:
        rule = mask_filter_config["area_rules"].get(class_id)
        if rule is None:
            print(f"  {class_id}: {class_info[class_id]['name']} 未设置面积过滤规则")
            continue
        print(
            f"  {class_id}: {class_info[class_id]['name']} "
            f"min={rule['min']} max={rule['max']}"
        )


def dry_run(settings, prompt_variants, mask_filter_config):
    """只检查配置、样本和预计 prompt 数量，不加载 SAM3 模型。"""
    class_info = settings["class_info"]
    prompt_class_ids = settings["prompt_class_ids"]
    excluded_prompt_class_ids = settings["excluded_prompt_class_ids"]

    total_patches = 0
    missing_files = []
    size_mismatches = []

    for image_id in settings["test_images"]:
        image_path, label_path = image_and_label_paths(settings, image_id)
        if not image_path.exists() or not label_path.exists():
            missing_files.append((image_id, str(image_path), str(label_path)))
            continue

        with Image.open(image_path) as image:
            image_w, image_h = image.size
        with Image.open(label_path) as label:
            label_w, label_h = label.size

        if (image_w, image_h) != (label_w, label_h):
            size_mismatches.append((image_id, (image_w, image_h), (label_w, label_h)))

        total_patches += count_patches_with_exp1_logic(
            image_w,
            image_h,
            settings["patch_size"],
            settings["stride"],
        )

    prompt_calls_per_patch = sum(len(prompt_variants[class_id]) for class_id in prompt_class_ids)

    print("=" * 72)
    print(EXPERIMENT_NAME)
    print("=" * 72)
    print(f"配置文件: {settings['config'] and '已读取' or '未读取，使用默认值'}")
    print(f"基础目录: {settings['base_dir']}")
    print(f"输出目录: {settings['output_dir']}")
    print(f"图像数量: {len(settings['test_images'])}")
    print(f"Patch 参数: size={settings['patch_size']}, stride={settings['stride']}")
    print(f"预计 patch 总数: {total_patches}")
    print(f"每个 patch 文本 prompt 调用数: {prompt_calls_per_patch}")
    print(f"预计 SAM3 文本 prompt 调用数: {total_patches * prompt_calls_per_patch}")
    print_prompt_variants(
        class_info,
        prompt_variants,
        prompt_class_ids,
        excluded_prompt_class_ids,
    )
    print_filter_rules(class_info, mask_filter_config, settings["score_threshold"])

    if missing_files:
        print(f"缺失文件数量: {len(missing_files)}")
        for image_id, image_path, label_path in missing_files[:5]:
            print(f"  {image_id}: image={image_path}, label={label_path}")
    else:
        print("文件检查: 通过")

    if size_mismatches:
        print(f"图像/标签尺寸不一致数量: {len(size_mismatches)}")
        for image_id, image_size, label_size in size_mismatches[:5]:
            print(f"  {image_id}: image={image_size}, label={label_size}")
    else:
        print("尺寸检查: 通过")

    print("=" * 72)


def print_header(settings, prompt_variants, mask_filter_config, config_path):
    print("=" * 72)
    print(EXPERIMENT_NAME)
    print("=" * 72)
    print(f"基础目录: {settings['base_dir']}")
    print(f"输出目录: {settings['output_dir']}")
    print(f"配置文件: {config_path if settings['config'] else '未使用配置文件'}")
    print(f"测试图像数量: {len(settings['test_images'])}")
    print(f"Patch 参数: size={settings['patch_size']}, stride={settings['stride']}, score_threshold={settings['score_threshold']}")
    print(f"设备配置: {settings['device_type']}")
    print(f"GPU dtype: {settings['gpu_dtype']}")
    print(f"当前 torch CUDA: {torch.cuda.is_available()}")
    print(f"每个 patch 文本 prompt 调用数: {sum(len(v) for v in prompt_variants.values())}")
    print_filter_rules(settings["class_info"], mask_filter_config, settings["score_threshold"])
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Exp2-C: SAM3 Potsdam prompt ensemble + mask filter"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Exp2-C YAML 配置文件路径",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Exp2-C 输出目录；默认 {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查样本、路径和预计 patch/prompt 数量，不加载 SAM3 模型",
    )
    args = parser.parse_args()

    settings = load_runtime_settings(args)
    if not args.output_dir and "paths" not in settings["config"]:
        settings["output_dir"] = DEFAULT_OUTPUT_DIR

    prompt_variants = load_prompt_variants(
        settings["config"],
        settings["class_info"],
        settings["prompt_class_ids"],
    )
    mask_filter_config = load_mask_filter_config(settings["config"])
    mask_filter_config["prompt_class_ids"] = settings["prompt_class_ids"]

    if args.dry_run:
        dry_run(settings, prompt_variants, mask_filter_config)
        return

    print_header(settings, prompt_variants, mask_filter_config, args.config)

    if not SAM3_AVAILABLE:
        print("错误: SAM3 模块不可用，请先安装 SAM3")
        return

    evaluator = Exp2CPromptEnsembleMaskFilterEvaluator(
        base_dir=settings["base_dir"],
        output_dir=settings["output_dir"],
        checkpoint_path=settings["checkpoint_path"],
        patch_size=settings["patch_size"],
        stride=settings["stride"],
        score_threshold=settings["score_threshold"],
        class_info=settings["class_info"],
        gpu_dtype=settings["gpu_dtype"],
        device_type=settings["device_type"],
        prompt_variants=prompt_variants,
        mask_filter_config=mask_filter_config,
        prompt_class_ids=settings["prompt_class_ids"],
        excluded_prompt_class_ids=settings["excluded_prompt_class_ids"],
    )

    all_results = []
    for image_id in settings["test_images"]:
        image_path, label_path = image_and_label_paths(settings, image_id)

        if not image_path.exists():
            print(f"错误: 图像文件不存在 - {image_path}")
            continue
        if not label_path.exists():
            print(f"错误: 标签文件不存在 - {label_path}")
            continue

        try:
            result = evaluator.process_image(image_path, label_path)
            all_results.append(result)

            print(f"\n{result['image_name']} 评估结果:")
            print(f"  Overall Accuracy: {result['metrics']['overall_accuracy']:.4f}")
            print(f"  Mean IoU: {result['metrics']['mean_iou']:.4f}")
            print(f"  Mean F1: {result['metrics']['mean_f1']:.4f}")
            print(f"  Mean Precision: {result['metrics']['mean_precision']:.4f}")
            print(f"  Mean Recall: {result['metrics']['mean_recall']:.4f}")
            print(f"  Frequency Weighted IoU: {result['metrics']['frequency_weighted_iou']:.4f}")

            for class_index, (class_id, info) in enumerate(evaluator.class_info.items()):
                iou = result["metrics"]["iou_per_class"][class_index]
                print(f"  IoU ({info['name']}): {iou:.4f}")

        except Exception as exc:
            print(f"处理 {image_id} 时出错: {exc}")
            import traceback

            traceback.print_exc()
            continue

    if all_results:
        print("\n" + "=" * 72)
        print("保存总体评估指标...")
        evaluator.save_overall_metrics(all_results)

        metrics_path = Path(settings["output_dir"]) / "metrics" / "overall_metrics.json"
        with open(metrics_path, "r", encoding="utf-8") as f:
            overall = json.load(f)

        print("\n总体评估结果:")
        print(f"  数据集 Overall Accuracy: {overall['dataset_overall_accuracy']:.4f}")
        print(f"  数据集 Mean IoU: {overall['dataset_mean_iou']:.4f}")
        print(f"  数据集 Mean F1: {overall['dataset_mean_f1']:.4f}")
        print(f"  数据集 Mean Precision: {overall['dataset_mean_precision']:.4f}")
        print(f"  数据集 Mean Recall: {overall['dataset_mean_recall']:.4f}")
        print(f"  数据集 Frequency Weighted IoU: {overall['dataset_frequency_weighted_iou']:.4f}")
        print(f"\n评估完成！结果保存在: {settings['output_dir']}")
    else:
        print("没有成功处理任何图像。")


if __name__ == "__main__":
    main()
