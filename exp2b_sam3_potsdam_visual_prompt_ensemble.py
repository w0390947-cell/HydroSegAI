#!/usr/bin/env python3
"""
Exp2-B: SAM3 + Potsdam visual prompt ensemble.

实验变量：
- 继承 Exp2-A 的背景处理：不主动输入 clutter/background prompt。
- 对 5 个前景类别使用更视觉化的 prompt ensemble。
- class 5 clutter/background 仍作为 Potsdam 标准评估类别和 fallback 类别。

运行示例：
  python exp2b_sam3_potsdam_visual_prompt_ensemble.py --dry-run
  python exp2b_sam3_potsdam_visual_prompt_ensemble.py
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from exp1_sam3_potsdam_zeroshot_baseline import (
    PotsdamSAM3Evaluator,
    SAM3_AVAILABLE,
)
from exp2a_sam3_potsdam_no_background_prompt import (
    EXCLUDED_PROMPT_CLASS_IDS,
    Exp2ANoBackgroundPromptEvaluator,
    count_patches_with_exp1_logic,
    image_and_label_paths,
    load_prompt_class_config,
    load_runtime_settings,
)


EXPERIMENT_NAME = "Exp2-B: SAM3 Potsdam visual prompt ensemble"
DEFAULT_CONFIG_PATH = "exp2b_sam3_potsdam_visual_prompt_ensemble.yaml"
DEFAULT_OUTPUT_DIR = "/home/anjou/PythonENV/Test_11/results_exp2b_sam3_potsdam_visual_prompt_ensemble"

DEFAULT_PROMPT_VARIANTS = {
    0: [
        "impervious surface",
        "road and pavement",
        "asphalt road",
        "paved area",
        "parking lot",
    ],
    1: [
        "building",
        "buildings",
        "rooftop",
        "building roof",
        "house roof",
    ],
    2: [
        "low vegetation",
        "grass",
        "lawn",
        "low plants",
        "ground vegetation",
    ],
    3: [
        "tree",
        "trees",
        "tree crown",
        "tall tree",
        "urban trees",
    ],
    4: [
        "car",
        "cars",
        "vehicle",
        "small vehicle",
        "parked car",
    ],
}


def load_prompt_variants(config, class_info, prompt_class_ids=None):
    """从配置文件读取 prompt ensemble，缺省时使用代码内置视觉化 prompt。"""
    ensemble_config = config.get("prompt_ensemble", {})
    prompts_config = ensemble_config.get("prompts", {})
    if prompt_class_ids is None:
        prompt_class_ids, _ = load_prompt_class_config(config, class_info)

    prompt_variants = {}
    for class_id in prompt_class_ids:
        key = f"class_{class_id}"
        default_variants = DEFAULT_PROMPT_VARIANTS.get(
            class_id,
            [class_info[class_id]["prompt"]],
        )
        variants = prompts_config.get(key, default_variants)
        variants = [str(prompt).strip() for prompt in variants if str(prompt).strip()]
        if not variants:
            variants = [class_info[class_id]["prompt"]]
        prompt_variants[class_id] = variants

    return prompt_variants


class Exp2BVisualPromptEnsembleEvaluator(Exp2ANoBackgroundPromptEvaluator):
    """实验二 B：前景类别使用 prompt ensemble，不主动 prompt 背景。"""

    def __init__(self, *args, prompt_variants=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_variants = prompt_variants or DEFAULT_PROMPT_VARIANTS
        self._validate_prompt_variants()

    def _validate_prompt_variants(self):
        missing = [
            class_id for class_id in self.prompt_class_ids
            if class_id not in self.prompt_variants or not self.prompt_variants[class_id]
        ]
        if missing:
            raise ValueError(f"Exp2-B prompt ensemble 缺少类别: {missing}")

    def predict_patch(self, patch, text_prompts):
        """
        复用实验一的 SAM3 patch 预测实现，仅将父类传入的单 prompt 展开为多 prompt。

        text_prompts 由实验一 process_image() 生成，包含 6 类；这里丢弃 class 5，
        并对 class 0-4 展开多个视觉化 prompt，所有变体仍映射回原 class_id。
        """
        expanded_prompts = []
        for prompt_index, prompt_item in enumerate(text_prompts):
            if isinstance(prompt_item, (tuple, list)) and len(prompt_item) == 2:
                class_id = int(prompt_item[0])
            else:
                class_id = prompt_index

            if class_id not in self.prompt_class_ids:
                continue

            for prompt in self.prompt_variants[class_id]:
                expanded_prompts.append((class_id, prompt))

        return PotsdamSAM3Evaluator.predict_patch(self, patch, expanded_prompts)

    def save_overall_metrics(self, all_results):
        """保存总体指标，并补充 Exp2-B 的实验元数据。"""
        PotsdamSAM3Evaluator.save_overall_metrics(self, all_results)

        prompt_calls_per_patch = sum(
            len(self.prompt_variants[class_id]) for class_id in self.prompt_class_ids
        )
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "variable_under_test": "visual prompt ensemble without active clutter/background prompt",
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
            "prompt_variants": self.prompt_variants,
            "prompt_calls_per_patch": prompt_calls_per_patch,
            "note": (
                "Each foreground class is queried with multiple visual prompts. "
                "All masks from variants of the same class are mapped back to that class."
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


def print_prompt_variants(
    class_info,
    prompt_variants,
    prompt_class_ids=None,
    excluded_prompt_class_ids=None,
):
    prompt_class_ids = prompt_class_ids or list(prompt_variants.keys())
    excluded_prompt_class_ids = excluded_prompt_class_ids or EXCLUDED_PROMPT_CLASS_IDS

    print("主动 prompt ensemble 类别:")
    for class_id in prompt_class_ids:
        print(f"  {class_id}: {class_info[class_id]['name']}")
        for prompt in prompt_variants[class_id]:
            print(f"    - {prompt}")
    print("不主动 prompt 类别:")
    for class_id in excluded_prompt_class_ids:
        print(f"  {class_id}: {class_info[class_id]['name']} -> {class_info[class_id]['prompt']}")


def dry_run(settings, prompt_variants):
    """只检查配置、样本和 prompt ensemble 数量，不加载 SAM3 模型。"""
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


def print_header(settings, prompt_variants, config_path):
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
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Exp2-B: SAM3 Potsdam visual prompt ensemble，不主动输入背景/杂波文本提示"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Exp2-B YAML 配置文件路径",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Exp2-B 输出目录；默认 {DEFAULT_OUTPUT_DIR}",
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

    if args.dry_run:
        dry_run(settings, prompt_variants)
        return

    print_header(settings, prompt_variants, args.config)

    if not SAM3_AVAILABLE:
        print("错误: SAM3 模块不可用，请先安装 SAM3")
        return

    evaluator = Exp2BVisualPromptEnsembleEvaluator(
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
