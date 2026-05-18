#!/usr/bin/env python3
"""
Exp2-A: SAM3 + Potsdam 零样本遥感分割，不主动输入 clutter/background 提示。

实验变量：
- 保持实验一的数据、切片、置信度融合、fallback 和指标计算逻辑不变。
- 仅对 Potsdam 5 个前景/语义类别输入文本提示：
  0 impervious surface, 1 building, 2 low vegetation, 3 tree, 4 car。
- 不对 5 clutter/background 输入文本提示。
- class 5 仍然作为 Potsdam 标准评估类别，并作为“已覆盖但无正置信度预测区域”的 fallback 类别。

运行示例：
  python exp2a_sam3_potsdam_no_background_prompt.py --dry-run
  python exp2a_sam3_potsdam_no_background_prompt.py
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
    build_class_info,
    build_test_images,
    load_yaml_config,
)


EXPERIMENT_NAME = "Exp2-A: SAM3 Potsdam no background prompt"
PROMPT_CLASS_IDS = [0, 1, 2, 3, 4]
EXCLUDED_PROMPT_CLASS_IDS = [5]
DEFAULT_CONFIG_PATH = "exp2a_sam3_potsdam_no_background_prompt.yaml"
DEFAULT_OUTPUT_DIR = "/home/anjou/PythonENV/Test_11/results_exp2a_sam3_potsdam_no_background_prompt"


def _read_class_id_list(config_value, default_value, field_name):
    if config_value is None:
        return list(default_value)
    if not isinstance(config_value, list):
        raise ValueError(f"{field_name} 必须是类别 id 列表，实际为: {config_value}")

    class_ids = []
    for raw_id in config_value:
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} 中包含无法转换为整数的类别 id: {raw_id}") from exc
        if class_id not in class_ids:
            class_ids.append(class_id)
    return class_ids


def load_prompt_class_config(config, class_info):
    """
    读取并校验 Exp2/Exp3 的主动 prompt 类别配置。

    这些实验的核心约束是：clutter/background 不作为主动文本 prompt，
    但仍然作为 Potsdam 评估类别和 fallback 类别。
    """
    ensemble_config = config.get("prompt_ensemble", {})
    prompt_class_ids = _read_class_id_list(
        ensemble_config.get("prompt_class_ids"),
        PROMPT_CLASS_IDS,
        "prompt_ensemble.prompt_class_ids",
    )
    excluded_prompt_class_ids = _read_class_id_list(
        ensemble_config.get("excluded_prompt_class_ids"),
        EXCLUDED_PROMPT_CLASS_IDS,
        "prompt_ensemble.excluded_prompt_class_ids",
    )

    if not prompt_class_ids:
        raise ValueError("prompt_ensemble.prompt_class_ids 不能为空")

    unknown_prompt_ids = [class_id for class_id in prompt_class_ids if class_id not in class_info]
    unknown_excluded_ids = [
        class_id for class_id in excluded_prompt_class_ids if class_id not in class_info
    ]
    if unknown_prompt_ids or unknown_excluded_ids:
        raise ValueError(
            f"类别配置不完整，prompt 缺失 {unknown_prompt_ids}，"
            f"排除 prompt 缺失 {unknown_excluded_ids}"
        )

    overlap = sorted(set(prompt_class_ids) & set(excluded_prompt_class_ids))
    if overlap:
        raise ValueError(f"主动 prompt 类别和排除 prompt 类别不能重叠: {overlap}")

    configured_class_ids = set(class_info.keys())
    covered_class_ids = set(prompt_class_ids) | set(excluded_prompt_class_ids)
    missing_from_partition = sorted(configured_class_ids - covered_class_ids)
    if missing_from_partition:
        raise ValueError(
            "prompt_ensemble.prompt_class_ids 和 "
            "prompt_ensemble.excluded_prompt_class_ids 必须共同覆盖所有评估类别；"
            f"以下类别既未主动 prompt，也未声明为排除: {missing_from_partition}"
        )

    if 5 in prompt_class_ids:
        raise ValueError(
            "Exp2/Exp3 当前定义为不主动 prompt clutter/background，"
            "因此 class 5 不能出现在 prompt_class_ids 中。"
        )

    return prompt_class_ids, excluded_prompt_class_ids


class Exp2ANoBackgroundPromptEvaluator(PotsdamSAM3Evaluator):
    """实验二 A：不主动 prompt clutter/background 的 Potsdam SAM3 评估器。"""

    def __init__(self, *args, **kwargs):
        prompt_class_ids = kwargs.pop("prompt_class_ids", None)
        excluded_prompt_class_ids = kwargs.pop("excluded_prompt_class_ids", None)
        super().__init__(*args, **kwargs)
        self.prompt_class_ids = (
            list(prompt_class_ids) if prompt_class_ids is not None else list(PROMPT_CLASS_IDS)
        )
        self.excluded_prompt_class_ids = (
            list(excluded_prompt_class_ids)
            if excluded_prompt_class_ids is not None
            else list(EXCLUDED_PROMPT_CLASS_IDS)
        )
        self._validate_exp2a_class_setup()

    def _validate_exp2a_class_setup(self):
        missing = [class_id for class_id in self.prompt_class_ids if class_id not in self.class_info]
        if missing:
            raise ValueError(f"Exp2-A prompt 类别不在 class_info 中: {missing}")

        missing_excluded = [
            class_id for class_id in self.excluded_prompt_class_ids
            if class_id not in self.class_info
        ]
        if missing_excluded:
            raise ValueError(f"Exp2-A 排除 prompt 的类别不在 class_info 中: {missing_excluded}")

        overlap = sorted(set(self.prompt_class_ids) & set(self.excluded_prompt_class_ids))
        if overlap:
            raise ValueError(f"Exp2-A 主动 prompt 类别和排除 prompt 类别不能重叠: {overlap}")

        if self.default_class_id in self.prompt_class_ids:
            raise ValueError(
                f"Exp2-A 要求 fallback 类别 {self.default_class_id} "
                "不参与主动文本 prompt。"
            )

    def predict_patch(self, patch, text_prompts):
        """
        复用实验一的 patch 预测逻辑，只在进入父类逻辑前过滤文本提示。

        父类 process_image() 仍会按实验一生成 6 类 prompt；这里删除 class 5，
        其余切片、融合、fallback、指标和保存流程全部继承父类。
        """
        filtered_prompts = []
        for prompt_index, prompt_item in enumerate(text_prompts):
            if isinstance(prompt_item, (tuple, list)) and len(prompt_item) == 2:
                class_id = int(prompt_item[0])
            else:
                class_id = prompt_index

            if class_id in self.prompt_class_ids:
                filtered_prompts.append(prompt_item)

        return super().predict_patch(patch, filtered_prompts)

    def save_overall_metrics(self, all_results):
        """保存总体指标，并补充 Exp2-A 的实验元数据。"""
        super().save_overall_metrics(all_results)

        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "variable_under_test": "remove active clutter/background text prompt",
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
            "note": (
                "Class 5 remains an evaluation class and fallback class, but SAM3 "
                "does not receive its text prompt in this experiment."
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


def load_runtime_settings(args):
    config = load_yaml_config(args.config)
    paths_config = config.get("paths", {})
    image_processing_config = config.get("image_processing", {})
    device_config = config.get("device", {})

    base_dir = paths_config.get("base_dir", "/home/anjou/PythonENV/Test_11/Potsdam")
    output_dir = args.output_dir or paths_config.get("output_dir", DEFAULT_OUTPUT_DIR)
    checkpoint_path = paths_config.get(
        "checkpoint_path",
        "/home/anjou/PythonENV/Test_11/sam3/checkpoints/sam3.1_multiplex.pt",
    )
    image_subdir = paths_config.get("image_subdir", "2_Ortho_RGB/2_Ortho_RGB")
    label_subdir = paths_config.get("label_subdir", "5_Labels_all_noBoundary")
    image_pattern = paths_config.get("image_pattern", "{image_id}_RGB.tif")
    label_pattern = paths_config.get("label_pattern", "{image_id}_label_noBoundary.tif")

    class_info = build_class_info(config)
    prompt_class_ids, excluded_prompt_class_ids = load_prompt_class_config(config, class_info)
    test_images = build_test_images(config, base_dir, image_subdir, label_subdir)

    return {
        "config": config,
        "base_dir": base_dir,
        "output_dir": output_dir,
        "checkpoint_path": checkpoint_path,
        "image_subdir": image_subdir,
        "label_subdir": label_subdir,
        "image_pattern": image_pattern,
        "label_pattern": label_pattern,
        "test_images": test_images,
        "class_info": class_info,
        "prompt_class_ids": prompt_class_ids,
        "excluded_prompt_class_ids": excluded_prompt_class_ids,
        "patch_size": image_processing_config.get(
            "patch_size", PotsdamSAM3Evaluator.PATCH_SIZE
        ),
        "stride": image_processing_config.get("stride", PotsdamSAM3Evaluator.STRIDE),
        "score_threshold": image_processing_config.get(
            "score_threshold", PotsdamSAM3Evaluator.SCORE_THRESHOLD
        ),
        "gpu_dtype": device_config.get("gpu_dtype", "float32"),
        "device_type": device_config.get("type", "auto"),
    }


def image_and_label_paths(settings, image_id):
    base_dir = Path(settings["base_dir"])
    image_path = (
        base_dir
        / settings["image_subdir"]
        / settings["image_pattern"].format(image_id=image_id)
    )
    label_path = (
        base_dir
        / settings["label_subdir"]
        / settings["label_pattern"].format(image_id=image_id)
    )
    return image_path, label_path


def count_patches_with_exp1_logic(image_w, image_h, patch_size, stride):
    """复用实验一评估器的边缘对齐切片起点逻辑，仅用于 dry-run 统计。"""
    evaluator = object.__new__(PotsdamSAM3Evaluator)
    evaluator.PATCH_SIZE = patch_size
    evaluator.STRIDE = stride
    x_starts = PotsdamSAM3Evaluator._generate_patch_starts(evaluator, image_w)
    y_starts = PotsdamSAM3Evaluator._generate_patch_starts(evaluator, image_h)
    return len(x_starts) * len(y_starts)


def dry_run(settings):
    """只检查配置、样本和切片数量，不加载 SAM3 模型。"""
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

    print("=" * 72)
    print(EXPERIMENT_NAME)
    print("=" * 72)
    print(f"配置文件: {settings['config'] and '已读取' or '未读取，使用默认值'}")
    print(f"基础目录: {settings['base_dir']}")
    print(f"输出目录: {settings['output_dir']}")
    print(f"图像数量: {len(settings['test_images'])}")
    print(f"Patch 参数: size={settings['patch_size']}, stride={settings['stride']}")
    print(f"预计 patch 总数: {total_patches}")
    print(f"预计 SAM3 文本 prompt 调用数: {total_patches * len(prompt_class_ids)}")
    print("主动 prompt 类别:")
    for class_id in prompt_class_ids:
        print(f"  {class_id}: {class_info[class_id]['name']} -> {class_info[class_id]['prompt']}")
    print("不主动 prompt 类别:")
    for class_id in excluded_prompt_class_ids:
        print(f"  {class_id}: {class_info[class_id]['name']} -> {class_info[class_id]['prompt']}")

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


def print_header(settings, config_path):
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
    print("主动 prompt 类别:", settings["prompt_class_ids"])
    print("不主动 prompt 类别:", settings["excluded_prompt_class_ids"])
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Exp2-A: SAM3 Potsdam 零样本评估，不主动输入背景/杂波文本提示"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="实验一 YAML 配置文件路径；Exp2-A 会复用其中的数据、设备、切片和类别配置",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Exp2-A 输出目录；默认 {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查样本、路径和预计 patch/prompt 数量，不加载 SAM3 模型",
    )
    args = parser.parse_args()

    settings = load_runtime_settings(args)

    if args.dry_run:
        dry_run(settings)
        return

    print_header(settings, args.config)

    if not SAM3_AVAILABLE:
        print("错误: SAM3 模块不可用，请先安装 SAM3")
        return

    evaluator = Exp2ANoBackgroundPromptEvaluator(
        base_dir=settings["base_dir"],
        output_dir=settings["output_dir"],
        checkpoint_path=settings["checkpoint_path"],
        patch_size=settings["patch_size"],
        stride=settings["stride"],
        score_threshold=settings["score_threshold"],
        class_info=settings["class_info"],
        gpu_dtype=settings["gpu_dtype"],
        device_type=settings["device_type"],
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
