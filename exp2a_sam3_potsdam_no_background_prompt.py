#!/usr/bin/env python3

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

from exp1_sam3_potsdam_zeroshot_baseline import (
    PotsdamSAM3Evaluator,
    SAM3_AVAILABLE,
    SAM3_IMPORT_ERROR,
    build_class_info,
    build_test_images,
    load_yaml_config,
    save_config_snapshot,
    save_run_context,
)


EXPERIMENT_NAME = "Exp2-A: SAM3 Potsdam no background prompt"
PROMPT_CLASS_IDS = [0, 1, 2, 3, 4]
EXCLUDED_PROMPT_CLASS_IDS = [5]
DEFAULT_CONFIG_PATH = "exp2a_sam3_potsdam_no_background_prompt.yaml"
DEFAULT_OUTPUT_DIR = "/home/anjou/PythonENV/Test_11/results_exp2a_sam3_potsdam_no_background_prompt"


def raise_sam3_unavailable_error(experiment_name):
    project_sam3_dir = Path.cwd() / "sam3"
    sam3_path_entries = [path for path in sys.path if "sam3" in path.lower()]
    raise RuntimeError(
        f"{experiment_name} 无法执行：SAM3 模块不可用。"
        f"import_error={SAM3_IMPORT_ERROR!r}; "
        f"python_executable={sys.executable}; "
        f"cwd={Path.cwd()}; "
        f"project_sam3_dir_exists={project_sam3_dir.exists()}; "
        f"sam3_entries_in_sys_path={sam3_path_entries}"
    )


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
    prompt_policy_config = config.get("prompt_policy", {})
    prompt_class_ids = _read_class_id_list(
        prompt_policy_config.get("prompt_class_ids"),
        PROMPT_CLASS_IDS,
        "prompt_policy.prompt_class_ids",
    )
    excluded_prompt_class_ids = _read_class_id_list(
        prompt_policy_config.get("excluded_prompt_class_ids"),
        EXCLUDED_PROMPT_CLASS_IDS,
        "prompt_policy.excluded_prompt_class_ids",
    )

    if not prompt_class_ids:
        raise ValueError("prompt_policy.prompt_class_ids 不能为空")

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
            "prompt_policy.prompt_class_ids 和 "
            "prompt_policy.excluded_prompt_class_ids 必须共同覆盖所有评估类别；"
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

    def predict_patch(self, patch, text_prompts, image_stats=None, patch_index=None, patch_position=None):
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

        return super().predict_patch(
            patch,
            filtered_prompts,
            image_stats=image_stats,
            patch_index=patch_index,
            patch_position=patch_position,
        )

    def build_common_metadata(self):
        return {
            "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        }

    def build_experiment_metadata(self):
        metadata = self.build_common_metadata()
        metadata.update({
            "experiment_name": EXPERIMENT_NAME,
            "variable_under_test": "remove active clutter/background text prompt",
            "note": (
                "Class 5 remains an evaluation class and fallback class, but SAM3 "
                "does not receive its text prompt in this experiment."
            ),
        })
        return metadata

    def save_experiment_metadata(self, metadata):
        metadata_path = self.output_dir / "metrics" / "experiment_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def save_overall_metrics_with_metadata(self, overall_metrics):
        overall_path = self.output_dir / "metrics" / "overall_metrics.json"
        with open(overall_path, "w", encoding="utf-8") as f:
            json.dump(overall_metrics, f, indent=2, ensure_ascii=False)

    def save_overall_metrics(self, all_results, run_context=None):
        """保存总体指标，并补充 Exp2 系列实验元数据。"""
        overall_metrics = super().save_overall_metrics(all_results, run_context=run_context)
        metadata = self.build_experiment_metadata()
        self.save_experiment_metadata(metadata)
        overall_metrics.update(metadata)
        self.save_overall_metrics_with_metadata(overall_metrics)

        return overall_metrics


def load_runtime_settings(args):
    config = load_yaml_config(args.config)
    paths_config = config.get("paths", {})
    image_processing_config = config.get("image_processing", {})
    device_config = config.get("device", {})
    model_config = config.get("model", {})
    output_config = config.get("output", {})
    metrics_config = config.get("metrics", {})

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
    test_images = build_test_images(
        config,
        base_dir=base_dir,
        image_subdir=image_subdir,
        label_subdir=label_subdir,
        image_pattern=image_pattern,
        label_pattern=label_pattern,
    )

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
        "allow_hf_fallback": model_config.get("allow_hf_fallback", False),
        "output_config": output_config,
        "metrics_config": metrics_config,
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


def resolve_fallback_class_id(class_info):
    """Exp2 继承 Exp1 约定：未覆盖区域回退到 clutter/background。"""
    for class_id, info in class_info.items():
        if info.get("name") == "clutter/background":
            return class_id
    return max(class_info.keys())


def print_prompt_policy_summary(settings):
    class_info = settings["class_info"]

    print("主动 prompt 类别:")
    for class_id in settings["prompt_class_ids"]:
        info = class_info[class_id]
        print(f"  {class_id}: {info['name']} -> {info['prompt']}")

    print("不主动 prompt 类别:")
    for class_id in settings["excluded_prompt_class_ids"]:
        info = class_info[class_id]
        print(f"  {class_id}: {info['name']} -> {info['prompt']}")

    fallback_class_id = resolve_fallback_class_id(class_info)
    fallback_info = class_info[fallback_class_id]
    print(
        "fallback 类别: "
        f"{fallback_class_id}: {fallback_info['name']} "
        "(未被有效正置信度 mask 覆盖的像素)"
    )


def print_prompt_ensemble_summary(settings, prompt_variants):
    prompt_calls_per_patch = sum(
        len(prompt_variants[class_id]) for class_id in settings["prompt_class_ids"]
    )
    print(f"每个 patch 文本 prompt 调用数: {prompt_calls_per_patch}")
    print("prompt ensemble:")
    for class_id in settings["prompt_class_ids"]:
        info = settings["class_info"][class_id]
        print(f"  {class_id}: {info['name']}")
        for prompt in prompt_variants[class_id]:
            print(f"    - {prompt}")


def print_exp2_header(settings, config_path, experiment_name, extra_lines_callback=None):
    output_config = settings.get("output_config", {})
    print("=" * 72)
    print(experiment_name)
    print("=" * 72)
    print(f"基础目录: {settings['base_dir']}")
    print(f"输出目录: {settings['output_dir']}")
    print(f"配置文件: {config_path if settings['config'] else '未使用配置文件'}")
    print(f"测试图像数量: {len(settings['test_images'])}")
    print(f"Patch 参数: size={settings['patch_size']}, stride={settings['stride']}, score_threshold={settings['score_threshold']}")
    print("融合策略: confidence (继承 Exp1)")
    print(f"设备配置: {settings['device_type']}")
    print(f"GPU dtype: {settings['gpu_dtype']}")
    print(f"允许 HuggingFace fallback: {settings['allow_hf_fallback']}")
    print(
        "输出配置: "
        f"save_predictions={output_config.get('save_predictions', True)}, "
        f"save_visualizations={output_config.get('save_visualizations', True)}, "
        f"image_format={output_config.get('image_format', 'png')}"
    )
    print(f"当前 torch CUDA: {torch.cuda.is_available()}")
    if extra_lines_callback is not None:
        extra_lines_callback()
    print("=" * 72)


def print_header(settings, config_path):
    print_exp2_header(
        settings,
        config_path,
        EXPERIMENT_NAME,
        extra_lines_callback=lambda: print_prompt_policy_summary(settings),
    )


def print_image_metrics(result, class_info):
    print(f"\n{result['image_name']} 评估结果:")
    print(f"  Overall Accuracy: {result['metrics']['overall_accuracy']:.4f}")
    print(f"  Mean IoU: {result['metrics']['mean_iou']:.4f}")
    print(f"  Mean F1: {result['metrics']['mean_f1']:.4f}")
    print(f"  Mean Precision: {result['metrics']['mean_precision']:.4f}")
    print(f"  Mean Recall: {result['metrics']['mean_recall']:.4f}")
    print(f"  Frequency Weighted IoU: {result['metrics']['frequency_weighted_iou']:.4f}")

    for class_index, (class_id, info) in enumerate(class_info.items()):
        iou = result["metrics"]["iou_per_class"][class_index]
        print(f"  IoU ({info['name']}): {iou:.4f}")


def print_overall_metrics(overall_metrics, class_info):
    print("\n总体评估结果:")
    print(f"  数据集 Overall Accuracy: {overall_metrics['dataset_overall_accuracy']:.4f}")
    print(f"  数据集 Mean IoU: {overall_metrics['dataset_mean_iou']:.4f}")
    print(f"  数据集 Mean F1: {overall_metrics['dataset_mean_f1']:.4f}")
    print(f"  数据集 Mean Precision: {overall_metrics['dataset_mean_precision']:.4f}")
    print(f"  数据集 Mean Recall: {overall_metrics['dataset_mean_recall']:.4f}")
    print(
        "  数据集 Frequency Weighted IoU: "
        f"{overall_metrics['dataset_frequency_weighted_iou']:.4f}"
    )
    print(f"  逐图平均 Overall Accuracy: {overall_metrics['average_overall_accuracy']:.4f}")
    print(f"  逐图平均 Mean IoU: {overall_metrics['average_mean_iou']:.4f}")
    print(f"  逐图平均 Mean F1: {overall_metrics['average_mean_f1']:.4f}")
    print(f"  逐图平均 Mean Precision: {overall_metrics['average_mean_precision']:.4f}")
    print(f"  逐图平均 Mean Recall: {overall_metrics['average_mean_recall']:.4f}")
    print(
        "  逐图平均 Frequency Weighted IoU: "
        f"{overall_metrics['average_frequency_weighted_iou']:.4f}"
    )

    print("\n各类别数据集 IoU:")
    for class_index, (class_id, info) in enumerate(class_info.items()):
        dataset_iou = overall_metrics["dataset_iou_per_class"][class_index]
        print(f"  {info['name']}: {dataset_iou:.4f}")


def run_exp2_evaluation(settings, evaluator, config_path):
    all_results = []
    expected_images = list(settings["test_images"])
    processed_images = []
    skipped_images = []
    failed_images = []

    for image_id in settings["test_images"]:
        image_path, label_path = image_and_label_paths(settings, image_id)

        if not image_path.exists():
            print(f"错误: 图像文件不存在 - {image_path}")
            skipped_images.append({
                "image_id": image_id,
                "reason": "missing_image_file",
                "image_path": str(image_path),
                "label_path": str(label_path),
            })
            continue
        if not label_path.exists():
            print(f"错误: 标签文件不存在 - {label_path}")
            skipped_images.append({
                "image_id": image_id,
                "reason": "missing_label_file",
                "image_path": str(image_path),
                "label_path": str(label_path),
            })
            continue

        try:
            result = evaluator.process_image(image_path, label_path)
            all_results.append(result)
            processed_images.append(result["image_name"])

            print_image_metrics(result, evaluator.class_info)

        except Exception as exc:
            print(f"处理 {image_id} 时出错: {exc}")
            failed_images.append({
                "image_id": image_id,
                "reason": str(exc),
                "exception_type": type(exc).__name__,
                "image_path": str(image_path),
                "label_path": str(label_path),
            })
            import traceback

            traceback.print_exc()
            continue

    config_snapshot_path = save_config_snapshot(config_path, settings["output_dir"])
    run_context = {
        "config_path": str(config_path) if config_path else None,
        "config_snapshot_path": config_snapshot_path,
        "expected_images": expected_images,
        "processed_images": processed_images,
        "skipped_images": skipped_images,
        "failed_images": failed_images,
        "num_expected_images": len(expected_images),
        "num_processed_images": len(processed_images),
        "num_skipped_images": len(skipped_images),
        "num_failed_images": len(failed_images),
        "metrics_scope": "successful_images_only",
    }
    run_context_path = save_run_context(settings["output_dir"], run_context)

    if all_results:
        print("\n" + "=" * 72)
        print("保存总体评估指标...")
        overall = evaluator.save_overall_metrics(all_results, run_context=run_context)

        if skipped_images or failed_images:
            print(
                "警告: 本次评估存在未纳入总体指标的样本。"
                f"期望 {len(expected_images)} 张，成功 {len(processed_images)} 张，"
                f"跳过 {len(skipped_images)} 张，失败 {len(failed_images)} 张。"
                "总体指标仅基于成功样本。"
            )

        print_overall_metrics(overall, evaluator.class_info)

        print(f"\n评估完成！结果保存在: {settings['output_dir']}")
        return overall

    raise RuntimeError(
        "没有成功处理任何图像；不会生成总体指标。"
        f"样本处理状态已保存到 {run_context_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Exp2-A: SAM3 Potsdam prompt 消融实验，不主动输入背景/杂波文本提示"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Exp2-A YAML 配置文件路径",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Exp2-A 输出目录；默认 {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()

    settings = load_runtime_settings(args)

    print_header(settings, args.config)

    if not SAM3_AVAILABLE:
        raise_sam3_unavailable_error(EXPERIMENT_NAME)

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
        allow_hf_fallback=settings["allow_hf_fallback"],
        output_config=settings["output_config"],
        metrics_config=settings["metrics_config"],
        prompt_class_ids=settings["prompt_class_ids"],
        excluded_prompt_class_ids=settings["excluded_prompt_class_ids"],
    )

    run_exp2_evaluation(settings, evaluator, args.config)


if __name__ == "__main__":
    main()
