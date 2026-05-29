#!/usr/bin/env python3

import argparse

import numpy as np

from exp1_sam3_potsdam_zeroshot_baseline import SAM3_AVAILABLE
from exp2a_sam3_potsdam_no_background_prompt import (
    PROMPT_CLASS_IDS,
    load_runtime_settings,
    print_exp2_header,
    print_prompt_ensemble_summary,
    print_prompt_policy_summary,
    raise_sam3_unavailable_error,
    run_exp2_evaluation,
)
from exp2b_sam3_potsdam_visual_prompt_ensemble import (
    Exp2BVisualPromptEnsembleEvaluator,
    load_prompt_variants,
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

    def predict_patch(self, patch, text_prompts, image_stats=None, patch_index=None, patch_position=None):
        predictions = super().predict_patch(
            patch,
            text_prompts,
            image_stats=image_stats,
            patch_index=patch_index,
            patch_position=patch_position,
        )
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

    def build_experiment_metadata(self):
        metadata = self.build_common_metadata()
        prompt_calls_per_patch = sum(
            len(self.prompt_variants[class_id]) for class_id in self.prompt_class_ids
        )
        metadata.update({
            "experiment_name": EXPERIMENT_NAME,
            "variable_under_test": (
                "visual prompt ensemble with class-aware mask area filtering"
            ),
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
        })
        return metadata


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


def print_header(settings, prompt_variants, mask_filter_config, config_path):
    def print_exp2c_variables():
        print_prompt_policy_summary(settings)
        print_prompt_ensemble_summary(settings, prompt_variants)
        print_filter_rules(settings["class_info"], mask_filter_config, settings["score_threshold"])

    print_exp2_header(
        settings,
        config_path,
        EXPERIMENT_NAME,
        extra_lines_callback=print_exp2c_variables,
    )


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

    print_header(settings, prompt_variants, mask_filter_config, args.config)

    if not SAM3_AVAILABLE:
        raise_sam3_unavailable_error(EXPERIMENT_NAME)

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
        allow_hf_fallback=settings["allow_hf_fallback"],
        output_config=settings["output_config"],
        metrics_config=settings["metrics_config"],
        prompt_variants=prompt_variants,
        mask_filter_config=mask_filter_config,
        prompt_class_ids=settings["prompt_class_ids"],
        excluded_prompt_class_ids=settings["excluded_prompt_class_ids"],
    )

    run_exp2_evaluation(settings, evaluator, args.config)


if __name__ == "__main__":
    main()
