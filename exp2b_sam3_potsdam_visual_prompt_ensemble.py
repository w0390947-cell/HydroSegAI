#!/usr/bin/env python3

import argparse

from exp1_sam3_potsdam_zeroshot_baseline import (
    PotsdamSAM3Evaluator,
    SAM3_AVAILABLE,
)
from exp2a_sam3_potsdam_no_background_prompt import (
    Exp2ANoBackgroundPromptEvaluator,
    load_prompt_class_config,
    load_runtime_settings,
    print_exp2_header,
    print_prompt_ensemble_summary,
    print_prompt_policy_summary,
    raise_sam3_unavailable_error,
    run_exp2_evaluation,
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

    def predict_patch(self, patch, text_prompts, image_stats=None, patch_index=None, patch_position=None):
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

        return PotsdamSAM3Evaluator.predict_patch(
            self,
            patch,
            expanded_prompts,
            image_stats=image_stats,
            patch_index=patch_index,
            patch_position=patch_position,
        )

    def build_experiment_metadata(self):
        metadata = self.build_common_metadata()
        prompt_calls_per_patch = sum(
            len(self.prompt_variants[class_id]) for class_id in self.prompt_class_ids
        )
        metadata.update({
            "experiment_name": EXPERIMENT_NAME,
            "variable_under_test": "visual prompt ensemble without active clutter/background prompt",
            "prompt_variants": self.prompt_variants,
            "prompt_calls_per_patch": prompt_calls_per_patch,
            "note": (
                "Each foreground class is queried with multiple visual prompts. "
                "All masks from variants of the same class are mapped back to that class."
            ),
        })
        return metadata


def print_header(settings, prompt_variants, config_path):
    def print_exp2b_variables():
        print_prompt_policy_summary(settings)
        print_prompt_ensemble_summary(settings, prompt_variants)

    print_exp2_header(
        settings,
        config_path,
        EXPERIMENT_NAME,
        extra_lines_callback=print_exp2b_variables,
    )


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
    args = parser.parse_args()

    settings = load_runtime_settings(args)
    if not args.output_dir and "paths" not in settings["config"]:
        settings["output_dir"] = DEFAULT_OUTPUT_DIR

    prompt_variants = load_prompt_variants(
        settings["config"],
        settings["class_info"],
        settings["prompt_class_ids"],
    )

    print_header(settings, prompt_variants, args.config)

    if not SAM3_AVAILABLE:
        raise_sam3_unavailable_error(EXPERIMENT_NAME)

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
        allow_hf_fallback=settings["allow_hf_fallback"],
        output_config=settings["output_config"],
        metrics_config=settings["metrics_config"],
        prompt_variants=prompt_variants,
        prompt_class_ids=settings["prompt_class_ids"],
        excluded_prompt_class_ids=settings["excluded_prompt_class_ids"],
    )

    run_exp2_evaluation(settings, evaluator, args.config)


if __name__ == "__main__":
    main()
