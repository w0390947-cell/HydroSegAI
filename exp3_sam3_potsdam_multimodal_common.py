#!/usr/bin/env python3

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from exp1_sam3_potsdam_zeroshot_baseline import (
    PotsdamSAM3Evaluator,
    SAM3_AVAILABLE,
    build_class_info,
    build_test_images,
    load_yaml_config,
    save_config_snapshot,
    save_run_context,
)
from exp2a_sam3_potsdam_no_background_prompt import (
    load_prompt_class_config,
    print_exp2_header,
    print_image_metrics,
    print_overall_metrics,
    print_prompt_ensemble_summary,
    print_prompt_policy_summary,
    raise_sam3_unavailable_error,
    run_exp2_evaluation,
)
from exp2b_sam3_potsdam_visual_prompt_ensemble import (
    load_prompt_variants,
)
from exp2c_sam3_potsdam_prompt_ensemble_mask_filter import (
    Exp2CPromptEnsembleMaskFilterEvaluator,
    load_mask_filter_config,
    print_filter_rules,
)

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


def build_exp3_test_images(config, settings):
    evaluation_config = config.get("evaluation", {})
    explicit_images = evaluation_config.get("test_images")
    if explicit_images:
        return explicit_images

    split = str(evaluation_config.get("split", "dev")).lower()
    if split in ("dev", "development", "participant", "participants"):
        return list(POTSDAM_DEV_IMAGE_IDS)
    if split in ("heldout", "held-out", "test", "official_test"):
        return list(POTSDAM_HELDOUT_IMAGE_IDS)
    if split in ("all", "full"):
        return build_test_images(
            config,
            base_dir=settings["base_dir"],
            image_subdir=settings["image_subdir"],
            label_subdir=settings["label_subdir"],
            image_pattern=settings["image_pattern"],
            label_pattern=settings["label_pattern"],
        )

    raise ValueError(
        "evaluation.split 必须是 dev、heldout 或 all；"
        f"实际为: {evaluation_config.get('split')}"
    )


def build_exp3_split_metadata(config, test_images):
    evaluation_config = config.get("evaluation", {})
    explicit_images = evaluation_config.get("test_images")
    split = str(evaluation_config.get("split", "dev")).lower()
    if explicit_images:
        split = "custom"
    if split in ("development", "participant", "participants"):
        split = "dev"
    elif split in ("held-out", "test", "official_test"):
        split = "heldout"
    elif split == "full":
        split = "all"

    return {
        "split": split,
        "split_description": (
            "24 official participant-labeled tiles for modality analysis"
            if split == "dev"
            else "14 held-out fully referenced tiles for final evaluation"
            if split == "heldout"
            else "explicit user-provided image list"
            if split == "custom"
            else "all paired fully referenced tiles"
        ),
        "num_split_images": len(test_images),
        "split_image_ids": list(test_images),
    }


def load_exp3_settings(args, default_output_dir):
    config = load_yaml_config(args.config)
    paths_config = config.get("paths", {})
    image_processing_config = config.get("image_processing", {})
    device_config = config.get("device", {})
    model_config = config.get("model", {})
    output_config = config.get("output", {})
    metrics_config = config.get("metrics", {})

    settings = {
        "config": config,
        "base_dir": paths_config.get("base_dir", "/home/anjou/PythonENV/Test_11/Potsdam"),
        "output_dir": args.output_dir or paths_config.get("output_dir", default_output_dir),
        "checkpoint_path": paths_config.get(
            "checkpoint_path",
            "/home/anjou/PythonENV/Test_11/sam3/checkpoints/sam3.1_multiplex.pt",
        ),
        "image_subdir": paths_config.get("image_subdir", "2_Ortho_RGB/2_Ortho_RGB"),
        "label_subdir": paths_config.get("label_subdir", "5_Labels_all_noBoundary"),
        "image_pattern": paths_config.get("image_pattern", "{image_id}_RGB.tif"),
        "label_pattern": paths_config.get("label_pattern", "{image_id}_label_noBoundary.tif"),
        "class_info": build_class_info(config),
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
        "input_modality": config.get("input_modality", {}),
        "multiview": config.get("multiview", {}),
    }
    (
        settings["prompt_class_ids"],
        settings["excluded_prompt_class_ids"],
    ) = load_prompt_class_config(config, settings["class_info"])
    settings["test_images"] = build_exp3_test_images(config, settings)
    settings["split_metadata"] = build_exp3_split_metadata(config, settings["test_images"])
    return settings


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


def view_image_path(base_dir, view_config, image_id):
    return (
        Path(base_dir)
        / view_config["image_subdir"]
        / view_config["image_pattern"].format(image_id=image_id)
    )


def read_rgbir_image(image_path):
    """Read RGBIR TIFF as HWC using GDAL, rasterio, or tifffile."""
    image_path = Path(image_path)

    try:
        from osgeo import gdal

        dataset = gdal.Open(str(image_path))
        if dataset is not None:
            image = dataset.ReadAsArray()
            dataset = None
            image = np.asarray(image)
            if image.ndim == 3 and image.shape[0] in (3, 4):
                image = np.transpose(image, (1, 2, 0))
            return image
    except ImportError:
        pass

    try:
        import rasterio

        with rasterio.open(image_path) as src:
            image = src.read()
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[0] in (3, 4):
            image = np.transpose(image, (1, 2, 0))
        return image
    except ImportError:
        pass

    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("读取 RGBIR 需要安装 GDAL、rasterio 或 tifffile") from exc

    try:
        image = tifffile.imread(image_path)
    except TypeError as exc:
        raise RuntimeError(
            "当前 tifffile/numpy 组合无法读取 Potsdam RGBIR 四波段 TIFF；"
            "请安装 rasterio/GDAL，或将 tifffile 降级到兼容 numpy 1.26 的版本。"
        ) from exc

    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.transpose(image, (1, 2, 0))
    return image


def rgbir_to_composite(rgbir_image, composite_mode):
    if rgbir_image.ndim != 3 or rgbir_image.shape[-1] < 4:
        raise ValueError(f"RGBIR 影像应为 HxWx4，实际为 {rgbir_image.shape}")

    r = rgbir_image[:, :, 0]
    g = rgbir_image[:, :, 1]
    b = rgbir_image[:, :, 2]
    nir = rgbir_image[:, :, 3]

    mode = str(composite_mode or "nir-r-g").lower()
    if mode == "rgb":
        channels = [r, g, b]
    elif mode == "nir-r-g":
        channels = [nir, r, g]
    elif mode == "nir-g-b":
        channels = [nir, g, b]
    elif mode == "r-g-ndvi":
        ndvi = compute_ndvi_uint8(nir, r)
        channels = [r, g, ndvi]
    elif mode == "ndvi-r-g":
        ndvi = compute_ndvi_uint8(nir, r)
        channels = [ndvi, r, g]
    else:
        raise ValueError(f"未知 RGBIR composite_mode={composite_mode}")

    return np.stack(channels, axis=-1).astype(np.uint8, copy=False)


def compute_ndvi_uint8(nir, red):
    nir_f = nir.astype(np.float32)
    red_f = red.astype(np.float32)
    ndvi = (nir_f - red_f) / np.maximum(nir_f + red_f, 1e-6)
    return np.clip((ndvi + 1.0) * 127.5, 0, 255).astype(np.uint8)


class Exp3SingleModalityEvaluator(Exp2CPromptEnsembleMaskFilterEvaluator):
    """Exp3 single-modality evaluator using the fixed Exp2-B prompt ensemble strategy."""

    def __init__(
        self,
        *args,
        experiment_name,
        input_modality,
        rgbir_composite_mode=None,
        split_metadata=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.experiment_name = experiment_name
        self.input_modality = input_modality
        self.rgbir_composite_mode = rgbir_composite_mode
        self.split_metadata = split_metadata or {}

    def read_image(self, image_path):
        if self.input_modality != "rgbir_composite":
            return super().read_image(image_path)

        image_path = Path(image_path)
        if "GDAL_AVAILABLE" in globals():
            pass

        # Prefer GDAL through the parent only if it is available there. The
        # parent converts HWC correctly and preserves 4 bands when using GDAL.
        try:
            image = super().read_image(image_path)
        except Exception:
            image = read_rgbir_image(image_path)

        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] < 4):
            image = read_rgbir_image(image_path)

        return rgbir_to_composite(image, self.rgbir_composite_mode)

    def save_overall_metrics(self, all_results, run_context=None):
        overall_metrics = PotsdamSAM3Evaluator.save_overall_metrics(
            self,
            all_results,
            run_context=run_context,
        )
        metadata = self._build_metadata()
        self._write_metadata(metadata)
        overall_metrics.update(metadata)
        return overall_metrics

    def _build_metadata(self):
        prompt_calls_per_patch = sum(
            len(self.prompt_variants[class_id]) for class_id in self.prompt_class_ids
        )
        return {
            "experiment_name": self.experiment_name,
            "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "variable_under_test": "input remote-sensing modality under fixed Exp2-B prompt ensemble strategy",
            "input_modality": self.input_modality,
            "rgbir_composite_mode": self.rgbir_composite_mode,
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
            "evaluation_split": self.split_metadata,
        }

    def _write_metadata(self, metadata):
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


class Exp3MultiViewFusionEvaluator(Exp3SingleModalityEvaluator):
    """Exp3-D evaluator: run SAM3 on multiple views and fuse patch scores."""

    def __init__(self, *args, view_configs, **kwargs):
        super().__init__(*args, input_modality="multiview_fusion", **kwargs)
        self.view_configs = view_configs

    def _read_view_image(self, view_config, image_id):
        image_path = view_image_path(self.base_dir, view_config, image_id)
        view_type = view_config.get("type", "rgb")
        if view_type == "rgbir_composite":
            image = read_rgbir_image(image_path)
            return rgbir_to_composite(image, view_config.get("composite_mode", "nir-r-g"))
        return super(Exp3SingleModalityEvaluator, self).read_image(image_path)

    def process_image_id(self, image_id, label_path):
        self.logger.info(f"正在处理多视图样本: {image_id}")

        label = self.read_label(label_path)
        image_stats = self._new_image_inference_stats(f"{image_id}_multiview")
        view_images = []
        for view_config in self.view_configs:
            image = self._read_view_image(view_config, image_id)
            if image.shape[:2] != label.shape:
                raise ValueError(
                    f"{image_id} 的视图 {view_config.get('name')} 尺寸 {image.shape[:2]} "
                    f"与标签尺寸 {label.shape} 不一致"
                )
            view_images.append((view_config, image))

        # Use RGB view for visualization when available; otherwise first view.
        visual_image = view_images[0][1]
        for view_config, image in view_images:
            if view_config.get("name") == "rgb":
                visual_image = image
                break

        predicted_patches = []
        patch_counter = 0
        for view_config, image in view_images:
            patches = self.slice_image(image)
            view_weight = float(view_config.get("weight", 1.0))
            self.logger.info(
                f"处理视图 {view_config.get('name')}，patches={len(patches)}, weight={view_weight}"
            )

            for i, (patch, x, y) in enumerate(patches):
                self.logger.info(
                    f"视图 {view_config.get('name')} patch {i + 1}/{len(patches)} at ({x}, {y})"
                )
                text_prompts = [
                    (class_id, self.class_info[class_id]["prompt"])
                    for class_id in self.class_ids
                ]
                patch_counter += 1
                predictions = self.predict_patch(
                    patch,
                    text_prompts,
                    image_stats=image_stats,
                    patch_index=patch_counter,
                    patch_position=(x, y),
                )
                fused_mask, confidence_map = self.fuse_multiclass_masks(
                    predictions,
                    patch.shape[:2],
                )
                patch_h = min(self.PATCH_SIZE, label.shape[0] - y)
                patch_w = min(self.PATCH_SIZE, label.shape[1] - x)
                fused_mask = fused_mask[:patch_h, :patch_w]
                confidence_map = confidence_map[:patch_h, :patch_w] * view_weight
                predicted_patches.append((fused_mask, confidence_map, x, y))

        self._validate_image_inference_health(image_stats)
        self._merge_image_inference_stats(image_stats)

        predicted_label = self.merge_patches(predicted_patches, label.shape)
        metrics = self.calculate_metrics(predicted_label, label)
        self.save_results(
            f"{image_id}_multiview",
            visual_image,
            label,
            predicted_label,
            metrics,
            inference_stats=image_stats,
        )

        return {
            "image_name": f"{image_id}_multiview",
            "metrics": metrics,
            "inference_stats": image_stats,
        }

    def _build_metadata(self):
        metadata = super()._build_metadata()
        metadata["view_configs"] = self.view_configs
        metadata["variable_under_test"] = "multi-view SAM3 inference and score fusion"
        return metadata


def print_exp3_header(settings, config_path, experiment_name, prompt_variants, mask_filter_config):
    def print_exp3_variables():
        print_prompt_policy_summary(settings)
        print_prompt_ensemble_summary(settings, prompt_variants)
        print_filter_rules(settings["class_info"], mask_filter_config, settings["score_threshold"])
        modality_config = settings.get("input_modality", {})
        split_metadata = settings.get("split_metadata", {})
        print(
            f"评估划分: {split_metadata.get('split', 'unknown')} "
            f"({split_metadata.get('num_split_images', len(settings['test_images']))} images)"
        )
        print(f"输入模态: {modality_config.get('name', 'rgb')}")
        if modality_config.get("rgbir_composite_mode"):
            print(f"RGBIR composite mode: {modality_config['rgbir_composite_mode']}")

        view_configs = settings.get("multiview", {}).get("views", [])
        if view_configs:
            print(f"多视图融合: {settings.get('multiview', {}).get('fusion', 'confidence_sum')}")
            for view_config in view_configs:
                print(
                    f"  view={view_config.get('name')} "
                    f"type={view_config.get('type')} "
                    f"weight={view_config.get('weight', 1.0)}"
                )

    print_exp2_header(
        settings,
        config_path,
        experiment_name,
        extra_lines_callback=print_exp3_variables,
    )


def run_exp3_multiview_evaluation(settings, evaluator, config_path):
    all_results = []
    expected_images = list(settings["test_images"])
    processed_images = []
    skipped_images = []
    failed_images = []
    view_configs = settings["multiview"].get("views", [])

    for image_id in settings["test_images"]:
        _, label_path = image_and_label_paths(settings, image_id)

        if not label_path.exists():
            print(f"错误: 标签文件不存在 - {label_path}")
            skipped_images.append({
                "image_id": image_id,
                "reason": "missing_label_file",
                "label_path": str(label_path),
            })
            continue

        missing_views = []
        for view_config in view_configs:
            image_path = view_image_path(settings["base_dir"], view_config, image_id)
            if not image_path.exists():
                missing_views.append({
                    "view": view_config.get("name"),
                    "image_path": str(image_path),
                })

        if missing_views:
            print(f"错误: 多视图样本缺失输入影像 - {image_id}")
            skipped_images.append({
                "image_id": image_id,
                "reason": "missing_view_image_file",
                "label_path": str(label_path),
                "missing_views": missing_views,
            })
            continue

        try:
            result = evaluator.process_image_id(image_id, label_path)
            all_results.append(result)
            processed_images.append(result["image_name"])
            print_image_metrics(result, evaluator.class_info)
        except Exception as exc:
            print(f"处理 {image_id} 时出错: {exc}")
            failed_images.append({
                "image_id": image_id,
                "reason": str(exc),
                "exception_type": type(exc).__name__,
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
        "evaluation_split": settings.get("split_metadata", {}),
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


def run_single_modality_experiment(experiment_name, default_config_path, default_output_dir):
    parser = argparse.ArgumentParser(description=experiment_name)
    parser.add_argument("--config", default=default_config_path, help="YAML 配置文件路径")
    parser.add_argument("--output-dir", default=None, help=f"输出目录；默认 {default_output_dir}")
    args = parser.parse_args()

    settings = load_exp3_settings(args, default_output_dir)
    prompt_variants = load_prompt_variants(
        settings["config"],
        settings["class_info"],
        settings["prompt_class_ids"],
    )
    mask_filter_config = load_mask_filter_config(settings["config"])
    mask_filter_config["prompt_class_ids"] = settings["prompt_class_ids"]

    print_exp3_header(settings, args.config, experiment_name, prompt_variants, mask_filter_config)

    if not SAM3_AVAILABLE:
        raise_sam3_unavailable_error(experiment_name)

    modality_config = settings.get("input_modality", {})
    evaluator = Exp3SingleModalityEvaluator(
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
        experiment_name=experiment_name,
        input_modality=modality_config.get("name", "rgb"),
        rgbir_composite_mode=modality_config.get("rgbir_composite_mode"),
        split_metadata=settings["split_metadata"],
    )

    return run_exp2_evaluation(settings, evaluator, args.config)


def run_multiview_experiment(experiment_name, default_config_path, default_output_dir):
    parser = argparse.ArgumentParser(description=experiment_name)
    parser.add_argument("--config", default=default_config_path, help="YAML 配置文件路径")
    parser.add_argument("--output-dir", default=None, help=f"输出目录；默认 {default_output_dir}")
    args = parser.parse_args()

    settings = load_exp3_settings(args, default_output_dir)
    prompt_variants = load_prompt_variants(
        settings["config"],
        settings["class_info"],
        settings["prompt_class_ids"],
    )
    mask_filter_config = load_mask_filter_config(settings["config"])
    mask_filter_config["prompt_class_ids"] = settings["prompt_class_ids"]
    view_configs = settings["multiview"].get("views", [])
    if not view_configs:
        raise ValueError("Exp3-D 配置缺少 multiview.views")

    print_exp3_header(settings, args.config, experiment_name, prompt_variants, mask_filter_config)

    if not SAM3_AVAILABLE:
        raise_sam3_unavailable_error(experiment_name)

    evaluator = Exp3MultiViewFusionEvaluator(
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
        experiment_name=experiment_name,
        rgbir_composite_mode=None,
        split_metadata=settings["split_metadata"],
        view_configs=view_configs,
    )

    return run_exp3_multiview_evaluation(settings, evaluator, args.config)
