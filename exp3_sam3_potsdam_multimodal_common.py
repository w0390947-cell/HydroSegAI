#!/usr/bin/env python3
"""Shared utilities for Exp3 Potsdam multi-modal SAM3 experiments."""

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
    build_class_info,
    load_yaml_config,
)
from exp2a_sam3_potsdam_no_background_prompt import (
    count_patches_with_exp1_logic,
    load_prompt_class_config,
)
from exp2b_sam3_potsdam_visual_prompt_ensemble import (
    load_prompt_variants,
    print_prompt_variants,
)
from exp2c_sam3_potsdam_prompt_ensemble_mask_filter import (
    Exp2CPromptEnsembleMaskFilterEvaluator,
    load_mask_filter_config,
    print_filter_rules,
)


def _potsdam_image_id_sort_key(image_id):
    return [int(part) if part.isdigit() else part for part in image_id.split("_")]


def _pattern_parts(pattern):
    if "{image_id}" not in pattern:
        raise ValueError(f"文件命名模式必须包含 {{image_id}}: {pattern}")
    prefix, suffix = pattern.split("{image_id}", 1)
    return prefix, suffix


def discover_paired_image_ids_by_pattern(
    base_dir,
    image_subdir,
    label_subdir,
    image_pattern,
    label_pattern,
):
    image_dir = Path(base_dir) / image_subdir
    label_dir = Path(base_dir) / label_subdir
    if not image_dir.exists() or not label_dir.exists():
        return []

    image_prefix, image_suffix = _pattern_parts(image_pattern)
    label_prefix, label_suffix = _pattern_parts(label_pattern)

    image_ids = set()
    for path in image_dir.glob(f"{image_prefix}*{image_suffix}"):
        name = path.name
        image_ids.add(name[len(image_prefix): len(name) - len(image_suffix)])

    label_ids = set()
    for path in label_dir.glob(f"{label_prefix}*{label_suffix}"):
        name = path.name
        label_ids.add(name[len(label_prefix): len(name) - len(label_suffix)])

    return sorted(image_ids & label_ids, key=_potsdam_image_id_sort_key)


def build_exp3_test_images(config, settings):
    evaluation_config = config.get("evaluation", {})
    explicit_images = evaluation_config.get("test_images")
    if explicit_images:
        return explicit_images

    mode = evaluation_config.get("mode", "quick")
    if evaluation_config.get("discover_from_files") or mode in ("full", "all", "auto"):
        discovered = discover_paired_image_ids_by_pattern(
            settings["base_dir"],
            settings["image_subdir"],
            settings["label_subdir"],
            settings["image_pattern"],
            settings["label_pattern"],
        )
        if discovered:
            return discovered

    if mode == "full":
        range_config = evaluation_config.get("full_test_range", {})
        rows = range_config.get("rows", [])
        cols = range_config.get("cols", [])
        return [f"top_potsdam_{row}_{col}" for row in rows for col in cols]

    return evaluation_config.get("quick_test_images", [
        "top_potsdam_2_10",
        "top_potsdam_5_11",
        "top_potsdam_7_9",
    ])


def load_exp3_settings(args, default_output_dir):
    config = load_yaml_config(args.config)
    paths_config = config.get("paths", {})
    image_processing_config = config.get("image_processing", {})
    device_config = config.get("device", {})

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
        "input_modality": config.get("input_modality", {}),
        "multiview": config.get("multiview", {}),
    }
    (
        settings["prompt_class_ids"],
        settings["excluded_prompt_class_ids"],
    ) = load_prompt_class_config(config, settings["class_info"])
    settings["test_images"] = build_exp3_test_images(config, settings)
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
    """Exp3 single-modality evaluator using Exp2-C reasoning strategy."""

    def __init__(
        self,
        *args,
        experiment_name,
        input_modality,
        rgbir_composite_mode=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.experiment_name = experiment_name
        self.input_modality = input_modality
        self.rgbir_composite_mode = rgbir_composite_mode

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

    def save_overall_metrics(self, all_results):
        PotsdamSAM3Evaluator.save_overall_metrics(self, all_results)
        metadata = self._build_metadata()
        self._write_metadata(metadata)

    def _build_metadata(self):
        prompt_calls_per_patch = sum(
            len(self.prompt_variants[class_id]) for class_id in self.prompt_class_ids
        )
        return {
            "experiment_name": self.experiment_name,
            "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "variable_under_test": "input remote-sensing modality under fixed Exp2-C reasoning strategy",
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
                predictions = self.predict_patch(patch, text_prompts)
                fused_mask, confidence_map = self.fuse_multiclass_masks(
                    predictions,
                    patch.shape[:2],
                )
                patch_h = min(self.PATCH_SIZE, label.shape[0] - y)
                patch_w = min(self.PATCH_SIZE, label.shape[1] - x)
                fused_mask = fused_mask[:patch_h, :patch_w]
                confidence_map = confidence_map[:patch_h, :patch_w] * view_weight
                predicted_patches.append((fused_mask, confidence_map, x, y))

        predicted_label = self.merge_patches(predicted_patches, label.shape)
        metrics = self.calculate_metrics(predicted_label, label)
        self.save_results(
            f"{image_id}_multiview",
            visual_image,
            label,
            predicted_label,
            metrics,
        )

        return {
            "image_name": f"{image_id}_multiview",
            "metrics": metrics,
        }

    def _build_metadata(self):
        metadata = super()._build_metadata()
        metadata["view_configs"] = self.view_configs
        metadata["variable_under_test"] = "multi-view SAM3 inference and score fusion"
        return metadata


def dry_run_single(settings, experiment_name, prompt_variants, mask_filter_config):
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
        with Image.open(label_path) as label:
            label_w, label_h = label.size
        # Read dimensions from label; Potsdam orthophotos are coregistered.
        image_w, image_h = label_w, label_h
        total_patches += count_patches_with_exp1_logic(
            image_w,
            image_h,
            settings["patch_size"],
            settings["stride"],
        )

    prompt_calls_per_patch = sum(len(prompt_variants[class_id]) for class_id in prompt_class_ids)
    print("=" * 72)
    print(experiment_name)
    print("=" * 72)
    print(f"基础目录: {settings['base_dir']}")
    print(f"输出目录: {settings['output_dir']}")
    print(f"输入影像目录: {settings['image_subdir']}")
    print(f"输入影像模式: {settings['image_pattern']}")
    print(f"输入模态: {settings['input_modality']}")
    print(f"图像数量: {len(settings['test_images'])}")
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
    print("文件检查:", "通过" if not missing_files else f"缺失 {len(missing_files)} 个")
    print("尺寸检查:", "通过" if not size_mismatches else f"异常 {len(size_mismatches)} 个")
    print("=" * 72)


def dry_run_multiview(settings, experiment_name, prompt_variants, mask_filter_config):
    view_configs = settings["multiview"].get("views", [])
    prompt_class_ids = settings["prompt_class_ids"]
    excluded_prompt_class_ids = settings["excluded_prompt_class_ids"]
    total_patches = 0
    missing_files = []
    for image_id in settings["test_images"]:
        _, label_path = image_and_label_paths(settings, image_id)
        if not label_path.exists():
            missing_files.append((image_id, str(label_path)))
            continue
        for view_config in view_configs:
            image_path = view_image_path(settings["base_dir"], view_config, image_id)
            if not image_path.exists():
                missing_files.append((image_id, str(image_path)))
        with Image.open(label_path) as label:
            image_w, image_h = label.size
        total_patches += count_patches_with_exp1_logic(
            image_w,
            image_h,
            settings["patch_size"],
            settings["stride"],
        )

    prompt_calls_per_patch = sum(len(prompt_variants[class_id]) for class_id in prompt_class_ids)
    print("=" * 72)
    print(experiment_name)
    print("=" * 72)
    print(f"基础目录: {settings['base_dir']}")
    print(f"输出目录: {settings['output_dir']}")
    print(f"图像数量: {len(settings['test_images'])}")
    print(f"视图数量: {len(view_configs)}")
    for view_config in view_configs:
        print(
            f"  {view_config.get('name')}: {view_config.get('type')} "
            f"weight={view_config.get('weight', 1.0)}"
        )
    print(f"预计单视图 patch 总数: {total_patches}")
    print(f"预计多视图 patch 总数: {total_patches * len(view_configs)}")
    print(f"每个 patch 文本 prompt 调用数: {prompt_calls_per_patch}")
    print(
        "预计 SAM3 文本 prompt 调用数: "
        f"{total_patches * len(view_configs) * prompt_calls_per_patch}"
    )
    print_prompt_variants(
        settings["class_info"],
        prompt_variants,
        prompt_class_ids,
        excluded_prompt_class_ids,
    )
    print_filter_rules(settings["class_info"], mask_filter_config, settings["score_threshold"])
    print("文件检查:", "通过" if not missing_files else f"缺失 {len(missing_files)} 个")
    print("=" * 72)


def run_single_modality_experiment(experiment_name, default_config_path, default_output_dir):
    parser = argparse.ArgumentParser(description=experiment_name)
    parser.add_argument("--config", default=default_config_path, help="YAML 配置文件路径")
    parser.add_argument("--output-dir", default=None, help=f"输出目录；默认 {default_output_dir}")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置，不加载 SAM3")
    args = parser.parse_args()

    settings = load_exp3_settings(args, default_output_dir)
    prompt_variants = load_prompt_variants(
        settings["config"],
        settings["class_info"],
        settings["prompt_class_ids"],
    )
    mask_filter_config = load_mask_filter_config(settings["config"])
    mask_filter_config["prompt_class_ids"] = settings["prompt_class_ids"]

    if args.dry_run:
        dry_run_single(settings, experiment_name, prompt_variants, mask_filter_config)
        return

    if not SAM3_AVAILABLE:
        print("错误: SAM3 模块不可用，请先安装 SAM3")
        return

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
        prompt_variants=prompt_variants,
        mask_filter_config=mask_filter_config,
        prompt_class_ids=settings["prompt_class_ids"],
        excluded_prompt_class_ids=settings["excluded_prompt_class_ids"],
        experiment_name=experiment_name,
        input_modality=modality_config.get("name", "rgb"),
        rgbir_composite_mode=modality_config.get("rgbir_composite_mode"),
    )

    all_results = []
    for image_id in settings["test_images"]:
        image_path, label_path = image_and_label_paths(settings, image_id)
        if not image_path.exists() or not label_path.exists():
            print(f"跳过缺失样本: {image_id}")
            continue
        try:
            result = evaluator.process_image(image_path, label_path)
            all_results.append(result)
            print(f"\n{result['image_name']} Mean IoU: {result['metrics']['mean_iou']:.4f}")
        except Exception as exc:
            print(f"处理 {image_id} 时出错: {exc}")
            import traceback
            traceback.print_exc()

    if all_results:
        evaluator.save_overall_metrics(all_results)
        print(f"\n评估完成！结果保存在: {settings['output_dir']}")
    else:
        print("没有成功处理任何图像。")


def run_multiview_experiment(experiment_name, default_config_path, default_output_dir):
    parser = argparse.ArgumentParser(description=experiment_name)
    parser.add_argument("--config", default=default_config_path, help="YAML 配置文件路径")
    parser.add_argument("--output-dir", default=None, help=f"输出目录；默认 {default_output_dir}")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置，不加载 SAM3")
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

    if args.dry_run:
        dry_run_multiview(settings, experiment_name, prompt_variants, mask_filter_config)
        return

    if not SAM3_AVAILABLE:
        print("错误: SAM3 模块不可用，请先安装 SAM3")
        return

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
        prompt_variants=prompt_variants,
        mask_filter_config=mask_filter_config,
        prompt_class_ids=settings["prompt_class_ids"],
        excluded_prompt_class_ids=settings["excluded_prompt_class_ids"],
        experiment_name=experiment_name,
        rgbir_composite_mode=None,
        view_configs=view_configs,
    )

    all_results = []
    for image_id in settings["test_images"]:
        _, label_path = image_and_label_paths(settings, image_id)
        if not label_path.exists():
            print(f"跳过缺失标签: {image_id}")
            continue
        try:
            result = evaluator.process_image_id(image_id, label_path)
            all_results.append(result)
            print(f"\n{result['image_name']} Mean IoU: {result['metrics']['mean_iou']:.4f}")
        except Exception as exc:
            print(f"处理 {image_id} 时出错: {exc}")
            import traceback
            traceback.print_exc()

    if all_results:
        evaluator.save_overall_metrics(all_results)
        print(f"\n评估完成！结果保存在: {settings['output_dir']}")
    else:
        print("没有成功处理任何图像。")
