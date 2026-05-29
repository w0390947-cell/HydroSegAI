#!/usr/bin/env python3

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import json
import argparse
import copy
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext
import warnings
warnings.filterwarnings('ignore')

# 尝试导入 SAM3 相关模块
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
    SAM3_IMPORT_ERROR = None
except ImportError as e:
    print(f"警告: SAM3 模块导入失败 - {e}")
    print("请确保 SAM3 已正确安装: pip install -e /path/to/sam3")
    SAM3_AVAILABLE = False
    SAM3_IMPORT_ERROR = e

# 尝试导入 GDAL 用于读取 GeoTIFF
try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    print("警告: GDAL 未安装，将尝试使用 PIL 读取图像")
    GDAL_AVAILABLE = False


class PotsdamSAM3Evaluator:
    """SAM3 零样本评估器 - Potsdam 数据集"""

    IGNORE_INDEX = 255

    # Potsdam 类别定义
    CLASS_INFO = {
        0: {"name": "impervious surface", "prompt": "impervious surface", "color": [255, 255, 255]},
        1: {"name": "building", "prompt": "building", "color": [0, 0, 255]},
        2: {"name": "low vegetation", "prompt": "low vegetation", "color": [0, 255, 255]},
        3: {"name": "tree", "prompt": "tree", "color": [0, 255, 0]},
        4: {"name": "car", "prompt": "car", "color": [255, 255, 0]},
        5: {"name": "clutter/background", "prompt": "background clutter", "color": [255, 0, 0]}
    }

    COLOR_TO_CLASS = {
        (255, 255, 255): 0,     # impervious surface
        (0, 0, 255): 1,         # building
        (0, 255, 255): 2,       # low vegetation
        (0, 255, 0): 3,         # tree
        (255, 255, 0): 4,       # car
        (255, 0, 0): 5,         # clutter/background
        (0, 0, 0): IGNORE_INDEX # noBoundary ignore/don't care
    }

    # 配置参数
    PATCH_SIZE = 1008  # SAM3 输入尺寸
    STRIDE = 672       # 切片步长（适当重叠）
    SCORE_THRESHOLD = 0.5  # mask 置信度阈值

    def __init__(self, base_dir, output_dir="results_exp1_sam3_potsdam_zeroshot_baseline", checkpoint_path=None,
                 patch_size=None, stride=None, score_threshold=None,
                 class_info=None, gpu_dtype="float32", device_type="auto",
                 allow_hf_fallback=False, output_config=None, metrics_config=None):
        """
        初始化评估器

        Args:
            base_dir: Potsdam 数据集基础目录
            output_dir: 输出目录
            checkpoint_path: SAM3 权重路径（None 则使用默认）
            allow_hf_fallback: 本地 checkpoint 缺失或未配置时是否允许从 HuggingFace 加载
            output_config: 输出控制配置
            metrics_config: 指标保存控制配置
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_path = checkpoint_path
        self.configured_checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.allow_hf_fallback = bool(allow_hf_fallback)
        self.model_source = None
        self.model_checkpoint_path = None
        self.model_hf_identifier = None
        self.model_fallback_reason = None
        self.model_bpe_path = None
        self.class_info = dict(sorted((class_info or self.CLASS_INFO).items()))
        self.class_ids = sorted(self.class_info.keys())
        self.num_classes = len(self.class_ids)
        self.prompt_sources = {
            class_id: self.class_info[class_id].get("prompt_source", "default")
            for class_id in self.class_ids
        }
        self._validate_potsdam_class_colors()
        self.default_class_id = self._resolve_default_class_id()
        self.PATCH_SIZE = patch_size or self.PATCH_SIZE
        self.STRIDE = stride or self.STRIDE
        self.SCORE_THRESHOLD = score_threshold if score_threshold is not None else self.SCORE_THRESHOLD
        self.gpu_dtype_config = gpu_dtype
        self.device_type_config = str(device_type or "auto").lower()
        self.fusion_strategy = "confidence"
        self.output_config = self._resolve_output_config(output_config or {})
        self.metrics_config = self._resolve_metrics_config(metrics_config or {})
        self.model_dtype = None

        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "predictions").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # 初始化模型和处理器
        self.model = None
        self.processor = None
        self.device = self._resolve_device()
        self.inference_stats = self._new_inference_totals()

        # 设置日志（必须在加载模型之前）
        self._setup_logging()

        if SAM3_AVAILABLE:
            self._load_sam3_model()

    def _resolve_device(self):
        """解析推理设备；正式 GPU 实验在 CUDA 不可用时快速失败，避免误跑 CPU。"""
        if self.device_type_config == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("配置要求使用 CUDA，但当前 torch.cuda.is_available() 为 False")
            return "cuda"
        if self.device_type_config == "cpu":
            return "cpu"
        if self.device_type_config != "auto":
            raise ValueError(f"未知 device.type={self.device_type_config}，应为 auto/cuda/cpu")
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_output_config(self, output_config):
        """解析当前 Exp1 已实际接入的输出配置。"""
        image_format = str(output_config.get("image_format", "png") or "png").lower().lstrip(".")
        if image_format not in {"png", "tif", "tiff"}:
            raise ValueError(
                f"output.image_format={image_format!r} 不适合保存类别 ID 图。"
                "Exp1 当前仅支持 png/tif/tiff。"
            )
        visualization_dpi = int(output_config.get("visualization_dpi", 150))
        if visualization_dpi <= 0:
            raise ValueError(f"output.visualization_dpi 必须为正整数，当前为 {visualization_dpi}")

        log_level = str(output_config.get("log_level", "INFO") or "INFO").upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"未知 output.log_level={log_level!r}，应为 DEBUG/INFO/WARNING/ERROR/CRITICAL")

        return {
            "save_predictions": bool(output_config.get("save_predictions", True)),
            "save_visualizations": bool(output_config.get("save_visualizations", True)),
            "visualization_dpi": visualization_dpi,
            "image_format": image_format,
            "log_level": log_level,
        }

    def _resolve_metrics_config(self, metrics_config):
        """解析当前 Exp1 已实际接入的指标保存配置。"""
        return {
            "save_confusion_matrix": bool(metrics_config.get("save_confusion_matrix", False)),
            "detailed_class_report": bool(metrics_config.get("detailed_class_report", True)),
        }

    def _resolve_default_class_id(self):
        """未被任何 SAM3 mask 覆盖的像素按 Potsdam clutter/background 处理。"""
        for class_id, info in self.class_info.items():
            if info.get("name") == "clutter/background":
                return class_id
        return max(self.class_info.keys())

    def _official_color_for_class(self, class_id):
        """返回 Potsdam 官方 RGB 标签颜色；黑色固定为 ignore，不属于普通类别。"""
        for color, mapped_class_id in self.COLOR_TO_CLASS.items():
            if mapped_class_id == class_id:
                return color
        return None

    def _validate_potsdam_class_colors(self):
        """校验内置 Potsdam 类别 schema 与官方 RGB 标签编码一致。"""
        seen_colors = {}
        ignore_color = (0, 0, 0)

        for class_id, info in self.class_info.items():
            official_color = self._official_color_for_class(class_id)
            if official_color is None:
                raise ValueError(f"类别 id={class_id} 不在 Potsdam 官方 6 类标签编码中")

            color = tuple(int(value) for value in info.get("color", []))
            if len(color) != 3 or any(value < 0 or value > 255 for value in color):
                raise ValueError(f"类别 id={class_id} 的 color 必须是 0-255 范围内的 RGB 三元组，当前为 {info.get('color')}")

            if color == ignore_color:
                raise ValueError(
                    f"类别 id={class_id} 使用了黑色 {ignore_color}。"
                    "Potsdam noBoundary 标签中黑色固定表示 ignore/don't care，不能作为普通类别颜色。"
                )

            if color != official_color:
                raise ValueError(
                    f"类别 id={class_id} ({info.get('name')}) 的内置 schema color={list(color)} "
                    f"与 Potsdam 官方标签颜色 {list(official_color)} 不一致。"
                    "Exp1 使用官方固定 COLOR_TO_CLASS 解析 GT 标签，请保持内置 schema 与官方规范一致。"
                )

            if color in seen_colors:
                raise ValueError(
                    f"类别 id={class_id} 和 id={seen_colors[color]} 使用了重复颜色 {list(color)}"
                )
            seen_colors[color] = class_id

    def _setup_logging(self):
        """设置日志记录"""
        log_file = self.output_dir / "logs" / f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        import logging
        self.logger = logging.getLogger(__name__)
        log_level_name = self.output_config["log_level"]
        log_level = getattr(logging, log_level_name, logging.INFO)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        self.logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.info(f"SAM3 零样本评估开始 - 设备: {self.device}, 日志级别: {log_level_name}")

    def _resolve_gpu_dtype(self):
        """解析 GPU 推理 dtype；默认使用 Float32，尽量不改变模型数值行为。"""
        if self.device != "cuda":
            return torch.float32

        dtype_config = str(self.gpu_dtype_config or "auto").lower()
        if dtype_config in ("bf16", "bfloat16"):
            return torch.bfloat16
        if dtype_config in ("fp16", "float16", "half"):
            return torch.float16
        if dtype_config in ("fp32", "float32", "full"):
            return torch.float32
        if dtype_config != "auto":
            self.logger.warning(f"未知 gpu_dtype={self.gpu_dtype_config}，将使用 float32 策略")

        return torch.float32

    def _configure_cuda_precision(self):
        """设置 CUDA 计算精度，并统一模型参数 dtype。"""
        if self.device != "cuda":
            self.model_dtype = torch.float32
            return

        self.model_dtype = self._resolve_gpu_dtype()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.model.to(device=self.device, dtype=self.model_dtype)

        first_param = next(self.model.parameters(), None)
        if first_param is not None:
            self.logger.info(
                f"CUDA 推理 dtype 已设置为 {self.model_dtype}，"
                f"首个参数 dtype={first_param.dtype}, device={first_param.device}"
            )

        nn_module = getattr(torch, "nn", None)
        module_type = getattr(nn_module, "Module", None)
        if self.processor is not None and module_type is not None:
            for name, value in vars(self.processor).items():
                if isinstance(value, module_type):
                    value.to(device=self.device, dtype=self.model_dtype)
                    self.logger.info(f"Processor 模块已同步 dtype: {name}")

        if self.model_dtype == torch.float32:
            self._install_float32_fused_mlp_fallback()

    def _install_float32_fused_mlp_fallback(self):
        """在不修改 SAM3 源码的前提下，禁用会强制 BF16 的 fused MLP 快捷路径。"""
        try:
            import torch.nn.functional as F
            import sam3.model.vitdet as vitdet
        except ImportError as e:
            self.logger.warning(f"无法安装 Float32 fused MLP fallback: {e}")
            return

        current = getattr(vitdet, "addmm_act", None)
        if getattr(current, "_sam3_float32_safe", False):
            return

        def addmm_act_float32(activation, linear, mat1):
            if torch.is_grad_enabled():
                raise ValueError("Expected grad to be disabled.")

            y = linear(mat1)
            if activation in (F.relu, torch.nn.ReLU):
                return F.relu(y)
            if activation in (F.gelu, torch.nn.GELU):
                return F.gelu(y)
            raise ValueError(f"Unexpected activation {activation}")

        addmm_act_float32._sam3_float32_safe = True
        vitdet.addmm_act = addmm_act_float32
        self.logger.info("已启用 Float32 fused MLP fallback")

    def _inference_autocast(self):
        """仅在显式选择低精度时启用 autocast；Float32 路径保持模型原始数值行为。"""
        if self.device != "cuda" or self.model_dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.model_dtype)

    def _load_sam3_model(self):
        """加载 SAM3 模型"""
        try:
            self.logger.info("正在加载 SAM3 模型...")

            # 检查 checkpoint 路径
            checkpoint_path = Path(self.checkpoint_path).expanduser() if self.checkpoint_path else None
            if checkpoint_path:
                if not checkpoint_path.exists():
                    message = f"Checkpoint 文件不存在: {checkpoint_path}"
                    if not self.allow_hf_fallback:
                        raise FileNotFoundError(
                            f"{message}。当前 model.allow_hf_fallback=false，"
                            "请修正 checkpoint_path 或显式开启 HuggingFace fallback。"
                        )
                    self.logger.warning(message)
                    self.logger.warning("model.allow_hf_fallback=true，将从 HuggingFace 加载模型")
                    self.checkpoint_path = None
                    self.model_source = "huggingface"
                    self.model_checkpoint_path = None
                    self.model_hf_identifier = "build_sam3_image_model default HuggingFace checkpoint (version=sam3)"
                    self.model_fallback_reason = message
                else:
                    self.checkpoint_path = str(checkpoint_path)
                    self.model_source = "local_checkpoint"
                    self.model_checkpoint_path = str(checkpoint_path)
                    self.model_hf_identifier = None
                    self.model_fallback_reason = None
                    self.logger.info(f"使用本地 checkpoint: {checkpoint_path}")
            else:
                message = "未配置 checkpoint_path"
                if not self.allow_hf_fallback:
                    raise ValueError(
                        f"{message}，且 model.allow_hf_fallback=false。"
                        "请配置本地 checkpoint_path 或显式开启 HuggingFace fallback。"
                    )
                self.logger.warning(f"{message}，model.allow_hf_fallback=true，将从 HuggingFace 加载模型")
                self.model_source = "huggingface"
                self.model_checkpoint_path = None
                self.model_hf_identifier = "build_sam3_image_model default HuggingFace checkpoint (version=sam3)"
                self.model_fallback_reason = message

            # 直接设置 BPE 文件路径
            bpe_path = "/home/anjou/PythonENV/Test_11/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

            if Path(bpe_path).exists():
                self.logger.info(f"找到 BPE 文件: {bpe_path}")
                self.model_bpe_path = bpe_path
            else:
                self.logger.warning(f"BPE 文件不存在: {bpe_path}")
                bpe_path = None
                self.model_bpe_path = None

            # 构建 SAM3 图像模型
            if self.checkpoint_path:
                self.logger.info(f"从本地加载模型: {self.checkpoint_path}")
                self.model = build_sam3_image_model(
                    bpe_path=bpe_path,
                    checkpoint_path=str(self.checkpoint_path),
                    load_from_HF=False,
                    compile=False  # 禁用编译以避免数据类型问题
                )
            else:
                # 从 HuggingFace 加载
                self.logger.info("从 HuggingFace 加载模型...")
                self.model_source = "huggingface"
                self.model_hf_identifier = "build_sam3_image_model default HuggingFace checkpoint (version=sam3)"
                self.model = build_sam3_image_model(
                    bpe_path=bpe_path,
                    compile=False
                )

            # 创建处理器，并让 SAM3 内部候选过滤阈值与实验配置保持一致。
            self.processor = Sam3Processor(
                self.model,
                confidence_threshold=float(self.SCORE_THRESHOLD),
            )
            self.model.to(self.device)
            self._configure_cuda_precision()
            self.model.eval()

            self.logger.info(f"SAM3 模型加载成功，使用设备: {self.device}")

        except Exception as e:
            self.logger.error(f"SAM3 模型加载失败: {e}")
            self.logger.error(f"Checkpoint 路径: {self.checkpoint_path}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            raise

    def _build_model_metadata(self):
        """记录模型加载来源，避免 checkpoint fallback 影响论文实验可复现性。"""
        return {
            "allow_hf_fallback": self.allow_hf_fallback,
            "configured_checkpoint_path": self.configured_checkpoint_path,
            "model_source": self.model_source,
            "actual_checkpoint_path": self.model_checkpoint_path,
            "huggingface_identifier": self.model_hf_identifier,
            "fallback_reason": self.model_fallback_reason,
            "bpe_path": self.model_bpe_path,
        }

    def _build_effective_config_metadata(self):
        """记录 Exp1 当前真正接入并生效的配置项。"""
        return {
            "image_processing": {
                "patch_size": self.PATCH_SIZE,
                "stride": self.STRIDE,
                "score_threshold": self.SCORE_THRESHOLD,
            },
            "fusion": {
                "strategy": self.fusion_strategy,
                "implemented_strategies": ["confidence"],
            },
            "output": dict(self.output_config),
            "metrics": {
                "save_confusion_matrix": self.metrics_config["save_confusion_matrix"],
                "detailed_class_report": self.metrics_config["detailed_class_report"],
            },
        }

    def _build_label_encoding_metadata(self):
        """记录 Potsdam 官方固定 RGB 标签编码。"""
        color_to_class = {}
        for color, class_id in self.COLOR_TO_CLASS.items():
            key = ",".join(str(value) for value in color)
            color_to_class[key] = int(class_id)

        class_to_color = {}
        for color, class_id in self.COLOR_TO_CLASS.items():
            if class_id == self.IGNORE_INDEX:
                continue
            class_to_color[str(class_id)] = list(color)

        return {
            "source": "Potsdam official fixed RGB label encoding",
            "class_schema_source": "Potsdam built-in official schema",
            "color_to_class": color_to_class,
            "class_to_color": class_to_color,
            "ignore_color": [0, 0, 0],
            "ignore_index": self.IGNORE_INDEX,
        }

    def _build_prompt_metadata(self):
        """记录 Exp1 实际使用的类别 prompt 及来源。"""
        return {
            str(class_id): {
                "class_name": self.class_info[class_id]["name"],
                "prompt": self.class_info[class_id]["prompt"],
                "source": self.prompt_sources.get(class_id, "default"),
            }
            for class_id in self.class_ids
        }

    def read_image(self, image_path):
        """读取影像文件"""
        image_path = Path(image_path)

        if GDAL_AVAILABLE:
            # 使用 GDAL 读取 GeoTIFF
            dataset = gdal.Open(str(image_path))
            if dataset is None:
                raise FileNotFoundError(f"无法打开文件: {image_path}")

            image = dataset.ReadAsArray()

            # 转换维度顺序 (H, W, C) 如果是多波段
            if len(image.shape) == 3:
                # GDAL 读取为 (C, H, W)，需要转换为 (H, W, C)
                image = np.transpose(image, (1, 2, 0))

            dataset = None

        else:
            # 使用 PIL 读取
            image = Image.open(image_path)
            image = np.array(image)

            # 如果是 RGBA，转为 RGB
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = image[:, :, :3]

        return image

    def read_label(self, label_path):
        """读取标签文件并转换为类别ID"""
        label_path = Path(label_path)

        if GDAL_AVAILABLE:
            dataset = gdal.Open(str(label_path))
            if dataset is None:
                raise FileNotFoundError(f"无法打开标签文件: {label_path}")
            label_rgb = dataset.ReadAsArray()
            dataset = None

            # 如果是(C,H,W)格式，转换为(H,W,C)
            if len(label_rgb.shape) == 3 and label_rgb.shape[0] in (3, 4):
                label_rgb = np.transpose(label_rgb, (1, 2, 0))
        else:
            label_rgb = Image.open(label_path)
            label_rgb = np.array(label_rgb)

        if label_rgb.ndim == 3 and label_rgb.shape[-1] == 4:
            label_rgb = label_rgb[:, :, :3]

        # 检查是否为RGB格式
        if label_rgb.ndim == 3 and label_rgb.shape[-1] == 3:
            # 转换RGB到类别ID
            h, w = label_rgb.shape[:2]
            label = np.full((h, w), self.IGNORE_INDEX, dtype=np.uint8)

            for color, class_id in self.COLOR_TO_CLASS.items():
                mask = (label_rgb[:, :, 0] == color[0]) & \
                       (label_rgb[:, :, 1] == color[1]) & \
                       (label_rgb[:, :, 2] == color[2])
                label[mask] = class_id

            unknown_mask = label == self.IGNORE_INDEX
            known_ignore = (
                (label_rgb[:, :, 0] == 0) &
                (label_rgb[:, :, 1] == 0) &
                (label_rgb[:, :, 2] == 0)
            )
            unknown_count = int(np.sum(unknown_mask & ~known_ignore))
            if unknown_count > 0:
                self.logger.warning(f"发现 {unknown_count} 个未知颜色标签像素，已按 ignore 处理")

            self.logger.info(f"RGB标签已转换为类别ID，形状: {label.shape}")
        else:
            # 假设已经是类别ID格式
            label = np.asarray(label_rgb)
            if label.ndim == 3 and label.shape[-1] == 1:
                label = label[:, :, 0]
            label = label.astype(np.uint8, copy=False)
            self.logger.info(f"标签已经是类别ID格式，形状: {label.shape}")

        return label

    def _validate_sample_shapes(self, image, label, image_path, label_path):
        """Exp1 RGB baseline 要求影像和标签严格像素对齐。"""
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(
                "Exp1 要求输入影像为 HWC 格式且至少包含 3 个通道。"
                f"当前 image_shape={image.shape}, image_path={image_path}"
            )

        if label.ndim != 2:
            raise ValueError(
                "Exp1 要求标签为二维类别 ID 图。"
                f"当前 label_shape={label.shape}, label_path={label_path}"
            )

        if image.shape[:2] != label.shape:
            raise ValueError(
                "图像和标签尺寸不一致，Potsdam RGB 与标签应严格像素对齐。"
                f"image_shape={image.shape}, label_shape={label.shape}, "
                f"image_path={image_path}, label_path={label_path}"
            )

    def _generate_patch_starts(self, length):
        """生成边缘对齐的 patch 起点，避免最后一个 patch 大面积补零。"""
        if length <= self.PATCH_SIZE:
            return [0]

        starts = list(range(0, length - self.PATCH_SIZE + 1, self.STRIDE))
        edge_start = length - self.PATCH_SIZE
        if starts[-1] != edge_start:
            starts.append(edge_start)
        return starts

    def slice_image(self, image):
        """
        将大图像切片成小 patches

        Args:
            image: 输入图像 (H, W, C)

        Returns:
            patches: 切片列表 [(patch, x, y), ...]
        """
        h, w = image.shape[:2]
        patches = []
        y_starts = self._generate_patch_starts(h)
        x_starts = self._generate_patch_starts(w)

        for y in y_starts:
            for x in x_starts:
                # 计算实际切片区域
                patch_h = min(self.PATCH_SIZE, h - y)
                patch_w = min(self.PATCH_SIZE, w - x)

                # 提取 patch
                if len(image.shape) == 3:
                    patch = image[y:y+patch_h, x:x+patch_w, :]
                else:
                    patch = image[y:y+patch_h, x:x+patch_w]

                # 如果 patch 小于 PATCH_SIZE，进行填充
                if patch_h < self.PATCH_SIZE or patch_w < self.PATCH_SIZE:
                    if len(image.shape) == 3:
                        padded_patch = np.zeros((self.PATCH_SIZE, self.PATCH_SIZE, image.shape[2]), dtype=image.dtype)
                        padded_patch[:patch_h, :patch_w, :] = patch
                    else:
                        padded_patch = np.zeros((self.PATCH_SIZE, self.PATCH_SIZE), dtype=image.dtype)
                        padded_patch[:patch_h, :patch_w] = patch
                    patch = padded_patch

                patches.append((patch, x, y))

        self.logger.info(f"图像切片完成: {len(patches)} 个 patches (原始尺寸: {h}x{w})")
        return patches

    def merge_patches(self, patches, original_shape):
        """
        合并 patches 回原始图像尺寸（优先使用 SAM3 置信度加权融合）

        Args:
            patches: patches 列表 [(patch, confidence, x, y), ...]
            original_shape: 原始图像形状 (H, W)

        Returns:
            merged: 合并后的图像
        """
        h, w = original_shape

        num_classes = max(self.class_ids) + 1
        class_scores = np.zeros((h, w, num_classes), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.uint8)
        positive_score_map = np.zeros((h, w), dtype=bool)

        for patch, confidence, x, y in patches:
            patch_h = min(self.PATCH_SIZE, h - y)
            patch_w = min(self.PATCH_SIZE, w - x)
            patch_data = patch[:patch_h, :patch_w]
            confidence_data = confidence[:patch_h, :patch_w].astype(np.float32, copy=False)

            count_map[y:y+patch_h, x:x+patch_w] += 1
            positive_score_map[y:y+patch_h, x:x+patch_w] |= confidence_data > 0

            # 对每个像素的类别累计置信度分数。
            for class_id in self.class_ids:
                class_mask = (patch_data == class_id)
                class_scores[y:y+patch_h, x:x+patch_w, class_id] += confidence_data * class_mask

        # 对每个像素选择累计置信度最高的类别
        merged = np.argmax(class_scores, axis=2).astype(np.uint8)

        uncovered_pixels = int(np.sum(count_map == 0))
        if uncovered_pixels > 0:
            raise RuntimeError(
                f"Patch 合并发现 {uncovered_pixels} 个未覆盖像素；"
                "请检查切片起点、patch_size 和 stride 设置。"
            )

        # 已被 patch 覆盖但没有任何 SAM3 正置信度的区域，回退为 Potsdam clutter/background。
        merged[~positive_score_map] = self.default_class_id

        return merged

    def _to_numpy(self, value):
        """将 SAM3 输出统一转换为 NumPy，避免 GPU tensor 进入后续 OpenCV/NumPy 逻辑。"""
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.bfloat16:
                value = value.float()
            return value.detach().cpu().numpy()
        return value

    def _normalize_masks(self, masks):
        masks = self._to_numpy(masks)
        if masks is None:
            return []
        if isinstance(masks, (list, tuple)):
            normalized = [np.squeeze(self._to_numpy(mask)) for mask in masks]
        else:
            masks = np.asarray(masks)
            if masks.ndim == 2:
                normalized = [masks]
            elif masks.ndim == 3:
                normalized = [np.squeeze(mask) for mask in masks]
            elif masks.ndim == 4:
                normalized = [np.squeeze(mask) for mask in masks]
            else:
                normalized = []
        return [mask for mask in normalized if np.asarray(mask).ndim == 2]

    def _normalize_scores(self, scores):
        scores = self._to_numpy(scores)
        if scores is None:
            return np.array([], dtype=np.float32)
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def _normalize_boxes(self, boxes):
        boxes = self._to_numpy(boxes)
        if boxes is None:
            return []
        if isinstance(boxes, (list, tuple)):
            return [self._to_numpy(box) for box in boxes]
        boxes = np.asarray(boxes)
        if boxes.ndim == 1:
            return [boxes]
        return [box for box in boxes]

    def _new_inference_totals(self):
        return {
            "patch_calls": 0,
            "patch_failures": 0,
            "prompt_calls": 0,
            "prompt_successes": 0,
            "prompt_failures": 0,
            "empty_prompt_outputs": 0,
            "valid_masks": 0,
            "images": {},
        }

    def _new_image_inference_stats(self, image_name):
        stats = self._new_inference_totals()
        stats["image_name"] = image_name
        stats.pop("images", None)
        return stats

    def _merge_image_inference_stats(self, image_stats):
        for key in (
            "patch_calls",
            "patch_failures",
            "prompt_calls",
            "prompt_successes",
            "prompt_failures",
            "empty_prompt_outputs",
            "valid_masks",
        ):
            self.inference_stats[key] += int(image_stats.get(key, 0))
        self.inference_stats["images"][image_stats["image_name"]] = dict(image_stats)

    def _validate_image_inference_health(self, image_stats):
        """区分正常 zero-shot 空 mask 与推理链路异常。"""
        if image_stats["patch_failures"] > 0:
            raise RuntimeError(
                f"{image_stats['image_name']} 存在 patch 级推理失败: "
                f"patch_failures={image_stats['patch_failures']}"
            )

        prompt_calls = int(image_stats["prompt_calls"])
        prompt_failures = int(image_stats["prompt_failures"])
        if prompt_calls == 0:
            raise RuntimeError(f"{image_stats['image_name']} 没有执行任何 prompt 推理调用")

        failure_rate = prompt_failures / prompt_calls
        image_stats["prompt_failure_rate"] = failure_rate
        if failure_rate > 0.05:
            raise RuntimeError(
                f"{image_stats['image_name']} prompt 推理失败率过高: "
                f"{prompt_failures}/{prompt_calls} ({failure_rate:.2%})"
            )

        if image_stats["valid_masks"] == 0:
            raise RuntimeError(
                f"{image_stats['image_name']} 没有产生任何有效 SAM3 mask。"
                "这更可能是模型加载、阈值、API 或推理链路异常，而不是合法 zero-shot 结果。"
            )

    def predict_patch(self, patch, text_prompts, image_stats=None, patch_index=None, patch_position=None):
        """
        对单个 patch 进行 SAM3 预测

        Args:
            patch: 图像 patch (H, W, 3)
            text_prompts: 文本提示列表

        Returns:
            result: 预测结果字典
        """
        if image_stats is not None:
            image_stats["patch_calls"] += 1

        try:
            # 转换为 PIL Image 并确保数据类型正确
            if isinstance(patch, np.ndarray):
                patch_rgb = patch[:, :, :3] if patch.shape[2] >= 3 else patch
                # 确保图像数据是 uint8 类型
                if patch_rgb.dtype != np.uint8:
                    patch_rgb = (patch_rgb * 255).astype(np.uint8) if patch_rgb.max() <= 1.0 else patch_rgb.astype(np.uint8)
                pil_image = Image.fromarray(patch_rgb)
            else:
                pil_image = patch

            # 设置图像
            with torch.inference_mode(), self._inference_autocast():
                inference_state = self.processor.set_image(pil_image)

            # 存储所有类别的预测结果
            all_masks = []
            all_boxes = []
            all_scores = []
            all_classes = []

            # 对每个类别进行预测
            for prompt_index, prompt_item in enumerate(text_prompts):
                if isinstance(prompt_item, (tuple, list)) and len(prompt_item) == 2:
                    class_id, text_prompt = prompt_item
                else:
                    class_id, text_prompt = prompt_index, prompt_item

                if image_stats is not None:
                    image_stats["prompt_calls"] += 1

                try:
                    # 使用文本提示
                    with torch.inference_mode(), self._inference_autocast():
                        output = self.processor.set_text_prompt(
                            state=inference_state,
                            prompt=text_prompt
                        )

                    masks = output.get("masks", [])
                    boxes = output.get("boxes", [])
                    scores = output.get("scores", [])

                    masks = self._normalize_masks(masks)
                    scores = self._normalize_scores(scores)
                    boxes = self._normalize_boxes(boxes)

                    if len(masks) > 0:
                        if len(scores) == 0:
                            self.logger.warning(f"类别 {class_id} ({text_prompt}) 返回 mask 但没有 score，已跳过")
                            if image_stats is not None:
                                image_stats["prompt_successes"] += 1
                                image_stats["empty_prompt_outputs"] += 1
                            continue

                        num_items = min(len(masks), len(scores))
                        if len(masks) != len(scores):
                            self.logger.warning(
                                f"类别 {class_id} ({text_prompt}) mask/score 数量不一致: "
                                f"{len(masks)} vs {len(scores)}，仅使用前 {num_items} 个"
                            )
                            masks = masks[:num_items]
                            scores = scores[:num_items]
                            boxes = boxes[:num_items]

                        # 过滤低置信度预测
                        valid_indices = scores >= self.SCORE_THRESHOLD

                        if np.any(valid_indices):
                            valid_count = int(np.sum(valid_indices))
                            all_masks.extend([masks[i] for i in range(num_items) if valid_indices[i]])
                            all_boxes.extend([boxes[i] if i < len(boxes) else None for i in range(num_items) if valid_indices[i]])
                            all_scores.extend([float(scores[i]) for i in range(num_items) if valid_indices[i]])
                            all_classes.extend([class_id] * valid_count)
                            if image_stats is not None:
                                image_stats["valid_masks"] += valid_count
                        else:
                            if image_stats is not None:
                                image_stats["empty_prompt_outputs"] += 1
                    else:
                        if image_stats is not None:
                            image_stats["empty_prompt_outputs"] += 1

                    if image_stats is not None:
                        image_stats["prompt_successes"] += 1

                except Exception as e:
                    if image_stats is not None:
                        image_stats["prompt_failures"] += 1
                    self.logger.warning(f"类别 {class_id} ({text_prompt}) 预测失败: {e}")
                    continue

            return {
                "masks": all_masks,
                "boxes": all_boxes,
                "scores": all_scores,
                "classes": all_classes
            }

        except Exception as e:
            if image_stats is not None:
                image_stats["patch_failures"] += 1
            self.logger.error(f"Patch 预测失败: {e}")
            raise RuntimeError(
                f"Patch 预测失败: patch_index={patch_index}, position={patch_position}, error={e}"
            ) from e

    def fuse_multiclass_masks(self, predictions, patch_shape):
        """
        基于置信度的多类别 mask 融合

        Args:
            predictions: SAM3 预测结果
            patch_shape: patch 形状 (H, W)

        Returns:
            fused_mask: 融合后的语义分割图 (H, W)
            confidence_map: 每个像素被采用 mask 的 SAM3 置信度
        """
        h, w = patch_shape
        fused_mask = np.full((h, w), self.default_class_id, dtype=np.uint8)
        confidence_map = np.zeros((h, w), dtype=np.float32)

        masks = predictions.get("masks", [])
        scores = predictions.get("scores", [])
        classes = predictions.get("classes", [])

        if len(masks) == 0:
            return fused_mask, confidence_map

        # 按置信度排序
        sorted_indices = np.argsort(scores)[::-1]

        for idx in sorted_indices:
            mask = np.squeeze(self._to_numpy(masks[idx]))
            score = float(scores[idx])
            class_id = classes[idx]

            if mask.ndim != 2:
                self.logger.warning(f"跳过维度异常的 mask: shape={mask.shape}")
                continue

            # 调整 mask 尺寸
            if mask.shape != (h, w):
                mask_resized = cv2.resize(mask.astype(np.float32), (w, h),
                                          interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask.astype(np.float32)

            # 只在置信度更高的区域更新
            update_mask = (mask_resized > 0.5) & (confidence_map < score)
            fused_mask[update_mask] = class_id
            confidence_map[update_mask] = score

        return fused_mask, confidence_map

    def process_image(self, image_path, label_path):
        """
        处理单张图像的完整流程

        Args:
            image_path: 图像路径
            label_path: 标签路径

        Returns:
            results: 处理结果字典
        """
        image_name = Path(image_path).stem
        self.logger.info(f"正在处理: {image_name}")
        image_stats = self._new_image_inference_stats(image_name)

        # 读取图像和标签
        image = self.read_image(image_path)
        label = self.read_label(label_path)
        self._validate_sample_shapes(image, label, image_path, label_path)

        self.logger.info(f"图像尺寸: {image.shape}, 标签尺寸: {label.shape}")

        # 切片
        patches = self.slice_image(image)

        # 对每个 patch 进行预测
        predicted_patches = []

        for i, (patch, x, y) in enumerate(patches):
            self.logger.info(f"处理 patch {i+1}/{len(patches)} at position ({x}, {y})")

            # 获取所有类别的文本提示
            text_prompts = [(i, self.class_info[i]["prompt"]) for i in self.class_ids]

            # 预测
            predictions = self.predict_patch(
                patch,
                text_prompts,
                image_stats=image_stats,
                patch_index=i + 1,
                patch_position=(x, y),
            )

            # 融合多类别 mask
            patch_shape = patch.shape[:2]
            fused_mask, confidence_map = self.fuse_multiclass_masks(predictions, patch_shape)

            # 裁剪回原始大小
            patch_h = min(self.PATCH_SIZE, label.shape[0] - y)
            patch_w = min(self.PATCH_SIZE, label.shape[1] - x)
            fused_mask = fused_mask[:patch_h, :patch_w]
            confidence_map = confidence_map[:patch_h, :patch_w]

            predicted_patches.append((fused_mask, confidence_map, x, y))

        self._validate_image_inference_health(image_stats)
        self._merge_image_inference_stats(image_stats)

        # 合并 patches
        predicted_label = self.merge_patches(predicted_patches, label.shape)

        # 计算评估指标
        metrics = self.calculate_metrics(predicted_label, label)

        # 保存结果
        self.save_results(image_name, image, label, predicted_label, metrics, inference_stats=image_stats)

        return {
            "image_name": image_name,
            "metrics": metrics,
            "inference_stats": image_stats,
        }

    def calculate_metrics(self, prediction, ground_truth):
        """
        计算评估指标（排除忽略区域）

        Args:
            prediction: 预测标签 (H, W)
            ground_truth: 真实标签 (H, W)

        Returns:
            metrics: 指标字典
        """
        from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

        # 排除忽略区域（255）
        valid_mask = ground_truth != self.IGNORE_INDEX

        # 展平并过滤
        pred_flat = prediction[valid_mask]
        gt_flat = ground_truth[valid_mask]

        if len(pred_flat) == 0:
            self.logger.warning("没有有效像素用于计算指标")
            return {
                "overall_accuracy": 0.0,
                "mean_iou": 0.0,
                "mean_f1": 0.0,
                "mean_precision": 0.0,
                "mean_recall": 0.0,
                "frequency_weighted_iou": 0.0,
                "iou_per_class": [0.0] * self.num_classes,
                "f1_per_class": [0.0] * self.num_classes,
                "precision_per_class": [0.0] * self.num_classes,
                "recall_per_class": [0.0] * self.num_classes,
                "confusion_matrix": [[0] * self.num_classes for _ in range(self.num_classes)],
                "valid_pixels": 0,
                "total_pixels": int(prediction.size),
                "ignored_pixels": int(prediction.size)
            }

        # 计算混淆矩阵
        cm = confusion_matrix(gt_flat, pred_flat, labels=self.class_ids)

        # 总体精度
        overall_accuracy = accuracy_score(gt_flat, pred_flat)

        # 每个类别的 IoU
        iou_per_class = []
        for i in self.class_ids:
            intersection = np.sum((pred_flat == i) & (gt_flat == i))
            union = np.sum((pred_flat == i) | (gt_flat == i))
            iou = intersection / union if union > 0 else 0
            iou_per_class.append(iou)

        mean_iou = np.mean(iou_per_class)

        # 频率加权 IoU：按真实标签中各类别像素占比加权。
        class_frequencies = []
        for i in self.class_ids:
            class_frequencies.append(np.sum(gt_flat == i) / len(gt_flat))
        frequency_weighted_iou = float(np.sum(np.asarray(class_frequencies) * np.asarray(iou_per_class)))

        # 每个类别的 F1 分数
        f1_per_class = f1_score(gt_flat, pred_flat, labels=self.class_ids, average=None, zero_division=0)
        mean_f1 = np.mean(f1_per_class)

        # 每个类别的精度和召回率
        precision_per_class = []
        recall_per_class = []
        for i in self.class_ids:
            true_positives = np.sum((pred_flat == i) & (gt_flat == i))
            false_positives = np.sum((pred_flat == i) & (gt_flat != i))
            false_negatives = np.sum((pred_flat != i) & (gt_flat == i))

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

            precision_per_class.append(precision)
            recall_per_class.append(recall)

        mean_precision = np.mean(precision_per_class)
        mean_recall = np.mean(recall_per_class)

        return {
            "overall_accuracy": float(overall_accuracy),
            "mean_iou": float(mean_iou),
            "mean_f1": float(mean_f1),
            "mean_precision": float(mean_precision),
            "mean_recall": float(mean_recall),
            "frequency_weighted_iou": float(frequency_weighted_iou),
            "iou_per_class": [float(x) for x in iou_per_class],
            "f1_per_class": [float(x) for x in f1_per_class],
            "precision_per_class": [float(x) for x in precision_per_class],
            "recall_per_class": [float(x) for x in recall_per_class],
            "confusion_matrix": cm.tolist(),
            "valid_pixels": int(len(pred_flat)),
            "total_pixels": int(prediction.size),
            "ignored_pixels": int(prediction.size - len(pred_flat))
        }

    def save_results(self, image_name, image, ground_truth, prediction, metrics, inference_stats=None):
        """
        保存结果和可视化

        Args:
            image_name: 图像名称
            image: 原始图像
            ground_truth: 真实标签
            prediction: 预测标签
            metrics: 评估指标
            inference_stats: 单图推理健康统计
        """
        image_format = self.output_config["image_format"]

        if self.output_config["save_predictions"]:
            # 保存单通道类别 ID，便于后续复算指标或进行机器读取。
            pred_id_path = self.output_dir / "predictions" / f"{image_name}_prediction_id.{image_format}"
            Image.fromarray(prediction.astype(np.uint8, copy=False)).save(pred_id_path)

            gt_id_path = self.output_dir / "predictions" / f"{image_name}_ground_truth_id.{image_format}"
            Image.fromarray(ground_truth.astype(np.uint8, copy=False)).save(gt_id_path)

            # 保存彩色预测标签，便于人工查看。
            pred_path = self.output_dir / "predictions" / f"{image_name}_prediction.{image_format}"
            pred_colored = self.colorize_label(prediction)
            Image.fromarray(pred_colored).save(pred_path)

            # 保存彩色真实标签。
            gt_path = self.output_dir / "predictions" / f"{image_name}_ground_truth.{image_format}"
            gt_colored = self.colorize_label(ground_truth)
            Image.fromarray(gt_colored).save(gt_path)

        if self.output_config["save_visualizations"]:
            # 创建可视化对比图
            vis_path = self.output_dir / "visualizations" / f"{image_name}_comparison.{image_format}"
            self.create_visualization(image, ground_truth, prediction, vis_path, metrics)

        # 保存单图完整指标，包括 confusion matrix 和 per-class precision/recall。
        metrics_path = self.output_dir / "metrics" / f"{image_name}_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "image_name": image_name,
                    "class_ids": self.class_ids,
                    "class_names": [self.class_info[i]["name"] for i in self.class_ids],
                    "metrics": metrics,
                    "inference_stats": inference_stats or {},
                },
                f,
                indent=2,
                ensure_ascii=False
            )

        self.logger.info(f"结果已保存: {image_name}")

    def colorize_label(self, label):
        """
        将标签转换为彩色图像

        Args:
            label: 标签图像 (H, W)

        Returns:
            colored: 彩色图像 (H, W, 3)
        """
        h, w = label.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        colored[label == self.IGNORE_INDEX] = [0, 0, 0]

        for class_id, info in self.class_info.items():
            mask = label == class_id
            colored[mask] = info["color"]

        return colored

    def create_visualization(self, image, ground_truth, prediction, save_path, metrics):
        """
        创建可视化对比图

        Args:
            image: 原始图像
            ground_truth: 真实标签
            prediction: 预测标签
            save_path: 保存路径
            metrics: 评估指标
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        # 原始图像
        axes[0, 0].imshow(image[:, :, :3] if image.shape[2] >= 3 else image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # 真实标签
        gt_colored = self.colorize_label(ground_truth)
        axes[0, 1].imshow(gt_colored)
        axes[0, 1].set_title('Ground Truth Label', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # 预测标签
        pred_colored = self.colorize_label(prediction)
        axes[1, 0].imshow(pred_colored)
        axes[1, 0].set_title('SAM3 Prediction', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # 叠加显示
        overlay = self.create_overlay(image, prediction)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        # 添加指标文本
        metrics_text = f"Metrics:\n"
        metrics_text += f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n"
        metrics_text += f"Mean IoU: {metrics['mean_iou']:.4f}\n"
        metrics_text += f"Mean F1: {metrics['mean_f1']:.4f}\n\n"
        metrics_text += f"Mean Precision: {metrics['mean_precision']:.4f}\n"
        metrics_text += f"Mean Recall: {metrics['mean_recall']:.4f}\n"
        metrics_text += f"FWIoU: {metrics['frequency_weighted_iou']:.4f}\n\n"

        metrics_text += "Per-class IoU:\n"
        for class_index, (class_id, info) in enumerate(self.class_info.items()):
            metrics_text += f"  {info['name']}: {metrics['iou_per_class'][class_index]:.4f}\n"

        fig.text(0.02, 0.02, metrics_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
                verticalalignment='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.output_config["visualization_dpi"], bbox_inches='tight')
        plt.close()

    def create_overlay(self, image, prediction):
        """
        创建预测叠加图

        Args:
            image: 原始图像
            prediction: 预测标签

        Returns:
            overlay: 叠加图像
        """
        # 调整图像大小
        if image.shape[:2] != prediction.shape[:2]:
            image_resized = cv2.resize(image, (prediction.shape[1], prediction.shape[0]))
        else:
            image_resized = image

        # 转换为 float
        if image_resized.dtype == np.uint8:
            image_float = image_resized.astype(np.float32) / 255.0
        else:
            image_float = image_resized

        # 创建彩色预测图
        pred_colored = self.colorize_label(prediction).astype(np.float32) / 255.0

        # 叠加
        overlay = 0.6 * image_float[:, :, :3] + 0.4 * pred_colored
        overlay = np.clip(overlay, 0, 1)

        return (overlay * 255).astype(np.uint8)

    def _build_inference_health_metadata(self, all_results):
        """汇总成功样本的 SAM3 推理健康统计。"""
        totals = self._new_inference_totals()
        for result in all_results:
            image_stats = result.get("inference_stats", {})
            if not image_stats:
                continue
            for key in (
                "patch_calls",
                "patch_failures",
                "prompt_calls",
                "prompt_successes",
                "prompt_failures",
                "empty_prompt_outputs",
                "valid_masks",
            ):
                totals[key] += int(image_stats.get(key, 0))
            totals["images"][result["image_name"]] = image_stats

        prompt_calls = totals["prompt_calls"]
        totals["prompt_failure_rate"] = (
            totals["prompt_failures"] / prompt_calls if prompt_calls > 0 else 0.0
        )
        totals["health_policy"] = {
            "patch_failures_allowed_per_image": 0,
            "max_prompt_failure_rate_per_image": 0.05,
            "min_valid_masks_per_image": 1,
            "pixel_without_valid_mask_fallback": "clutter/background",
        }
        return totals

    def save_overall_metrics(self, all_results, run_context=None):
        """
        保存总体评估指标

        Args:
            all_results: 所有图像的处理结果
            run_context: 样本完整性、配置路径等运行上下文
        """
        # 计算总体指标
        overall_metrics = {
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_images": len(all_results),
            "image_names": [r["image_name"] for r in all_results],
            "class_ids": self.class_ids,
            "class_names": [self.class_info[i]["name"] for i in self.class_ids],
            "class_colors": [self.class_info[i]["color"] for i in self.class_ids],
            "ignore_index": self.IGNORE_INDEX,
            "model": self._build_model_metadata(),
            "effective_config": self._build_effective_config_metadata(),
            "label_encoding": self._build_label_encoding_metadata(),
            "prompts": self._build_prompt_metadata(),
            "run_context": run_context or {},
            "inference_health": self._build_inference_health_metadata(all_results),
            "primary_metrics": [
                "dataset_mean_iou",
                "dataset_mean_f1",
                "dataset_overall_accuracy",
                "dataset_frequency_weighted_iou",
            ],
        }

        confusion_matrices = [
            np.asarray(r["metrics"]["confusion_matrix"], dtype=np.int64)
            for r in all_results
        ]
        dataset_cm = np.sum(confusion_matrices, axis=0)
        dataset_metrics = self.metrics_from_confusion_matrix(dataset_cm)
        overall_metrics["dataset_overall_accuracy"] = dataset_metrics["overall_accuracy"]
        overall_metrics["dataset_mean_iou"] = dataset_metrics["mean_iou"]
        overall_metrics["dataset_mean_f1"] = dataset_metrics["mean_f1"]
        overall_metrics["dataset_mean_precision"] = dataset_metrics["mean_precision"]
        overall_metrics["dataset_mean_recall"] = dataset_metrics["mean_recall"]
        overall_metrics["dataset_frequency_weighted_iou"] = dataset_metrics["frequency_weighted_iou"]
        overall_metrics["dataset_iou_per_class"] = dataset_metrics["iou_per_class"]
        overall_metrics["dataset_f1_per_class"] = dataset_metrics["f1_per_class"]
        overall_metrics["dataset_precision_per_class"] = dataset_metrics["precision_per_class"]
        overall_metrics["dataset_recall_per_class"] = dataset_metrics["recall_per_class"]
        overall_metrics["dataset_confusion_matrix"] = dataset_cm.tolist()
        overall_metrics["dataset_valid_pixels"] = int(sum(r["metrics"]["valid_pixels"] for r in all_results))
        overall_metrics["dataset_ignored_pixels"] = int(sum(r["metrics"]["ignored_pixels"] for r in all_results))

        # 逐图平均指标作为辅助统计；论文主结果优先看 dataset_* 累计指标。
        overall_metrics["average_overall_accuracy"] = float(np.mean([r["metrics"]["overall_accuracy"] for r in all_results]))
        overall_metrics["average_mean_iou"] = float(np.mean([r["metrics"]["mean_iou"] for r in all_results]))
        overall_metrics["average_mean_f1"] = float(np.mean([r["metrics"]["mean_f1"] for r in all_results]))
        overall_metrics["average_mean_precision"] = float(np.mean([r["metrics"]["mean_precision"] for r in all_results]))
        overall_metrics["average_mean_recall"] = float(np.mean([r["metrics"]["mean_recall"] for r in all_results]))
        overall_metrics["average_frequency_weighted_iou"] = float(np.mean([r["metrics"]["frequency_weighted_iou"] for r in all_results]))

        if self.metrics_config["detailed_class_report"]:
            for class_index, class_id in enumerate(self.class_ids):
                class_iou = [r["metrics"]["iou_per_class"][class_index] for r in all_results]
                class_f1 = [r["metrics"]["f1_per_class"][class_index] for r in all_results]
                class_precision = [r["metrics"]["precision_per_class"][class_index] for r in all_results]
                class_recall = [r["metrics"]["recall_per_class"][class_index] for r in all_results]
                overall_metrics[f"average_iou_class_{class_id}"] = float(np.mean(class_iou))
                overall_metrics[f"average_f1_class_{class_id}"] = float(np.mean(class_f1))
                overall_metrics[f"average_precision_class_{class_id}"] = float(np.mean(class_precision))
                overall_metrics[f"average_recall_class_{class_id}"] = float(np.mean(class_recall))
                overall_metrics[f"dataset_iou_class_{class_id}"] = dataset_metrics["iou_per_class"][class_index]
                overall_metrics[f"dataset_f1_class_{class_id}"] = dataset_metrics["f1_per_class"][class_index]
                overall_metrics[f"dataset_precision_class_{class_id}"] = dataset_metrics["precision_per_class"][class_index]
                overall_metrics[f"dataset_recall_class_{class_id}"] = dataset_metrics["recall_per_class"][class_index]

        # 保存 JSON
        json_path = self.output_dir / "metrics" / "overall_metrics.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(overall_metrics, f, indent=2, ensure_ascii=False)

        # 保存 CSV
        csv_path = self.output_dir / "metrics" / "per_image_metrics.csv"

        import csv
        df_data = []
        for r in all_results:
            row = {
                "image_name": r["image_name"],
                "overall_accuracy": r["metrics"]["overall_accuracy"],
                "mean_iou": r["metrics"]["mean_iou"],
                "mean_f1": r["metrics"]["mean_f1"],
                "mean_precision": r["metrics"]["mean_precision"],
                "mean_recall": r["metrics"]["mean_recall"],
                "frequency_weighted_iou": r["metrics"]["frequency_weighted_iou"]
            }
            for class_index, class_id in enumerate(self.class_ids):
                row[f"iou_class_{class_id}"] = r["metrics"]["iou_per_class"][class_index]
                row[f"f1_class_{class_id}"] = r["metrics"]["f1_per_class"][class_index]
                row[f"precision_class_{class_id}"] = r["metrics"]["precision_per_class"][class_index]
                row[f"recall_class_{class_id}"] = r["metrics"]["recall_per_class"][class_index]

            df_data.append(row)

        fieldnames = list(df_data[0].keys()) if df_data else []
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(df_data)

        if self.metrics_config["save_confusion_matrix"]:
            cm_csv_path = self.output_dir / "metrics" / "dataset_confusion_matrix.csv"
            with open(cm_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["ground_truth\\prediction"] + [
                    f"{class_id}:{self.class_info[class_id]['name']}"
                    for class_id in self.class_ids
                ])
                for class_index, class_id in enumerate(self.class_ids):
                    writer.writerow(
                        [f"{class_id}:{self.class_info[class_id]['name']}"] +
                        dataset_cm[class_index].astype(int).tolist()
                    )

        self.logger.info(f"总体指标已保存到 {json_path} 和 {csv_path}")

        return overall_metrics

    def metrics_from_confusion_matrix(self, cm):
        """从累计混淆矩阵计算数据集级别指标。"""
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


def load_yaml_config(config_path):
    """读取 YAML 配置；正式 Exp1 要求配置文件和 PyYAML 都存在。"""
    if not config_path:
        raise ValueError("Exp1 正式评估必须显式提供 YAML 配置文件路径")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("未安装 PyYAML，无法读取 Exp1 YAML 配置文件") from e

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config_snapshot(config_path, output_dir):
    """保存原始 YAML 配置快照，便于论文实验复现。"""
    if not config_path:
        return None

    source_path = Path(config_path)
    if not source_path.exists():
        return None

    snapshot_path = Path(output_dir) / "metrics" / "config_snapshot.yaml"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    return str(snapshot_path)


def save_run_context(output_dir, run_context):
    """即使没有成功样本，也保存本次运行的样本处理状态。"""
    context_path = Path(output_dir) / "metrics" / "run_context.json"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    with open(context_path, "w", encoding="utf-8") as f:
        json.dump(run_context, f, indent=2, ensure_ascii=False)
    return str(context_path)


def _parse_prompt_class_id(key):
    if isinstance(key, int):
        return key
    if isinstance(key, str) and key.startswith("class_"):
        suffix = key.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    raise ValueError(f"prompt key 必须是 class_0 到 class_5，当前为 {key!r}")


def build_class_info(config):
    """构建 Potsdam 官方类别 schema，并应用 YAML 中可调的类别文本 prompt。"""
    class_info = copy.deepcopy(PotsdamSAM3Evaluator.CLASS_INFO)
    for class_id in class_info:
        class_info[class_id]["prompt_source"] = "default"

    prompt_config = config.get("prompts", {})
    if prompt_config is None:
        prompt_config = {}
    if not isinstance(prompt_config, dict):
        raise ValueError("prompts 必须是 class_id 到 prompt 字符串的映射")

    for key, prompt in prompt_config.items():
        class_id = _parse_prompt_class_id(key)
        if class_id not in class_info:
            raise ValueError(f"prompts.{key} 指向不存在的 Potsdam 类别 id={class_id}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompts.{key} 必须是非空字符串")

        class_info[class_id]["prompt"] = prompt.strip()
        class_info[class_id]["prompt_source"] = "yaml"

    return dict(sorted(class_info.items()))


def _potsdam_image_id_sort_key(image_id):
    return [int(part) if part.isdigit() else part for part in image_id.split("_")]


def _pattern_parts(pattern):
    """拆分包含 {image_id} 的文件命名模式，用于按配置发现可配对样本。"""
    if "{image_id}" not in pattern:
        raise ValueError(f"文件命名模式必须包含 {{image_id}}: {pattern}")
    prefix, suffix = pattern.split("{image_id}", 1)
    return prefix, suffix


def _extract_image_id_from_pattern(path, prefix, suffix):
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    end = len(name) - len(suffix) if suffix else len(name)
    image_id = name[len(prefix):end]
    return image_id or None


def discover_paired_image_ids(base_dir, image_subdir, label_subdir,
                              image_pattern="{image_id}_RGB.tif",
                              label_pattern="{image_id}_label_noBoundary.tif"):
    """发现同时具备 RGB 影像和 noBoundary 标签的 Potsdam 瓦片 ID。"""
    if not base_dir or not image_subdir or not label_subdir:
        return []

    image_dir = Path(base_dir) / image_subdir
    label_dir = Path(base_dir) / label_subdir

    if not image_dir.exists() or not label_dir.exists():
        return []

    image_prefix, image_suffix = _pattern_parts(image_pattern)
    label_prefix, label_suffix = _pattern_parts(label_pattern)

    image_ids = set()
    for path in image_dir.glob(f"{image_prefix}*{image_suffix}"):
        image_id = _extract_image_id_from_pattern(path, image_prefix, image_suffix)
        if image_id:
            image_ids.add(image_id)

    label_ids = set()
    for path in label_dir.glob(f"{label_prefix}*{label_suffix}"):
        image_id = _extract_image_id_from_pattern(path, label_prefix, label_suffix)
        if image_id:
            label_ids.add(image_id)

    paired_ids = image_ids & label_ids
    return sorted(paired_ids, key=_potsdam_image_id_sort_key)


def build_test_images(config, base_dir=None, image_subdir=None, label_subdir=None,
                      image_pattern="{image_id}_RGB.tif",
                      label_pattern="{image_id}_label_noBoundary.tif"):
    evaluation_config = config.get("evaluation", {})

    explicit_images = evaluation_config.get("test_images")
    if explicit_images:
        return explicit_images

    discovered = discover_paired_image_ids(
        base_dir,
        image_subdir,
        label_subdir,
        image_pattern,
        label_pattern,
    )
    if not discovered:
        raise RuntimeError(
            "未能自动发现任何可配对样本。请检查 paths.base_dir、image_subdir、label_subdir、"
            "image_pattern、label_pattern，或在 evaluation.test_images 中显式指定样本。"
        )

    return discovered


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SAM3 零样本评估 - Potsdam 数据集")
    parser.add_argument(
        "--config",
        default="exp1_sam3_potsdam_zeroshot_baseline.yaml",
        help="YAML 配置文件路径；正式 Exp1 要求该文件存在且 PyYAML 可用"
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    paths_config = config.get("paths", {})
    image_processing_config = config.get("image_processing", {})
    device_config = config.get("device", {})
    model_config = config.get("model", {})
    output_config = config.get("output", {})
    metrics_config = config.get("metrics", {})

    # 配置路径
    BASE_DIR = paths_config.get("base_dir", "/home/anjou/PythonENV/Test_11/Potsdam")
    OUTPUT_DIR = paths_config.get(
        "output_dir",
        "/home/anjou/PythonENV/Test_11/results_exp1_sam3_potsdam_zeroshot_baseline"
    )
    CHECKPOINT_PATH = paths_config.get(
        "checkpoint_path",
        "/home/anjou/PythonENV/Test_11/sam3/checkpoints/sam3.1_multiplex.pt"
    )
    IMAGE_SUBDIR = paths_config.get("image_subdir", "2_Ortho_RGB/2_Ortho_RGB")
    LABEL_SUBDIR = paths_config.get(
        "label_subdir",
        "5_Labels_all_noBoundary"
    )
    IMAGE_PATTERN = paths_config.get("image_pattern", "{image_id}_RGB.tif")
    LABEL_PATTERN = paths_config.get("label_pattern", "{image_id}_label_noBoundary.tif")

    # 根据配置选择图像；论文主实验推荐从 RGB 与 noBoundary 标签交集自动发现。
    TEST_IMAGES = build_test_images(
        config,
        BASE_DIR,
        IMAGE_SUBDIR,
        LABEL_SUBDIR,
        IMAGE_PATTERN,
        LABEL_PATTERN,
    )
    CLASS_INFO = build_class_info(config)

    PATCH_SIZE = image_processing_config.get("patch_size", PotsdamSAM3Evaluator.PATCH_SIZE)
    STRIDE = image_processing_config.get("stride", PotsdamSAM3Evaluator.STRIDE)
    SCORE_THRESHOLD = image_processing_config.get("score_threshold", PotsdamSAM3Evaluator.SCORE_THRESHOLD)
    GPU_DTYPE = device_config.get("gpu_dtype", "float32")
    DEVICE_TYPE = device_config.get("type", "auto")
    ALLOW_HF_FALLBACK = model_config.get("allow_hf_fallback", False)

    print("=" * 60)
    print("SAM3 零样本评估 - Potsdam 数据集")
    print("=" * 60)
    print(f"基础目录: {BASE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"配置文件: {args.config}")
    print(f"测试图像: {TEST_IMAGES}")
    print(f"Patch 参数: size={PATCH_SIZE}, stride={STRIDE}, score_threshold={SCORE_THRESHOLD}")
    print("融合策略: confidence (Exp1 固定)")
    print(f"设备配置: {DEVICE_TYPE}")
    print(f"GPU dtype: {GPU_DTYPE}")
    print(f"允许 HuggingFace fallback: {ALLOW_HF_FALLBACK}")
    print(
        "输出配置: "
        f"save_predictions={output_config.get('save_predictions', True)}, "
        f"save_visualizations={output_config.get('save_visualizations', True)}, "
        f"image_format={output_config.get('image_format', 'png')}"
    )
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # 检查依赖
    if not SAM3_AVAILABLE:
        project_sam3_dir = Path.cwd() / "sam3"
        sam3_path_entries = [path for path in sys.path if "sam3" in path.lower()]
        raise RuntimeError(
            "SAM3 模块不可用，无法执行 Exp1 正式评估。"
            f"import_error={SAM3_IMPORT_ERROR!r}; "
            f"python_executable={sys.executable}; "
            f"cwd={Path.cwd()}; "
            f"project_sam3_dir_exists={project_sam3_dir.exists()}; "
            f"sam3_entries_in_sys_path={sam3_path_entries}"
        )

    if DEVICE_TYPE != "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU（可能很慢）")

    # 创建评估器
    evaluator = PotsdamSAM3Evaluator(
        base_dir=BASE_DIR,
        output_dir=OUTPUT_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        score_threshold=SCORE_THRESHOLD,
        class_info=CLASS_INFO,
        gpu_dtype=GPU_DTYPE,
        device_type=DEVICE_TYPE,
        allow_hf_fallback=ALLOW_HF_FALLBACK,
        output_config=output_config,
        metrics_config=metrics_config
    )

    # 处理每张测试图像
    all_results = []
    expected_images = list(TEST_IMAGES)
    processed_images = []
    skipped_images = []
    failed_images = []

    for image_id in TEST_IMAGES:
        # 构造文件路径
        image_path = Path(BASE_DIR) / IMAGE_SUBDIR / IMAGE_PATTERN.format(image_id=image_id)
        label_path = Path(BASE_DIR) / LABEL_SUBDIR / LABEL_PATTERN.format(image_id=image_id)

        # 检查文件是否存在
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
            # 处理图像
            result = evaluator.process_image(image_path, label_path)
            all_results.append(result)
            processed_images.append(result["image_name"])

            # 打印结果
            print(f"\n{result['image_name']} 评估结果:")
            print(f"  Overall Accuracy: {result['metrics']['overall_accuracy']:.4f}")
            print(f"  Mean IoU: {result['metrics']['mean_iou']:.4f}")
            print(f"  Mean F1: {result['metrics']['mean_f1']:.4f}")
            print(f"  Mean Precision: {result['metrics']['mean_precision']:.4f}")
            print(f"  Mean Recall: {result['metrics']['mean_recall']:.4f}")
            print(f"  Frequency Weighted IoU: {result['metrics']['frequency_weighted_iou']:.4f}")

            for class_index, (class_id, info) in enumerate(evaluator.class_info.items()):
                iou = result['metrics']['iou_per_class'][class_index]
                print(f"  IoU ({info['name']}): {iou:.4f}")

        except Exception as e:
            print(f"处理 {image_id} 时出错: {e}")
            failed_images.append({
                "image_id": image_id,
                "reason": str(e),
                "exception_type": type(e).__name__,
                "image_path": str(image_path),
                "label_path": str(label_path),
            })
            import traceback
            traceback.print_exc()
            continue

    config_snapshot_path = save_config_snapshot(args.config, OUTPUT_DIR)
    run_context = {
        "config_path": str(args.config) if args.config else None,
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
    run_context_path = save_run_context(OUTPUT_DIR, run_context)

    # 保存总体指标
    if len(all_results) > 0:
        print("\n" + "=" * 60)
        print("保存总体评估指标...")
        overall_metrics = evaluator.save_overall_metrics(all_results, run_context=run_context)

        if skipped_images or failed_images:
            print(
                "警告: 本次评估存在未纳入总体指标的样本。"
                f"期望 {len(expected_images)} 张，成功 {len(processed_images)} 张，"
                f"跳过 {len(skipped_images)} 张，失败 {len(failed_images)} 张。"
                "总体指标仅基于成功样本。"
            )

        print("\n总体评估结果:")
        print(f"  数据集 Overall Accuracy: {overall_metrics['dataset_overall_accuracy']:.4f}")
        print(f"  数据集 Mean IoU: {overall_metrics['dataset_mean_iou']:.4f}")
        print(f"  数据集 Mean F1: {overall_metrics['dataset_mean_f1']:.4f}")
        print(f"  数据集 Mean Precision: {overall_metrics['dataset_mean_precision']:.4f}")
        print(f"  数据集 Mean Recall: {overall_metrics['dataset_mean_recall']:.4f}")
        print(f"  数据集 Frequency Weighted IoU: {overall_metrics['dataset_frequency_weighted_iou']:.4f}")
        print(f"  逐图平均 Overall Accuracy: {overall_metrics['average_overall_accuracy']:.4f}")
        print(f"  逐图平均 Mean IoU: {overall_metrics['average_mean_iou']:.4f}")
        print(f"  逐图平均 Mean F1: {overall_metrics['average_mean_f1']:.4f}")
        print(f"  逐图平均 Mean Precision: {overall_metrics['average_mean_precision']:.4f}")
        print(f"  逐图平均 Mean Recall: {overall_metrics['average_mean_recall']:.4f}")
        print(f"  逐图平均 Frequency Weighted IoU: {overall_metrics['average_frequency_weighted_iou']:.4f}")

        print("\n各类别数据集 IoU:")
        for class_index, (class_id, info) in enumerate(evaluator.class_info.items()):
            dataset_iou = overall_metrics["dataset_iou_per_class"][class_index]
            print(f"  {info['name']}: {dataset_iou:.4f}")
    else:
        raise RuntimeError(
            "没有成功处理任何图像；不会生成总体指标。"
            f"样本处理状态已保存到 {run_context_path}"
        )

    print("\n评估完成！")
    print(f"结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
