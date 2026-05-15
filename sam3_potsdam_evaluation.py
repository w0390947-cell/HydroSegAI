#!/usr/bin/env python3
"""
SAM3 零样本评估 - Potsdam 数据集
测试 SAM3 对 Potsdam 6 类地物的分割能力

作者: Claude
日期: 2026-05-14
"""

import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
except ImportError as e:
    print(f"警告: SAM3 模块导入失败 - {e}")
    print("请确保 SAM3 已正确安装: pip install -e /path/to/sam3")
    SAM3_AVAILABLE = False

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
        0: {"name": "clutter/background", "prompt": "background clutter", "color": [255, 255, 255]},
        1: {"name": "impervious surface", "prompt": "impervious surface", "color": [0, 0, 0]},
        2: {"name": "building", "prompt": "building", "color": [0, 0, 255]},
        3: {"name": "low vegetation", "prompt": "low vegetation", "color": [0, 255, 255]},
        4: {"name": "tree", "prompt": "tree", "color": [0, 255, 0]},
        5: {"name": "car", "prompt": "car", "color": [255, 255, 0]}
    }

    COLOR_TO_CLASS = {
        (255, 255, 255): 0,     # clutter/background
        (0, 0, 0): 1,           # impervious surface
        (0, 0, 255): 2,         # building
        (0, 255, 255): 3,       # low vegetation
        (0, 255, 0): 4,         # tree
        (255, 255, 0): 5,       # car
        (255, 0, 0): IGNORE_INDEX
    }

    # 配置参数
    PATCH_SIZE = 1008  # SAM3 输入尺寸
    STRIDE = 672       # 切片步长（适当重叠）
    SCORE_THRESHOLD = 0.5  # mask 置信度阈值

    def __init__(self, base_dir, output_dir="results", checkpoint_path=None,
                 patch_size=None, stride=None, score_threshold=None,
                 class_info=None, gpu_dtype="auto"):
        """
        初始化评估器

        Args:
            base_dir: Potsdam 数据集基础目录
            output_dir: 输出目录
            checkpoint_path: SAM3 权重路径（None 则使用默认）
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_path = checkpoint_path
        self.class_info = class_info or self.CLASS_INFO
        self.PATCH_SIZE = patch_size or self.PATCH_SIZE
        self.STRIDE = stride or self.STRIDE
        self.SCORE_THRESHOLD = score_threshold if score_threshold is not None else self.SCORE_THRESHOLD
        self.gpu_dtype_config = gpu_dtype
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 设置日志（必须在加载模型之前）
        self._setup_logging()

        if SAM3_AVAILABLE:
            self._load_sam3_model()

    def _setup_logging(self):
        """设置日志记录"""
        log_file = self.output_dir / "logs" / f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.info(f"SAM3 零样本评估开始 - 设备: {self.device}")

    def _resolve_gpu_dtype(self):
        """解析 GPU 推理 dtype；保持 GPU 运行，不通过降级 CPU 规避 dtype 问题。"""
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
            self.logger.warning(f"未知 gpu_dtype={self.gpu_dtype_config}，将使用 auto 策略")

        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

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

    def _inference_autocast(self):
        """SAM3 GPU 推理使用和模型一致的 autocast dtype，避免 BF16/Float 混用。"""
        if self.device != "cuda" or self.model_dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.model_dtype)

    def _load_sam3_model(self):
        """加载 SAM3 模型"""
        try:
            self.logger.info("正在加载 SAM3 模型...")

            # 检查 checkpoint 路径
            if self.checkpoint_path:
                checkpoint_path = Path(self.checkpoint_path)
                if not checkpoint_path.exists():
                    self.logger.warning(f"Checkpoint 文件不存在: {checkpoint_path}")
                    self.logger.info("尝试从 HuggingFace 加载模型...")
                    self.checkpoint_path = None
                else:
                    self.logger.info(f"使用本地 checkpoint: {checkpoint_path}")

            # 直接设置 BPE 文件路径
            bpe_path = "/home/anjou/PythonENV/Test_11/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

            if Path(bpe_path).exists():
                self.logger.info(f"找到 BPE 文件: {bpe_path}")
            else:
                self.logger.warning(f"BPE 文件不存在: {bpe_path}")
                bpe_path = None

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
                self.model = build_sam3_image_model(
                    bpe_path=bpe_path,
                    compile=False
                )

            # 创建处理器
            self.processor = Sam3Processor(self.model)
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
                (label_rgb[:, :, 0] == 255) &
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

        for y in range(0, h, self.STRIDE):
            for x in range(0, w, self.STRIDE):
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
        合并 patches 回原始图像尺寸（针对离散类别ID使用投票法）

        Args:
            patches: patches 列表 [(patch, x, y), ...]
            original_shape: 原始图像形状 (H, W)

        Returns:
            merged: 合并后的图像
        """
        h, w = original_shape

        # 对于离散类别，使用投票法而非平均
        # 存储每个像素的类别投票情况
        num_classes = len(self.class_info)
        class_votes = np.zeros((h, w, num_classes), dtype=np.uint8)
        count_map = np.zeros((h, w), dtype=np.uint8)

        for patch, x, y in patches:
            patch_h = min(self.PATCH_SIZE, h - y)
            patch_w = min(self.PATCH_SIZE, w - x)
            patch_data = patch[:patch_h, :patch_w]

            # 对每个像素的类别进行投票
            for class_id in range(num_classes):
                class_mask = (patch_data == class_id)
                class_votes[y:y+patch_h, x:x+patch_w, class_id] += class_mask.astype(np.uint8)

            count_map[y:y+patch_h, x:x+patch_w] += 1

        # 对每个像素选择票数最多的类别
        merged = np.argmax(class_votes, axis=2).astype(np.uint8)

        # 处理没有覆盖的区域（设为背景类0）
        merged[count_map == 0] = 0

        return merged

    def _to_numpy(self, value):
        """将 SAM3 输出统一转换为 NumPy，避免 GPU tensor 进入后续 OpenCV/NumPy 逻辑。"""
        if isinstance(value, torch.Tensor):
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

    def predict_patch(self, patch, text_prompts):
        """
        对单个 patch 进行 SAM3 预测

        Args:
            patch: 图像 patch (H, W, 3)
            text_prompts: 文本提示列表

        Returns:
            result: 预测结果字典
        """
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
            for class_id, text_prompt in enumerate(text_prompts):
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
                            all_masks.extend([masks[i] for i in range(num_items) if valid_indices[i]])
                            all_boxes.extend([boxes[i] if i < len(boxes) else None for i in range(num_items) if valid_indices[i]])
                            all_scores.extend([float(scores[i]) for i in range(num_items) if valid_indices[i]])
                            all_classes.extend([class_id] * int(np.sum(valid_indices)))

                except Exception as e:
                    self.logger.warning(f"类别 {class_id} ({text_prompt}) 预测失败: {e}")
                    continue

            return {
                "masks": all_masks,
                "boxes": all_boxes,
                "scores": all_scores,
                "classes": all_classes
            }

        except Exception as e:
            self.logger.error(f"Patch 预测失败: {e}")
            return {
                "masks": [],
                "boxes": [],
                "scores": [],
                "classes": []
            }

    def fuse_multiclass_masks(self, predictions, patch_shape):
        """
        基于置信度的多类别 mask 融合

        Args:
            predictions: SAM3 预测结果
            patch_shape: patch 形状 (H, W)

        Returns:
            fused_mask: 融合后的语义分割图 (H, W)
        """
        h, w = patch_shape
        fused_mask = np.zeros((h, w), dtype=np.uint8)
        confidence_map = np.zeros((h, w), dtype=np.float32)

        masks = predictions.get("masks", [])
        scores = predictions.get("scores", [])
        classes = predictions.get("classes", [])

        if len(masks) == 0:
            return fused_mask

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

        return fused_mask

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

        # 读取图像和标签
        image = self.read_image(image_path)
        label = self.read_label(label_path)

        self.logger.info(f"图像尺寸: {image.shape}, 标签尺寸: {label.shape}")

        # 切片
        patches = self.slice_image(image)

        # 对每个 patch 进行预测
        predicted_patches = []

        for i, (patch, x, y) in enumerate(patches):
            self.logger.info(f"处理 patch {i+1}/{len(patches)} at position ({x}, {y})")

            # 获取所有类别的文本提示
            text_prompts = [self.class_info[i]["prompt"] for i in range(len(self.class_info))]

            # 预测
            predictions = self.predict_patch(patch, text_prompts)

            # 融合多类别 mask
            patch_shape = patch.shape[:2]
            fused_mask = self.fuse_multiclass_masks(predictions, patch_shape)

            # 裁剪回原始大小
            patch_h = min(self.PATCH_SIZE, label.shape[0] - y)
            patch_w = min(self.PATCH_SIZE, label.shape[1] - x)
            fused_mask = fused_mask[:patch_h, :patch_w]

            predicted_patches.append((fused_mask, x, y))

        # 合并 patches
        predicted_label = self.merge_patches(predicted_patches, label.shape)

        # 计算评估指标
        metrics = self.calculate_metrics(predicted_label, label)

        # 保存结果
        self.save_results(image_name, image, label, predicted_label, metrics)

        return {
            "image_name": image_name,
            "predicted_label": predicted_label,
            "ground_truth": label,
            "metrics": metrics
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
        from sklearn.metrics import confusion_matrix, jaccard_score, f1_score, accuracy_score

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
                "iou_per_class": [0.0] * 6,
                "f1_per_class": [0.0] * 6,
                "precision_per_class": [0.0] * 6,
                "recall_per_class": [0.0] * 6,
                "confusion_matrix": [[0] * 6 for _ in range(6)],
                "valid_pixels": 0,
                "total_pixels": int(prediction.size),
                "ignored_pixels": int(prediction.size)
            }

        # 计算混淆矩阵
        cm = confusion_matrix(gt_flat, pred_flat, labels=list(range(6)))

        # 总体精度
        overall_accuracy = accuracy_score(gt_flat, pred_flat)

        # 每个类别的 IoU
        iou_per_class = []
        for i in range(6):
            intersection = np.sum((pred_flat == i) & (gt_flat == i))
            union = np.sum((pred_flat == i) | (gt_flat == i))
            iou = intersection / union if union > 0 else 0
            iou_per_class.append(iou)

        mean_iou = np.mean(iou_per_class)

        # 每个类别的 F1 分数
        f1_per_class = f1_score(gt_flat, pred_flat, labels=list(range(6)), average=None, zero_division=0)
        mean_f1 = np.mean(f1_per_class)

        # 每个类别的精度和召回率
        precision_per_class = []
        recall_per_class = []
        for i in range(6):
            true_positives = np.sum((pred_flat == i) & (gt_flat == i))
            false_positives = np.sum((pred_flat == i) & (gt_flat != i))
            false_negatives = np.sum((pred_flat != i) & (gt_flat == i))

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

            precision_per_class.append(precision)
            recall_per_class.append(recall)

        return {
            "overall_accuracy": float(overall_accuracy),
            "mean_iou": float(mean_iou),
            "mean_f1": float(mean_f1),
            "iou_per_class": [float(x) for x in iou_per_class],
            "f1_per_class": [float(x) for x in f1_per_class],
            "precision_per_class": [float(x) for x in precision_per_class],
            "recall_per_class": [float(x) for x in recall_per_class],
            "confusion_matrix": cm.tolist(),
            "valid_pixels": int(len(pred_flat)),
            "total_pixels": int(prediction.size),
            "ignored_pixels": int(prediction.size - len(pred_flat))
        }

    def save_results(self, image_name, image, ground_truth, prediction, metrics):
        """
        保存结果和可视化

        Args:
            image_name: 图像名称
            image: 原始图像
            ground_truth: 真实标签
            prediction: 预测标签
            metrics: 评估指标
        """
        # 保存预测标签
        pred_path = self.output_dir / "predictions" / f"{image_name}_prediction.png"
        pred_colored = self.colorize_label(prediction)
        Image.fromarray(pred_colored).save(pred_path)

        # 保存真实标签
        gt_path = self.output_dir / "predictions" / f"{image_name}_ground_truth.png"
        gt_colored = self.colorize_label(ground_truth)
        Image.fromarray(gt_colored).save(gt_path)

        # 创建可视化对比图
        vis_path = self.output_dir / "visualizations" / f"{image_name}_comparison.png"
        self.create_visualization(image, ground_truth, prediction, vis_path, metrics)

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

        colored[label == self.IGNORE_INDEX] = [255, 0, 0]

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

        metrics_text += "Per-class IoU:\n"
        for i, info in self.class_info.items():
            metrics_text += f"  {info['name']}: {metrics['iou_per_class'][i]:.4f}\n"

        fig.text(0.02, 0.02, metrics_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
                verticalalignment='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
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

    def save_overall_metrics(self, all_results):
        """
        保存总体评估指标

        Args:
            all_results: 所有图像的处理结果
        """
        # 计算总体指标
        overall_metrics = {
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_images": len(all_results),
            "image_names": [r["image_name"] for r in all_results],
        }

        # 平均指标
        overall_metrics["average_overall_accuracy"] = float(np.mean([r["metrics"]["overall_accuracy"] for r in all_results]))
        overall_metrics["average_mean_iou"] = float(np.mean([r["metrics"]["mean_iou"] for r in all_results]))
        overall_metrics["average_mean_f1"] = float(np.mean([r["metrics"]["mean_f1"] for r in all_results]))

        # 每个类别的平均 IoU
        for i in range(6):
            class_iou = [r["metrics"]["iou_per_class"][i] for r in all_results]
            overall_metrics[f"average_iou_class_{i}"] = float(np.mean(class_iou))

        # 保存 JSON
        json_path = self.output_dir / "metrics" / "overall_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(overall_metrics, f, indent=2)

        # 保存 CSV
        csv_path = self.output_dir / "metrics" / "per_image_metrics.csv"

        import pandas as pd
        df_data = []
        for r in all_results:
            row = {
                "image_name": r["image_name"],
                "overall_accuracy": r["metrics"]["overall_accuracy"],
                "mean_iou": r["metrics"]["mean_iou"],
                "mean_f1": r["metrics"]["mean_f1"]
            }
            for i in range(6):
                row[f"iou_class_{i}"] = r["metrics"]["iou_per_class"][i]
                row[f"f1_class_{i}"] = r["metrics"]["f1_per_class"][i]

            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)

        self.logger.info(f"总体指标已保存到 {json_path} 和 {csv_path}")

        return overall_metrics


def load_yaml_config(config_path):
    """读取 YAML 配置；缺少 PyYAML 或配置文件时返回空配置。"""
    if not config_path:
        return {}

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"警告: 配置文件不存在，将使用代码默认值 - {config_path}")
        return {}

    try:
        import yaml
    except ImportError:
        print("警告: 未安装 PyYAML，无法读取配置文件，将使用代码默认值")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_class_info(config):
    """从配置文件构建类别定义，缺省时使用代码内置 Potsdam 定义。"""
    class_config = config.get("classes", {})
    class_info = {}

    for key, value in class_config.items():
        if not key.startswith("class_") or not isinstance(value, dict):
            continue

        class_id = int(value.get("id", key.split("_", 1)[1]))
        class_info[class_id] = {
            "name": value.get("name", PotsdamSAM3Evaluator.CLASS_INFO[class_id]["name"]),
            "prompt": value.get("prompt", PotsdamSAM3Evaluator.CLASS_INFO[class_id]["prompt"]),
            "color": value.get("color", PotsdamSAM3Evaluator.CLASS_INFO[class_id]["color"]),
        }

    if not class_info:
        return PotsdamSAM3Evaluator.CLASS_INFO

    return dict(sorted(class_info.items()))


def build_test_images(config):
    evaluation_config = config.get("evaluation", {})
    mode = evaluation_config.get("mode", "quick")

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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SAM3 零样本评估 - Potsdam 数据集")
    parser.add_argument(
        "--config",
        default="config_sam3_evaluation.yaml",
        help="YAML 配置文件路径；不存在或缺少 PyYAML 时使用代码默认值"
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    paths_config = config.get("paths", {})
    image_processing_config = config.get("image_processing", {})
    device_config = config.get("device", {})

    # 配置路径
    BASE_DIR = paths_config.get("base_dir", "/home/anjou/PythonENV/Test_11/Potsdam")
    OUTPUT_DIR = paths_config.get("output_dir", "/home/anjou/PythonENV/Test_11/results")
    CHECKPOINT_PATH = paths_config.get(
        "checkpoint_path",
        "/home/anjou/PythonENV/Test_11/sam3/checkpoints/sam3.1_multiplex.pt"
    )
    IMAGE_SUBDIR = paths_config.get("image_subdir", "2_Ortho_RGB/2_Ortho_RGB")
    LABEL_SUBDIR = paths_config.get(
        "label_subdir",
        "5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary"
    )
    IMAGE_PATTERN = paths_config.get("image_pattern", "{image_id}_RGB.tif")
    LABEL_PATTERN = paths_config.get("label_pattern", "{image_id}_label_noBoundary.tif")

    # 快速验证的图像选择（代表性场景组合）
    TEST_IMAGES = build_test_images(config)
    CLASS_INFO = build_class_info(config)

    PATCH_SIZE = image_processing_config.get("patch_size", PotsdamSAM3Evaluator.PATCH_SIZE)
    STRIDE = image_processing_config.get("stride", PotsdamSAM3Evaluator.STRIDE)
    SCORE_THRESHOLD = image_processing_config.get("score_threshold", PotsdamSAM3Evaluator.SCORE_THRESHOLD)
    GPU_DTYPE = device_config.get("gpu_dtype", "auto")

    print("=" * 60)
    print("SAM3 零样本评估 - Potsdam 数据集")
    print("=" * 60)
    print(f"基础目录: {BASE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"配置文件: {args.config if config else '未使用配置文件'}")
    print(f"测试图像: {TEST_IMAGES}")
    print(f"Patch 参数: size={PATCH_SIZE}, stride={STRIDE}, score_threshold={SCORE_THRESHOLD}")
    print(f"GPU dtype: {GPU_DTYPE}")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # 检查依赖
    if not SAM3_AVAILABLE:
        print("错误: SAM3 模块不可用，请先安装 SAM3")
        return

    if not torch.cuda.is_available():
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
        gpu_dtype=GPU_DTYPE
    )

    # 处理每张测试图像
    all_results = []

    for image_id in TEST_IMAGES:
        # 构造文件路径
        image_path = Path(BASE_DIR) / IMAGE_SUBDIR / IMAGE_PATTERN.format(image_id=image_id)
        label_path = Path(BASE_DIR) / LABEL_SUBDIR / LABEL_PATTERN.format(image_id=image_id)

        # 检查文件是否存在
        if not image_path.exists():
            print(f"错误: 图像文件不存在 - {image_path}")
            continue

        if not label_path.exists():
            print(f"错误: 标签文件不存在 - {label_path}")
            continue

        try:
            # 处理图像
            result = evaluator.process_image(image_path, label_path)
            all_results.append(result)

            # 打印结果
            print(f"\n{result['image_name']} 评估结果:")
            print(f"  Overall Accuracy: {result['metrics']['overall_accuracy']:.4f}")
            print(f"  Mean IoU: {result['metrics']['mean_iou']:.4f}")
            print(f"  Mean F1: {result['metrics']['mean_f1']:.4f}")

            for i, info in evaluator.class_info.items():
                iou = result['metrics']['iou_per_class'][i]
                print(f"  IoU ({info['name']}): {iou:.4f}")

        except Exception as e:
            print(f"处理 {image_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存总体指标
    if len(all_results) > 0:
        print("\n" + "=" * 60)
        print("保存总体评估指标...")
        overall_metrics = evaluator.save_overall_metrics(all_results)

        print("\n总体评估结果:")
        print(f"  平均 Overall Accuracy: {overall_metrics['average_overall_accuracy']:.4f}")
        print(f"  平均 Mean IoU: {overall_metrics['average_mean_iou']:.4f}")
        print(f"  平均 Mean F1: {overall_metrics['average_mean_f1']:.4f}")

        print("\n各类别平均 IoU:")
        for i, info in evaluator.class_info.items():
            avg_iou = overall_metrics[f"average_iou_class_{i}"]
            print(f"  {info['name']}: {avg_iou:.4f}")
    else:
        print("没有成功处理任何图像")

    print("\n评估完成！")
    print(f"结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
