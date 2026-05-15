#!/usr/bin/env python3
"""
SAM3 零样本评估 - Potsdam 数据集
测试 SAM3 对 Potsdam 6 类地物的分割能力

作者: Claude
日期: 2026-05-14
"""

import os
import json
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
from datetime import datetime
from pathlib import Path
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

    # Potsdam 类别定义
    CLASS_INFO = {
        0: {"name": " clutter/background", "prompt": "background clutter", "color": [255, 255, 255]},
        1: {"name": "impervious surface", "prompt": "impervious surface", "color": [0, 0, 0]},
        2: {"name": "building", "prompt": "building", "color": [0, 0, 255]},
        3: {"name": "low vegetation", "prompt": "low vegetation", "color": [0, 255, 255]},
        4: {"name": "tree", "prompt": "tree", "color": [0, 255, 0]},
        5: {"name": "car", "prompt": "car", "color": [255, 255, 0]}
    }

    # 配置参数
    PATCH_SIZE = 1008  # SAM3 输入尺寸
    STRIDE = 672       # 切片步长（适当重叠）
    SCORE_THRESHOLD = 0.5  # mask 置信度阈值

    def __init__(self, base_dir, output_dir="results", checkpoint_path=None):
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

        if SAM3_AVAILABLE:
            self._load_sam3_model()

        # 设置日志
        self._setup_logging()

    def _setup_logging(self):
        """设置日志记录"""
        log_file = self.output_dir / "logs" / f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SAM3 零样本评估开始 - 设备: {self.device}")

    def _load_sam3_model(self):
        """加载 SAM3 模型"""
        try:
            self.logger.info("正在加载 SAM3 模型...")

            # 构建 SAM3 图像模型
            if self.checkpoint_path:
                self.model = build_sam3_image_model(
                    checkpoint_path=self.checkpoint_path,
                    load_from_HF=False
                )
            else:
                # 从 HuggingFace 加载
                self.model = build_sam3_image_model()

            # 创建处理器
            self.processor = Sam3Processor(self.model)
            self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"SAM3 模型加载成功，使用设备: {self.device}")

        except Exception as e:
            self.logger.error(f"SAM3 模型加载失败: {e}")
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
        """读取标签文件"""
        label_path = Path(label_path)

        if GDAL_AVAILABLE:
            dataset = gdal.Open(str(label_path))
            label = dataset.ReadAsArray()
            dataset = None
        else:
            label = Image.open(label_path)
            label = np.array(label)

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
        合并 patches 回原始图像尺寸

        Args:
            patches: patches 列表 [(patch, x, y), ...]
            original_shape: 原始图像形状 (H, W)

        Returns:
            merged: 合并后的图像
        """
        h, w = original_shape
        if len(patches[0][0].shape) == 3:
            merged = np.zeros((h, w, patches[0][0].shape[2]), dtype=patches[0][0].dtype)
        else:
            merged = np.zeros((h, w), dtype=patches[0][0].dtype)

        # 用于统计每个像素被多少个 patch 覆盖
        count_map = np.zeros((h, w), dtype=np.int32)

        for patch, x, y in patches:
            patch_h = min(self.PATCH_SIZE, h - y)
            patch_w = min(self.PATCH_SIZE, w - x)

            # 累加（用于后续平均）
            if len(patch.shape) == 3:
                merged[y:y+patch_h, x:x+patch_w, :] += patch[:patch_h, :patch_w, :]
            else:
                merged[y:y+patch_h, x:x+patch_w] += patch[:patch_h, :patch_w]

            count_map[y:y+patch_h, x:x+patch_w] += 1

        # 避免除零
        count_map[count_map == 0] = 1

        # 平均重叠区域
        if len(merged.shape) == 3:
            for c in range(merged.shape[2]):
                merged[:, :, c] = merged[:, :, c] / count_map
        else:
            merged = merged / count_map

        return merged.astype(patches[0][0].dtype)

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
            # 转换为 PIL Image
            if isinstance(patch, np.ndarray):
                patch_rgb = patch[:, :, :3] if patch.shape[2] >= 3 else patch
                pil_image = Image.fromarray(patch_rgb.astype('uint8'))
            else:
                pil_image = patch

            # 设置图像
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
                    output = self.processor.set_text_prompt(
                        state=inference_state,
                        prompt=text_prompt
                    )

                    masks = output.get("masks", [])
                    boxes = output.get("boxes", [])
                    scores = output.get("scores", [])

                    if len(masks) > 0:
                        # 过滤低置信度预测
                        valid_indices = np.array(scores) >= self.SCORE_THRESHOLD

                        if np.any(valid_indices):
                            all_masks.extend([masks[i] for i in range(len(masks)) if valid_indices[i]])
                            all_boxes.extend([boxes[i] for i in range(len(boxes)) if valid_indices[i]])
                            all_scores.extend([scores[i] for i in range(len(scores)) if valid_indices[i]])
                            all_classes.extend([class_id] * np.sum(valid_indices))

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
            mask = masks[idx]
            score = scores[idx]
            class_id = classes[idx]

            # 调整 mask 尺寸
            if mask.shape != (h, w):
                mask_resized = cv2.resize(mask.astype(np.uint8), (w, h),
                                        interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask.astype(np.uint8)

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
            text_prompts = [self.CLASS_INFO[i]["prompt"] for i in range(6)]

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
        计算评估指标

        Args:
            prediction: 预测标签 (H, W)
            ground_truth: 真实标签 (H, W)

        Returns:
            metrics: 指标字典
        """
        from sklearn.metrics import confusion_matrix, jaccard_score, f1_score, accuracy_score

        # 展平
        pred_flat = prediction.flatten()
        gt_flat = ground_truth.flatten()

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
        f1_per_class = f1_score(gt_flat, pred_flat, labels=list(range(6)), average=None)
        mean_f1 = np.mean(f1_per_class)

        # 每个类别的精度
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
            "confusion_matrix": cm.tolist()
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

        for class_id, info in self.CLASS_INFO.items():
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
        for i, info in self.CLASS_INFO.items():
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
        overall_metrics["average_overall_accuracy"] = np.mean([r["metrics"]["overall_accuracy"] for r in all_results])
        overall_metrics["average_mean_iou"] = np.mean([r["metrics"]["mean_iou"] for r in all_results])
        overall_metrics["average_mean_f1"] = np.mean([r["metrics"]["mean_f1"] for r in all_results])

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


def main():
    """主函数"""

    # 配置路径
    BASE_DIR = "/home/anjou/PythonENV/Test_11/Potsdam"
    OUTPUT_DIR = "/home/anjou/PythonENV/Test_11/results"
    CHECKPOINT_PATH = "/home/anjou/PythonENV/Test_11/sam3/checkpoints/sam3.1_multiplex.pt"

    # 快速验证的图像选择（代表性场景组合）
    TEST_IMAGES = [
        "top_potsdam_2_10",  # 简单城市场景
        "top_potsdam_5_11",  # 中等复杂度场景
        "top_potsdam_7_9",   # 复杂密集建筑场景
    ]

    print("=" * 60)
    print("SAM3 零样本评估 - Potsdam 数据集")
    print("=" * 60)
    print(f"基础目录: {BASE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"测试图像: {TEST_IMAGES}")
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
        checkpoint_path=CHECKPOINT_PATH
    )

    # 处理每张测试图像
    all_results = []

    for image_id in TEST_IMAGES:
        # 构造文件路径
        image_path = f"{BASE_DIR}/2_Ortho_RGB/2_Ortho_RGB/{image_id}_RGB.tif"
        label_path = f"{BASE_DIR}/5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary/{image_id}_label_noBoundary.tif"

        # 检查文件是否存在
        if not Path(image_path).exists():
            print(f"错误: 图像文件不存在 - {image_path}")
            continue

        if not Path(label_path).exists():
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

            for i, info in evaluator.CLASS_INFO.items():
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
        for i, info in evaluator.CLASS_INFO.items():
            avg_iou = overall_metrics[f"average_iou_class_{i}"]
            print(f"  {info['name']}: {avg_iou:.4f}")
    else:
        print("没有成功处理任何图像")

    print("\n评估完成！")
    print(f"结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()