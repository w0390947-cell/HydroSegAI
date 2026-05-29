# 实验四：SegFormer + Potsdam RGB 监督基线 dev 结果说明

## 1. 实验目的

本实验用于建立 Potsdam 数据集上的监督语义分割基线，验证 SegFormer 是否能够提供比 SAM3 zero-shot prompt 方案更稳定的 dense semantic prior。

Exp1-Exp3 已经说明：

```text
SAM3 在 Potsdam 上可以通过视觉 prompt ensemble 获得一定前景候选能力，
但 zero-shot / prompt-driven 方式难以单独承担六类密集语义分割。
```

因此，Exp4 转向监督语义模型，目标是回答：

```text
使用 Potsdam 标注数据训练 SegFormer-B0 后，能否得到稳定的 6 类像素级语义预测，
并为后续 SAM3 + semantic prior / coarse-to-fine 融合提供基础模型？
```

本实验不是 SAM3 prompt 消融，而是监督 baseline 实验。

## 2. 数据设置

当前配置使用以下数据：

| 项目 | 路径 | 说明 |
|---|---|---|
| RGB 影像 | `Potsdam/2_Ortho_RGB/2_Ortho_RGB` | Potsdam RGB 正射影像 |
| 训练/验证标签 | `Potsdam/5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary` | 24 张 participant noBoundary 标签 |
| 输出目录 | `results_segformer_potsdam_rgb` | SegFormer RGB supervised baseline 输出 |

默认 split 如下：

| Split | 图像数 | 图像 ID |
|---|---:|---|
| train | 18 | `top_potsdam_2_10` 到 `top_potsdam_6_12` 中的默认训练集合 |
| val | 6 | `top_potsdam_7_7` 到 `top_potsdam_7_12` |

实际 patch 数：

| Split | Patch size | Stride | Patch 数 |
|---|---:|---:|---:|
| train | 512 | 512 | 2592 |
| val | 512 | 512 | 864 |

这里的 patch 切分与 SAM3 实验不同。SAM3 使用较大的 `1008 x 1008` patch 是为了适配 prompt mask 生成和减少推理调用；SegFormer 使用 `512 x 512` patch 是为了适配监督训练的 batch size、显存和 ADE20K 预训练模型的常见输入尺度。

## 3. 类别定义

训练和评估使用 Potsdam 标准 6 类：

| 类别 ID | 类别名称 | 英文名称 | 标签颜色 |
|---:|---|---|---|
| 0 | 不透水表面 | impervious surface | 白色 `(255,255,255)` |
| 1 | 建筑物 | building | 蓝色 `(0,0,255)` |
| 2 | 低矮植被 | low vegetation | 青色 `(0,255,255)` |
| 3 | 树木 | tree | 绿色 `(0,255,0)` |
| 4 | 汽车 | car | 黄色 `(255,255,0)` |
| 5 | 杂乱/背景 | clutter/background | 红色 `(255,0,0)` |
| 255 | 忽略区域 | ignore/don't care | 黑色 `(0,0,0)` |

其中 `255` 用于 noBoundary 标签中的忽略区域，不参与 loss 和指标计算。

## 4. 模型设置

本实验使用：

```text
SegFormer-B0
```

本地预训练权重路径为：

```text
models/segformer/segformer-b0-finetuned-ade-512-512
```

该模型原始分类头对应 ADE20K 150 类。训练时脚本会将分类头改为 Potsdam 6 类：

```yaml
model:
  pretrained_model_name_or_path: "models/segformer/segformer-b0-finetuned-ade-512-512"
  local_files_only: true
  ignore_mismatched_sizes: true
```

因此，加载报告中出现 150 类分类头被重建为 6 类是正常现象。

## 5. 训练配置

核心训练配置如下：

| 项目 | 数值 |
|---|---:|
| Epochs | 40 |
| Batch size | 4 |
| Eval batch size | 4 |
| Learning rate | 0.00006 |
| Weight decay | 0.01 |
| Optimizer | AdamW |
| Gradient accumulation | 1 |
| Max grad norm | 1.0 |
| AMP | true |
| Data augmentation | horizontal flip / vertical flip |
| Cache tiles | true |
| Num workers | 0 |

图像归一化使用 ImageNet mean/std：

```yaml
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]
```

## 6. 训练流程

训练脚本执行流程如下：

1. 读取 `segformer_potsdam_rgb.yaml`。
2. 根据默认 train/val split 构建图像列表。
3. 检查 train 与 val 是否存在重叠。
4. 读取 RGB 图像和 noBoundary 标签。
5. 将标签 RGB 颜色映射为 0-5 类别 ID，黑色映射为 255。
6. 将 6000×6000 大图切成 512×512 patch。
7. 过滤有效像素比例过低的 patch。
8. 对训练 patch 进行随机水平翻转和垂直翻转。
9. 使用 SegFormer 进行监督训练。
10. 每个 epoch 后在 validation patch 上计算 OA、mIoU、Mean F1 等指标。
11. 按 validation mIoU 保存 best checkpoint。
12. 保存 latest checkpoint、training history、summary 和运行上下文。

## 7. 输出文件

训练完成后，主要输出为：

```text
results_segformer_potsdam_rgb/
```

关键文件如下：

| 文件 | 说明 |
|---|---|
| `checkpoints/best/` | validation mIoU 最优 checkpoint |
| `checkpoints/latest/` | 最后一轮 checkpoint |
| `metrics/training_summary.json` | 最优 epoch、最优 mIoU 和运行上下文 |
| `metrics/training_history.json` | 逐 epoch 训练 loss 与验证指标 |
| `metrics/run_context.json` | train/val 图像 ID、patch 数、类别 ID |
| `metrics/config_snapshot.yaml` | 本次运行配置快照 |

训练阶段不会自动保存整图预测图。本次已额外运行 full-tile validation evaluation：

```bash
python eval_segformer_potsdam.py --config segformer_potsdam_rgb.yaml --split val
```

对应输出目录为：

```text
results_segformer_potsdam_rgb/eval_val/
```

其中保存了 6 张 validation tile 的整图预测、GT、对比可视化、单图指标和总体指标。

## 8. 指标定义

所有指标均在 `ground_truth != 255` 的有效像素上计算。

### 8.1 Overall Accuracy

```text
OA = 正确分类像素数 / 有效像素总数
```

### 8.2 Per-class IoU 与 Mean IoU

```text
IoU_c = TP_c / (TP_c + FP_c + FN_c)
mIoU = mean(IoU_0, IoU_1, IoU_2, IoU_3, IoU_4, IoU_5)
```

### 8.3 Precision、Recall 与 F1

```text
Precision_c = TP_c / (TP_c + FP_c)
Recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
```

### 8.4 Frequency Weighted IoU

```text
FWIoU = sum(freq_c * IoU_c)
freq_c = GT 类别 c 的有效像素数 / 全部有效像素数
```

## 9. 本次 full-tile validation 主结果

本次 full-tile validation 使用 best checkpoint：

```text
results_segformer_potsdam_rgb/checkpoints/best
```

评估输出保存于：

```text
results_segformer_potsdam_rgb/eval_val/
```

评估图像为 6 张 validation tile：

```text
top_potsdam_7_7_RGB
top_potsdam_7_8_RGB
top_potsdam_7_9_RGB
top_potsdam_7_10_RGB
top_potsdam_7_11_RGB
top_potsdam_7_12_RGB
```

full-tile sliding-window 评估的主结果如下：

| 指标 | 数值 |
|---|---:|
| Overall Accuracy | 0.8932 |
| Mean IoU | 0.7856 |
| Mean F1 | 0.8762 |
| Mean Precision | 0.8744 |
| Mean Recall | 0.8785 |
| Frequency Weighted IoU | 0.8130 |

逐类 IoU 如下：

| 类别 | IoU |
|---|---:|
| impervious surface | 0.8557 |
| building | 0.8962 |
| low vegetation | 0.6729 |
| tree | 0.7481 |
| car | 0.8936 |
| clutter/background | 0.6470 |

逐图结果如下：

| 图像 | OA | mIoU | Mean F1 | FWIoU |
|---|---:|---:|---:|---:|
| top_potsdam_7_7_RGB | 0.8930 | 0.7415 | 0.8301 | 0.8132 |
| top_potsdam_7_8_RGB | 0.8901 | 0.7207 | 0.8156 | 0.8117 |
| top_potsdam_7_9_RGB | 0.9110 | 0.8296 | 0.9052 | 0.8391 |
| top_potsdam_7_10_RGB | 0.8457 | 0.7473 | 0.8476 | 0.7442 |
| top_potsdam_7_11_RGB | 0.9202 | 0.7526 | 0.8413 | 0.8635 |
| top_potsdam_7_12_RGB | 0.8996 | 0.6842 | 0.7815 | 0.8431 |

## 10. Heldout full-tile 结果

本次还运行了 heldout full-tile evaluation：

```bash
python eval_segformer_potsdam.py --config segformer_potsdam_rgb.yaml --split heldout
```

输出目录为：

```text
results_segformer_potsdam_rgb/eval_heldout/
```

评估图像为 14 张 heldout tile：

```text
top_potsdam_2_13_RGB
top_potsdam_2_14_RGB
top_potsdam_3_13_RGB
top_potsdam_3_14_RGB
top_potsdam_4_13_RGB
top_potsdam_4_14_RGB
top_potsdam_4_15_RGB
top_potsdam_5_13_RGB
top_potsdam_5_14_RGB
top_potsdam_5_15_RGB
top_potsdam_6_13_RGB
top_potsdam_6_14_RGB
top_potsdam_6_15_RGB
top_potsdam_7_13_RGB
```

heldout 数据集级指标如下：

| 指标 | 数值 |
|---|---:|
| Overall Accuracy | 0.9021 |
| Mean IoU | 0.7687 |
| Mean F1 | 0.8535 |
| Mean Precision | 0.8772 |
| Mean Recall | 0.8413 |
| Frequency Weighted IoU | 0.8251 |

逐类 IoU 如下：

| 类别 | IoU |
|---|---:|
| impervious surface | 0.8576 |
| building | 0.9248 |
| low vegetation | 0.7767 |
| tree | 0.7860 |
| car | 0.9034 |
| clutter/background | 0.3635 |

Val 与 heldout 对比如下：

| Split | OA | mIoU | Mean F1 | FWIoU |
|---|---:|---:|---:|---:|
| val full-tile | 0.8932 | 0.7856 | 0.8762 | 0.8130 |
| heldout full-tile | 0.9021 | 0.7687 | 0.8535 | 0.8251 |

heldout 的 OA 和 FWIoU 高于 val，但 mIoU 和 Mean F1 略低。主要原因是 `clutter/background` 类下降明显，而其他 0-4 类均保持稳定或提升。

Heldout 逐类详细结果：

| 类别 | GT 占比 | 预测占比 | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 31.52% | 32.64% | 0.8576 | 0.9233 | 0.9075 | 0.9397 |
| building | 24.93% | 24.85% | 0.9248 | 0.9609 | 0.9625 | 0.9593 |
| low vegetation | 20.61% | 22.02% | 0.7767 | 0.8743 | 0.8464 | 0.9041 |
| tree | 17.09% | 16.27% | 0.7860 | 0.8802 | 0.9022 | 0.8593 |
| car | 1.59% | 1.61% | 0.9034 | 0.9493 | 0.9433 | 0.9553 |
| clutter/background | 4.26% | 2.61% | 0.3635 | 0.5332 | 0.7014 | 0.4301 |

因此，heldout 结果说明：

```text
SegFormer RGB baseline 对主要地物类别泛化良好；
当前主要短板是 clutter/background 欠召回。
```

## 11. 训练过程参考结果

本次训练的 best checkpoint 位于：

```text
results_segformer_potsdam_rgb/checkpoints/best
```

best epoch 为第 16 轮，训练过程中的 validation patch-level 指标如下：

| 指标 | 数值 |
|---|---:|
| Overall Accuracy | 0.8894 |
| Mean IoU | 0.7799 |
| Mean F1 | 0.8727 |
| Mean Precision | 0.8717 |
| Mean Recall | 0.8742 |
| Frequency Weighted IoU | 0.8067 |
| Validation loss | 0.4404 |

逐类 IoU 如下：

| 类别 | IoU |
|---|---:|
| impervious surface | 0.8495 |
| building | 0.8908 |
| low vegetation | 0.6713 |
| tree | 0.7381 |
| car | 0.8861 |
| clutter/background | 0.6436 |

full-tile validation 与 patch-level validation 对比如下：

| 评估方式 | OA | mIoU | Mean F1 | FWIoU |
|---|---:|---:|---:|---:|
| Patch-level validation | 0.8894 | 0.7799 | 0.8727 | 0.8067 |
| Full-tile sliding-window validation | 0.8932 | 0.7856 | 0.8762 | 0.8130 |

这说明 full-tile 滑窗融合没有削弱模型效果，反而带来轻微提升。

## 12. 使用建议

后续实验建议统一使用 best checkpoint：

```text
results_segformer_potsdam_rgb/checkpoints/best
```

不要默认使用 latest checkpoint，因为第 40 轮虽然训练 loss 更低，但 validation mIoU 从 0.7799 小幅下降到 0.7730。

推荐后续顺序：

1. 分析 heldout 中 `clutter/background` 的主要混淆方向。
2. 导出 SegFormer 的 softmax confidence / entropy，定位低置信区域。
3. 基于 SegFormer 输出设计 SAM3 refinement，例如边界细化、低置信区域重分割或类别候选 mask 约束。
