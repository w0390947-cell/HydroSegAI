# 实验一：SAM3 + Potsdam 零样本遥感分割基线结果说明

## 1. 实验目的

本实验用于评估 SAM3 在不训练、不微调条件下，对 ISPRS Potsdam 遥感 RGB 正射影像的零样本语义分割能力。

实验输入为 Potsdam RGB 图像，文本提示为 Potsdam 六个语义类别，输出为 6 类语义分割结果，并与 `noBoundary` 参考标签进行像素级评估。

## 2. 数据设置

当前脚本默认使用以下数据：

| 项目 | 路径 | 说明 |
|---|---|---|
| 输入影像 | `Potsdam/2_Ortho_RGB/2_Ortho_RGB` | RGB 真彩色正射影像 |
| 参考标签 | `Potsdam/5_Labels_all_noBoundary` | 38 张 noBoundary 语义标签 |

脚本通过 `discover_from_files: true` 自动发现同时存在 RGB 图像和 noBoundary 标签的样本。当前项目中可配对样本数为 38 张。

## 3. 类别定义

评估使用 Potsdam 标准 6 类语义定义：

| 类别 ID | 类别名称 | 英文名称 | 标签颜色 |
|---:|---|---|---|
| 0 | 不透水表面 | impervious surface | 白色 `(255,255,255)` |
| 1 | 建筑物 | building | 蓝色 `(0,0,255)` |
| 2 | 低矮植被 | low vegetation | 青色 `(0,255,255)` |
| 3 | 树木 | tree | 绿色 `(0,255,0)` |
| 4 | 汽车 | car | 黄色 `(255,255,0)` |
| 5 | 杂乱/背景 | clutter/background | 红色 `(255,0,0)` |
| 255 | 忽略区域 | ignore/don't care | 黑色 `(0,0,0)` |

其中 `255` 只用于 noBoundary 标签中的边界或忽略区域，不参与指标计算。

## 4. 推理流程

1. 读取 RGB 图像和 noBoundary 标签。
2. 将 RGB 大图切成 `1008 x 1008` patch。
3. 使用边缘对齐切片，避免图像右侧和下侧出现大面积补零 patch。
4. 对每个 patch 分别使用 6 个文本提示调用 SAM3。
5. 对 patch 内多个 mask 按 SAM3 score 进行置信度融合。
6. 对重叠 patch 的预测按类别累计置信度，得到整图预测。
7. 将预测与标签在非 ignore 像素上计算指标。
8. 保存预测图、GT 可视化、对比图、单图指标和数据集总体指标。

## 5. 指标定义

所有指标均在 `ground_truth != 255` 的有效像素上计算。

### 5.1 Overall Accuracy

Overall Accuracy，简称 OA，表示所有有效像素中预测正确的比例：

```text
OA = 正确分类像素数 / 有效像素总数
```

OA 直观反映整体像素准确率，但会受到大面积类别影响。Potsdam 中道路、建筑、植被面积较大，因此 OA 不应单独作为主结论。

### 5.2 Per-class IoU 与 Mean IoU

每类 IoU 定义为：

```text
IoU_c = TP_c / (TP_c + FP_c + FN_c)
```

Mean IoU 是 6 个 Potsdam 类别 IoU 的宏平均：

```text
mIoU = mean(IoU_0, IoU_1, IoU_2, IoU_3, IoU_4, IoU_5)
```

当前脚本将 `clutter/background` 也纳入 mIoU。论文中应明确说明 mIoU averaged over all six Potsdam classes, including clutter/background。

### 5.3 Frequency Weighted IoU

Frequency Weighted IoU，简称 FWIoU，按真实标签中各类别像素频率对 IoU 加权：

```text
FWIoU = sum(freq_c * IoU_c)
freq_c = GT 类别 c 的有效像素数 / 全部有效像素数
```

FWIoU 对大面积类别更敏感，可作为 mIoU 的补充。mIoU 更强调类别均衡，FWIoU 更接近像素面积加权整体质量。

### 5.4 Precision、Recall 与 F1

每类 Precision：

```text
Precision_c = TP_c / (TP_c + FP_c)
```

每类 Recall：

```text
Recall_c = TP_c / (TP_c + FN_c)
```

每类 F1：

```text
F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
```

Mean Precision、Mean Recall、Mean F1 均为 6 个类别的宏平均。

## 6. 推荐论文主结果

建议论文主表优先报告数据集级累计指标，即 `overall_metrics.json` 中的 `dataset_*` 字段：

| 指标字段 | 推荐用途 |
|---|---|
| `dataset_overall_accuracy` | 整体像素准确率 |
| `dataset_mean_iou` | 主分割指标 |
| `dataset_frequency_weighted_iou` | 面积加权 IoU 补充指标 |
| `dataset_mean_f1` | 类别均衡 F1 |
| `dataset_mean_precision` | 宏平均精确率 |
| `dataset_mean_recall` | 宏平均召回率 |
| `dataset_iou_per_class` | 各类别 IoU |
| `dataset_f1_per_class` | 各类别 F1 |
| `dataset_confusion_matrix` | 混淆矩阵 |

逐图平均指标 `average_*` 可作为补充分析，不建议作为论文主结果。

## 7. 输出文件说明

完整运行后，结果默认保存到 `results_exp1_sam3_potsdam_zeroshot_baseline/`：

| 输出路径 | 内容 |
|---|---|
| `results_exp1_sam3_potsdam_zeroshot_baseline/predictions/*_prediction.png` | 每张图的预测标签彩色图 |
| `results_exp1_sam3_potsdam_zeroshot_baseline/predictions/*_ground_truth.png` | 每张图的 GT 标签彩色图 |
| `results_exp1_sam3_potsdam_zeroshot_baseline/visualizations/*_comparison.png` | 原图、GT、预测、叠加图对比 |
| `results_exp1_sam3_potsdam_zeroshot_baseline/metrics/per_image_metrics.csv` | 每张图的指标 |
| `results_exp1_sam3_potsdam_zeroshot_baseline/metrics/overall_metrics.json` | 数据集级累计指标和逐图平均指标 |
| `results_exp1_sam3_potsdam_zeroshot_baseline/logs/evaluation_log_*.txt` | 运行日志 |

## 8. 结果解读注意事项

1. 本实验是零样本评估，不使用 Potsdam 训练集微调 SAM3。
2. 当前输入只使用 RGB，不使用 DSM、nDSM、IRRG 或 RGBIR。
3. noBoundary 标签中的黑色像素作为 ignore，不参与指标计算。
4. 没有被 SAM3 任一有效 mask 覆盖的像素回退为 `clutter/background`。
5. mIoU、mean F1、mean precision、mean recall 均为 6 类宏平均。
6. `dataset_*` 指标由 38 张图的混淆矩阵累加后计算，适合作为论文主结果。
7. `average_*` 指标是逐图指标的算术平均，适合分析不同图像之间的波动。

## 9. 论文表述建议

可以在论文实验设置中写：

```text
We evaluate SAM3 on the ISPRS Potsdam RGB orthophotos under a zero-shot setting. The input images are cropped into 1008 x 1008 patches with edge-aligned sliding windows. For each patch, SAM3 is prompted with the six Potsdam semantic categories. The patch-level masks are fused by SAM3 confidence scores, and overlapping patches are merged using confidence-weighted class accumulation. The evaluation is conducted on the noBoundary reference labels, where black boundary pixels are ignored. We report dataset-level OA, mIoU, FWIoU, mean F1, mean precision, mean recall, and per-class IoU/F1 over all six Potsdam classes.
```

如果论文使用中文表述，可以写：

```text
本文在零样本设置下评估 SAM3 在 ISPRS Potsdam RGB 正射影像上的语义分割能力。输入图像被切分为 1008 x 1008 的边缘对齐滑窗 patch，并分别使用 Potsdam 六个语义类别作为文本提示。patch 内 mask 根据 SAM3 置信度融合，重叠 patch 通过类别置信度累加方式合并。评估采用 noBoundary 标签，黑色边界像素作为 ignore，不参与指标计算。本文报告数据集级 OA、mIoU、FWIoU、mean F1、mean precision、mean recall 以及各类别 IoU/F1。
```

## 10. 结果填写模板

完整运行 `python exp1_sam3_potsdam_zeroshot_baseline.py` 后，可从 `results_exp1_sam3_potsdam_zeroshot_baseline/metrics/overall_metrics.json` 中填写下表：

| 指标 | 数值 |
|---|---:|
| OA | 待运行后填写 |
| mIoU | 待运行后填写 |
| FWIoU | 待运行后填写 |
| Mean F1 | 待运行后填写 |
| Mean Precision | 待运行后填写 |
| Mean Recall | 待运行后填写 |

各类别结果：

| 类别 | IoU | F1 |
|---|---:|---:|
| impervious surface | 待运行后填写 | 待运行后填写 |
| building | 待运行后填写 | 待运行后填写 |
| low vegetation | 待运行后填写 | 待运行后填写 |
| tree | 待运行后填写 | 待运行后填写 |
| car | 待运行后填写 | 待运行后填写 |
| clutter/background | 待运行后填写 | 待运行后填写 |
