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
| `results_exp1_sam3_potsdam_zeroshot_baseline/metrics/config_snapshot.yaml` | 本次运行使用的配置快照 |
| `results_exp1_sam3_potsdam_zeroshot_baseline/metrics/run_context.json` | 自动发现样本、成功处理样本和跳过/失败样本记录 |
| `results_exp1_sam3_potsdam_zeroshot_baseline/logs/evaluation_log_*.txt` | 运行日志 |

本次运行共生成：

| 输出类型 | 数量 | 说明 |
|---|---:|---|
| predictions | 152 | 38 张图，每张包含彩色预测、彩色 GT、预测 ID、GT ID |
| visualizations | 38 | 每张图 1 张对比可视化 |
| metrics | 42 | 38 个单图 JSON + 4 个总体/配置文件 |

当前配置中 `metrics.save_confusion_matrix=false`，因此没有单独生成 `dataset_confusion_matrix.csv`；数据集级混淆矩阵保存在 `overall_metrics.json` 的 `dataset_confusion_matrix` 字段中。

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

## 10. 本次实验结果

本次 `exp1_sam3_potsdam_zeroshot_baseline.py` 已完成全量运行，38 张自动发现样本全部成功处理。其中 `top_potsdam_4_12_RGB` 的 noBoundary 标签有效像素为 0，所有像素均为 ignore；它不影响 `dataset_*` 累计混淆矩阵指标，但会进入 `average_*` 逐图平均。

### 10.1 数据集级主结果

论文主表建议使用 `overall_metrics.json` 中的 `dataset_*` 指标：

| 指标 | 数值 |
|---|---:|
| OA | 0.0892 |
| mIoU | 0.0578 |
| FWIoU | 0.0426 |
| Mean F1 | 0.1020 |
| Mean Precision | 0.5776 |
| Mean Recall | 0.1702 |

逐图平均指标仅作为补充分析：

| 指标 | 数值 |
|---|---:|
| Average OA | 0.0865 |
| Average mIoU | 0.0559 |
| Average Mean F1 | 0.0948 |

### 10.2 各类别结果

| 类别 | GT 占比 | 预测占比 | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 29.41% | 20.47% | 0.1197 | 0.2138 | 0.2605 | 0.1813 |
| building | 26.54% | 0.31% | 0.0112 | 0.0221 | 0.9656 | 0.0112 |
| low vegetation | 22.31% | 0.07% | 0.0007 | 0.0014 | 0.2231 | 0.0007 |
| tree | 15.66% | 0.00% | 0.0001 | 0.0002 | 0.9955 | 0.0001 |
| car | 1.41% | 0.26% | 0.1778 | 0.3019 | 0.9823 | 0.1784 |
| clutter/background | 4.65% | 78.89% | 0.0375 | 0.0723 | 0.0383 | 0.6495 |

### 10.3 结果简要解读

本次实验表明，直接使用 SAM3 和 Potsdam 官方类别名 prompt 进行 RGB 零样本密集语义分割效果较弱。数据集级 mIoU 为 0.0578，Mean F1 为 0.1020，说明模型很难稳定覆盖 Potsdam 的六类像素级语义。

主要失败模式是过度回退到 `clutter/background`。真实标签中 `clutter/background` 只占 4.65%，但预测结果中该类占 78.89%。这意味着大量不透水表面、建筑物、低矮植被、树木和车辆像素没有被 SAM3 的有效 mask 覆盖，最终被归入 fallback 背景类。

从类别指标看，`car` 的 IoU 相对最高，为 0.1778；`impervious surface` 次之，为 0.1197。`building`、`low vegetation` 和 `tree` 的 recall 极低，说明模型即使在少数预测上 precision 较高，也没有召回大多数真实目标。因此论文中应将本实验定位为负基线或动机实验，用于说明通用开放词汇分割模型直接迁移到遥感密集语义分割存在明显不足。
