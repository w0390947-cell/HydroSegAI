# 实验二A：SAM3 + Potsdam 无背景 Prompt 消融结果说明

## 1. 实验目的

本实验用于分析在 SAM3 + Potsdam 零样本语义分割中，是否应该主动向模型输入 `clutter/background` 文本 prompt。

实验一使用 Potsdam 六个官方类别作为文本提示，包括：

```text
impervious surface
building
low vegetation
tree
car
background clutter
```

Exp2-A 在实验一基础上只改变一个变量：

```text
不主动输入 background clutter / clutter-background prompt。
```

也就是说，SAM3 只接收前 5 个前景类别 prompt；`clutter/background` 仍然保留为评估类别，并作为未被任何有效前景 mask 覆盖像素的 fallback 类别。

## 2. 数据设置

当前脚本默认使用以下数据：

| 项目 | 路径 | 说明 |
|---|---|---|
| 输入影像 | `Potsdam/2_Ortho_RGB/2_Ortho_RGB` | RGB 真彩色正射影像 |
| 参考标签 | `Potsdam/5_Labels_all_noBoundary` | 38 张 noBoundary 语义标签 |

脚本通过配置自动发现同时存在 RGB 图像和 noBoundary 标签的样本。当前项目中可配对样本数为 38 张，本次运行 38 张全部成功处理。

## 3. 类别定义

评估仍使用 Potsdam 标准 6 类语义定义：

| 类别 ID | 类别名称 | 英文名称 | 标签颜色 | Exp2-A prompt 策略 |
|---:|---|---|---|---|
| 0 | 不透水表面 | impervious surface | 白色 `(255,255,255)` | 主动 prompt |
| 1 | 建筑物 | building | 蓝色 `(0,0,255)` | 主动 prompt |
| 2 | 低矮植被 | low vegetation | 青色 `(0,255,255)` | 主动 prompt |
| 3 | 树木 | tree | 绿色 `(0,255,0)` | 主动 prompt |
| 4 | 汽车 | car | 黄色 `(255,255,0)` | 主动 prompt |
| 5 | 杂乱/背景 | clutter/background | 红色 `(255,0,0)` | 不主动 prompt，仅作为 fallback |
| 255 | 忽略区域 | ignore/don't care | 黑色 `(0,0,0)` | 不参与指标计算 |

其中 `255` 只用于 noBoundary 标签中的边界或忽略区域，不参与指标计算。

## 4. 推理流程

1. 读取 RGB 图像和 noBoundary 标签。
2. 将 RGB 大图切成 `1008 x 1008` patch。
3. 使用边缘对齐切片，避免图像右侧和下侧出现大面积补零 patch。
4. 对每个 patch 只使用 5 个前景文本提示调用 SAM3。
5. 不对 `clutter/background` 主动调用 SAM3。
6. 对 patch 内多个 mask 按 SAM3 score 进行置信度融合。
7. 未被任何有效 mask 覆盖的像素回退为 `clutter/background`。
8. 对重叠 patch 的预测按类别累计置信度，得到整图预测。
9. 将预测与标签在非 ignore 像素上计算指标。
10. 保存预测图、GT 可视化、对比图、单图指标和数据集总体指标。

## 5. 实验元数据

本次实验的元数据保存在：

```text
results_exp2a_sam3_potsdam_no_background_prompt/metrics/experiment_metadata.json
```

核心字段如下：

| 字段 | 内容 |
|---|---|
| `experiment_name` | `Exp2-A: SAM3 Potsdam no background prompt` |
| `variable_under_test` | `remove active clutter/background text prompt` |
| `prompt_class_ids` | `[0, 1, 2, 3, 4]` |
| `excluded_prompt_class_ids` | `[5]` |
| `fallback_class_id` | `5` |
| `fallback_class_name` | `clutter/background` |

## 6. 指标定义

所有指标均在 `ground_truth != 255` 的有效像素上计算。

### 6.1 Overall Accuracy

Overall Accuracy，简称 OA，表示所有有效像素中预测正确的比例：

```text
OA = 正确分类像素数 / 有效像素总数
```

### 6.2 Per-class IoU 与 Mean IoU

每类 IoU 定义为：

```text
IoU_c = TP_c / (TP_c + FP_c + FN_c)
```

Mean IoU 是 6 个 Potsdam 类别 IoU 的宏平均：

```text
mIoU = mean(IoU_0, IoU_1, IoU_2, IoU_3, IoU_4, IoU_5)
```

虽然 Exp2-A 不主动 prompt `clutter/background`，但该类仍然参与 mIoU 计算。

### 6.3 Frequency Weighted IoU

Frequency Weighted IoU，简称 FWIoU，按真实标签中各类别像素频率对 IoU 加权：

```text
FWIoU = sum(freq_c * IoU_c)
freq_c = GT 类别 c 的有效像素数 / 全部有效像素数
```

### 6.4 Precision、Recall 与 F1

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

## 7. 推荐论文主结果

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

## 8. 输出文件说明

完整运行后，结果默认保存到 `results_exp2a_sam3_potsdam_no_background_prompt/`：

| 输出路径 | 内容 |
|---|---|
| `results_exp2a_sam3_potsdam_no_background_prompt/predictions/*_prediction.png` | 每张图的预测标签彩色图 |
| `results_exp2a_sam3_potsdam_no_background_prompt/predictions/*_ground_truth.png` | 每张图的 GT 标签彩色图 |
| `results_exp2a_sam3_potsdam_no_background_prompt/visualizations/*_comparison.png` | 原图、GT、预测、叠加图对比 |
| `results_exp2a_sam3_potsdam_no_background_prompt/metrics/per_image_metrics.csv` | 每张图的指标 |
| `results_exp2a_sam3_potsdam_no_background_prompt/metrics/overall_metrics.json` | 数据集级累计指标和逐图平均指标 |
| `results_exp2a_sam3_potsdam_no_background_prompt/metrics/experiment_metadata.json` | Exp2-A 实验变量和 prompt 策略 |
| `results_exp2a_sam3_potsdam_no_background_prompt/metrics/config_snapshot.yaml` | 本次运行使用的配置快照 |
| `results_exp2a_sam3_potsdam_no_background_prompt/metrics/run_context.json` | 自动发现样本、成功处理样本和跳过/失败样本记录 |
| `results_exp2a_sam3_potsdam_no_background_prompt/logs/evaluation_log_*.txt` | 运行日志 |

本次运行共生成：

| 输出类型 | 数量 | 说明 |
|---|---:|---|
| predictions | 152 | 38 张图，每张包含彩色预测、彩色 GT、预测 ID、GT ID |
| visualizations | 38 | 每张图 1 张对比可视化 |
| metrics | 43 | 38 个单图 JSON + 5 个总体/配置/元数据文件 |

当前配置中 `metrics.save_confusion_matrix=false`，因此没有单独生成 `dataset_confusion_matrix.csv`；数据集级混淆矩阵保存在 `overall_metrics.json` 的 `dataset_confusion_matrix` 字段中。

## 9. 结果解读注意事项

1. 本实验是零样本评估，不使用 Potsdam 训练集微调 SAM3。
2. 当前输入只使用 RGB，不使用 DSM、nDSM、IRRG 或 RGBIR。
3. noBoundary 标签中的黑色像素作为 ignore，不参与指标计算。
4. `clutter/background` 不主动输入 SAM3，但仍作为评估类别。
5. 没有被 SAM3 任一有效前景 mask 覆盖的像素回退为 `clutter/background`。
6. mIoU、mean F1、mean precision、mean recall 均为 6 类宏平均。
7. `dataset_*` 指标由 38 张图的混淆矩阵累加后计算，适合作为论文主结果。
8. `average_*` 指标是逐图指标的算术平均，适合分析不同图像之间的波动。
9. `top_potsdam_4_12_RGB` 的 noBoundary 标签有效像素为 0，不影响数据集级累计指标，但会影响逐图平均。

## 10. 论文表述建议

可以在论文实验设置中写：

```text
Exp2-A evaluates the effect of removing the active clutter/background text prompt. SAM3 is prompted only with the five foreground Potsdam categories, while clutter/background is retained as an evaluation category and used as the residual fallback class for pixels not covered by any valid foreground mask. All other settings, including RGB input, patch size, stride, confidence-based mask fusion, and noBoundary evaluation, are kept identical to the zero-shot baseline.
```

如果论文使用中文表述，可以写：

```text
Exp2-A 用于评估是否需要主动向 SAM3 输入 clutter/background 文本提示。该实验仅使用 Potsdam 五个前景类别作为文本 prompt，而将 clutter/background 保留为评估类别和未覆盖像素的 residual fallback 类别。除 prompt 策略外，RGB 输入、patch 尺寸、stride、置信度融合方式和 noBoundary 评估设置均与零样本基线保持一致。
```

## 11. 本次实验结果

本次 `exp2a_sam3_potsdam_no_background_prompt.py` 已完成全量运行，38 张自动发现样本全部成功处理。其中 `top_potsdam_4_12_RGB` 的 noBoundary 标签有效像素为 0，所有像素均为 ignore；它不影响 `dataset_*` 累计混淆矩阵指标，但会进入 `average_*` 逐图平均。

### 11.1 数据集级主结果

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
| Average Mean Precision | 0.3009 |
| Average Mean Recall | 0.1787 |
| Average FWIoU | 0.0449 |

### 11.2 各类别结果

| 类别 | GT 占比 | 预测占比 | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 29.41% | 20.47% | 0.1197 | 0.2138 | 0.2605 | 0.1813 |
| building | 26.54% | 0.31% | 0.0112 | 0.0221 | 0.9656 | 0.0112 |
| low vegetation | 22.31% | 0.07% | 0.0007 | 0.0014 | 0.2231 | 0.0007 |
| tree | 15.66% | 0.00% | 0.0001 | 0.0002 | 0.9955 | 0.0001 |
| car | 1.41% | 0.26% | 0.1778 | 0.3019 | 0.9823 | 0.1784 |
| clutter/background | 4.65% | 78.89% | 0.0375 | 0.0723 | 0.0383 | 0.6495 |

### 11.3 与 Exp1 的差异

| 实验 | 主动 prompt 类别数 | Prompt 调用数 | Valid masks | Empty prompt outputs | mIoU | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|
| Exp1：六类官方 prompt | 6 | 18468 | 6823 | 16645 | 0.0578 | 0.1020 |
| Exp2-A：无背景 prompt | 5 | 15390 | 6823 | 13567 | 0.0578 | 0.1020 |

Exp2-A 的 prompt 调用次数比 Exp1 少 3078 次，正好对应每个 patch 少调用一次背景 prompt：

```text
38 images * 81 patches = 3078 fewer prompt calls
```

但有效 mask 数量仍为 6823，数据集级指标也完全一致。这说明在当前设置下，`background clutter` prompt 没有产生有效分割贡献。

### 11.4 结果简要解读

本次实验表明，直接去掉 `clutter/background` 主动 prompt 并不会改善 SAM3 在 Potsdam RGB 零样本语义分割上的表现。数据集级 mIoU 仍为 0.0578，Mean F1 仍为 0.1020。

更重要的是，Exp2-A 排除了一个可能误解：背景过预测并不是因为主动输入了 `background clutter` prompt。即使不输入该 prompt，预测结果中 `clutter/background` 仍占 78.89%。因此，背景过预测主要来自前景 mask 覆盖不足，以及未覆盖像素统一 fallback 到背景类。

从实验设计角度看，Exp2-A 支持后续实验继续将 `clutter/background` 作为 residual 类别处理，而把改进重点转向前景 prompt ensemble、mask 质量过滤和多模态遥感先验。
