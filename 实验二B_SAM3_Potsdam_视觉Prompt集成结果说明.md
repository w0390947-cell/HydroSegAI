# 实验二B：SAM3 + Potsdam 视觉 Prompt 集成结果说明

## 1. 实验目的

本实验用于分析 prompt 构造方式对 SAM3 + Potsdam 零样本语义分割结果的影响。

实验一直接使用 Potsdam 官方类别名作为文本提示；Exp2-A 去掉了 `clutter/background` 主动 prompt，但结果与实验一完全一致。Exp2-B 在 Exp2-A 基础上继续推进：

```text
对 5 个前景类别使用多个更视觉化的 prompt 变体。
```

实验希望回答的问题是：

```text
将遥感类别名转换为更接近日常视觉概念的文本提示，是否能提升 SAM3 的候选 mask 覆盖和最终语义分割结果？
```

## 2. 数据设置

当前脚本默认使用以下数据：

| 项目 | 路径 | 说明 |
|---|---|---|
| 输入影像 | `Potsdam/2_Ortho_RGB/2_Ortho_RGB` | RGB 真彩色正射影像 |
| 参考标签 | `Potsdam/5_Labels_all_noBoundary` | 38 张 noBoundary 语义标签 |

脚本通过配置自动发现同时存在 RGB 图像和 noBoundary 标签的样本。当前项目中可配对样本数为 38 张，本次运行 38 张全部成功处理。

## 3. 类别定义与 Prompt 策略

评估仍使用 Potsdam 标准 6 类语义定义：

| 类别 ID | 类别名称 | 英文名称 | 标签颜色 | Exp2-B prompt 策略 |
|---:|---|---|---|---|
| 0 | 不透水表面 | impervious surface | 白色 `(255,255,255)` | 5 个 prompt 变体 |
| 1 | 建筑物 | building | 蓝色 `(0,0,255)` | 5 个 prompt 变体 |
| 2 | 低矮植被 | low vegetation | 青色 `(0,255,255)` | 5 个 prompt 变体 |
| 3 | 树木 | tree | 绿色 `(0,255,0)` | 5 个 prompt 变体 |
| 4 | 汽车 | car | 黄色 `(255,255,0)` | 5 个 prompt 变体 |
| 5 | 杂乱/背景 | clutter/background | 红色 `(255,0,0)` | 不主动 prompt，仅作为 fallback |
| 255 | 忽略区域 | ignore/don't care | 黑色 `(0,0,0)` | 不参与指标计算 |

其中 `255` 只用于 noBoundary 标签中的边界或忽略区域，不参与指标计算。

## 4. Prompt Ensemble 配置

本次实验使用 5 个前景类别，每类 5 个 prompt 变体。因此每个 patch 共调用：

```text
5 classes * 5 prompts = 25 prompt calls
```

具体 prompt 如下：

| 类别 | Prompt variants |
|---|---|
| impervious surface | `impervious surface`, `road and pavement`, `asphalt road`, `paved area`, `parking lot` |
| building | `building`, `buildings`, `rooftop`, `building roof`, `house roof` |
| low vegetation | `low vegetation`, `grass`, `lawn`, `low plants`, `ground vegetation` |
| tree | `tree`, `trees`, `tree crown`, `tall tree`, `urban trees` |
| car | `car`, `cars`, `vehicle`, `small vehicle`, `parked car` |

所有 prompt 变体输出的 mask 都映射回对应 Potsdam 类别。例如 `grass`、`lawn`、`ground vegetation` 都映射回 `low vegetation`。

## 5. 推理流程

1. 读取 RGB 图像和 noBoundary 标签。
2. 将 RGB 大图切成 `1008 x 1008` patch。
3. 使用边缘对齐切片，避免图像右侧和下侧出现大面积补零 patch。
4. 对每个 patch 使用 25 个前景 prompt 调用 SAM3。
5. 不对 `clutter/background` 主动调用 SAM3。
6. 将同一类别下不同 prompt 变体产生的 mask 映射回同一个类别 ID。
7. 对 patch 内多个 mask 按 SAM3 score 进行置信度融合。
8. 未被任何有效 mask 覆盖的像素回退为 `clutter/background`。
9. 对重叠 patch 的预测按类别累计置信度，得到整图预测。
10. 将预测与标签在非 ignore 像素上计算指标。
11. 保存预测图、GT 可视化、对比图、单图指标和数据集总体指标。

## 6. 实验元数据

本次实验的元数据保存在：

```text
results_exp2b_sam3_potsdam_visual_prompt_ensemble/metrics/experiment_metadata.json
```

核心字段如下：

| 字段 | 内容 |
|---|---|
| `experiment_name` | `Exp2-B: SAM3 Potsdam visual prompt ensemble` |
| `variable_under_test` | `visual prompt ensemble without active clutter/background prompt` |
| `prompt_class_ids` | `[0, 1, 2, 3, 4]` |
| `excluded_prompt_class_ids` | `[5]` |
| `fallback_class_id` | `5` |
| `fallback_class_name` | `clutter/background` |
| `prompt_calls_per_patch` | `25` |

## 7. 指标定义

所有指标均在 `ground_truth != 255` 的有效像素上计算。

### 7.1 Overall Accuracy

Overall Accuracy，简称 OA，表示所有有效像素中预测正确的比例：

```text
OA = 正确分类像素数 / 有效像素总数
```

### 7.2 Per-class IoU 与 Mean IoU

每类 IoU 定义为：

```text
IoU_c = TP_c / (TP_c + FP_c + FN_c)
```

Mean IoU 是 6 个 Potsdam 类别 IoU 的宏平均：

```text
mIoU = mean(IoU_0, IoU_1, IoU_2, IoU_3, IoU_4, IoU_5)
```

虽然 Exp2-B 不主动 prompt `clutter/background`，但该类仍然参与 mIoU 计算。

### 7.3 Frequency Weighted IoU

Frequency Weighted IoU，简称 FWIoU，按真实标签中各类别像素频率对 IoU 加权：

```text
FWIoU = sum(freq_c * IoU_c)
freq_c = GT 类别 c 的有效像素数 / 全部有效像素数
```

### 7.4 Precision、Recall 与 F1

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

## 8. 推荐论文主结果

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

## 9. 输出文件说明

完整运行后，结果默认保存到 `results_exp2b_sam3_potsdam_visual_prompt_ensemble/`：

| 输出路径 | 内容 |
|---|---|
| `results_exp2b_sam3_potsdam_visual_prompt_ensemble/predictions/*_prediction.png` | 每张图的预测标签彩色图 |
| `results_exp2b_sam3_potsdam_visual_prompt_ensemble/predictions/*_ground_truth.png` | 每张图的 GT 标签彩色图 |
| `results_exp2b_sam3_potsdam_visual_prompt_ensemble/visualizations/*_comparison.png` | 原图、GT、预测、叠加图对比 |
| `results_exp2b_sam3_potsdam_visual_prompt_ensemble/metrics/per_image_metrics.csv` | 每张图的指标 |
| `results_exp2b_sam3_potsdam_visual_prompt_ensemble/metrics/overall_metrics.json` | 数据集级累计指标和逐图平均指标 |
| `results_exp2b_sam3_potsdam_visual_prompt_ensemble/metrics/experiment_metadata.json` | Exp2-B 实验变量和 prompt ensemble |
| `results_exp2b_sam3_potsdam_visual_prompt_ensemble/metrics/config_snapshot.yaml` | 本次运行使用的配置快照 |
| `results_exp2b_sam3_potsdam_visual_prompt_ensemble/metrics/run_context.json` | 自动发现样本、成功处理样本和跳过/失败样本记录 |
| `results_exp2b_sam3_potsdam_visual_prompt_ensemble/logs/evaluation_log_*.txt` | 运行日志 |

本次运行共生成：

| 输出类型 | 数量 | 说明 |
|---|---:|---|
| predictions | 152 | 38 张图，每张包含彩色预测、彩色 GT、预测 ID、GT ID |
| visualizations | 38 | 每张图 1 张对比可视化 |
| metrics | 43 | 38 个单图 JSON + 5 个总体/配置/元数据文件 |

当前配置中 `metrics.save_confusion_matrix=false`，因此没有单独生成 `dataset_confusion_matrix.csv`；数据集级混淆矩阵保存在 `overall_metrics.json` 的 `dataset_confusion_matrix` 字段中。

## 10. 结果解读注意事项

1. 本实验是零样本评估，不使用 Potsdam 训练集微调 SAM3。
2. 当前输入只使用 RGB，不使用 DSM、nDSM、IRRG 或 RGBIR。
3. noBoundary 标签中的黑色像素作为 ignore，不参与指标计算。
4. `clutter/background` 不主动输入 SAM3，但仍作为评估类别。
5. 没有被 SAM3 任一有效前景 mask 覆盖的像素回退为 `clutter/background`。
6. 同一类别的多个 prompt 变体输出均映射回同一个类别 ID。
7. mIoU、mean F1、mean precision、mean recall 均为 6 类宏平均。
8. `dataset_*` 指标由 38 张图的混淆矩阵累加后计算，适合作为论文主结果。
9. `average_*` 指标是逐图指标的算术平均，适合分析不同图像之间的波动。
10. `top_potsdam_4_12_RGB` 的 noBoundary 标签有效像素为 0，不影响数据集级累计指标，但会影响逐图平均。

## 11. 论文表述建议

可以在论文实验设置中写：

```text
Exp2-B evaluates a visual prompt ensemble strategy for the five foreground Potsdam classes. Each foreground category is queried with five text variants, and all masks generated by variants of the same category are mapped back to the original Potsdam class. The clutter/background category is not actively prompted and is only used as a residual fallback class. All other settings are identical to Exp2-A.
```

如果论文使用中文表述，可以写：

```text
Exp2-B 用于评估前景类别视觉化 prompt ensemble 策略。每个前景类别使用 5 个文本变体进行查询，同一类别下不同 prompt 生成的 mask 均映射回原始 Potsdam 类别。clutter/background 不作为主动 prompt 类别，仅作为未覆盖像素的 residual fallback 类别。除 prompt 构造方式外，其余设置均与 Exp2-A 保持一致。
```

## 12. 本次实验结果

本次 `exp2b_sam3_potsdam_visual_prompt_ensemble.py` 已完成全量运行，38 张自动发现样本全部成功处理。其中 `top_potsdam_4_12_RGB` 的 noBoundary 标签有效像素为 0，所有像素均为 ignore；它不影响 `dataset_*` 累计混淆矩阵指标，但会进入 `average_*` 逐图平均。

### 12.1 数据集级主结果

论文主表建议使用 `overall_metrics.json` 中的 `dataset_*` 指标：

| 指标 | 数值 |
|---|---:|
| OA | 0.1659 |
| mIoU | 0.1052 |
| FWIoU | 0.1056 |
| Mean F1 | 0.1806 |
| Mean Precision | 0.6647 |
| Mean Recall | 0.2231 |

逐图平均指标仅作为补充分析：

| 指标 | 数值 |
|---|---:|
| Average OA | 0.1611 |
| Average mIoU | 0.1031 |
| Average Mean F1 | 0.1715 |
| Average Mean Precision | 0.4793 |
| Average Mean Recall | 0.2323 |
| Average FWIoU | 0.1065 |

### 12.2 各类别结果

| 类别 | GT 占比 | 预测占比 | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 29.41% | 24.08% | 0.2014 | 0.3353 | 0.3724 | 0.3049 |
| building | 26.54% | 1.27% | 0.0458 | 0.0875 | 0.9614 | 0.0458 |
| low vegetation | 22.31% | 4.96% | 0.1310 | 0.2316 | 0.6364 | 0.1415 |
| tree | 15.66% | 0.00% | 0.0002 | 0.0004 | 0.9965 | 0.0002 |
| car | 1.41% | 0.31% | 0.2114 | 0.3490 | 0.9792 | 0.2124 |
| clutter/background | 4.65% | 69.38% | 0.0415 | 0.0796 | 0.0425 | 0.6336 |

### 12.3 与 Exp1 / Exp2-A 的差异

| 实验 | 主动 prompt 类别数 | Prompt 调用数 | Valid masks | Empty prompt outputs | mIoU | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|
| Exp1：六类官方 prompt | 6 | 18468 | 6823 | 16645 | 0.0578 | 0.1020 |
| Exp2-A：无背景 prompt | 5 | 15390 | 6823 | 13567 | 0.0578 | 0.1020 |
| Exp2-B：视觉 prompt ensemble | 25 | 76950 | 23066 | 70628 | 0.1052 | 0.1806 |

Exp2-B 相比 Exp1 的 mIoU 提升为 +0.0474，Mean F1 提升为 +0.0786。有效 mask 数量从 6823 增加到 23066，说明视觉化 prompt 变体显著改善了前景候选 mask 的覆盖。

### 12.4 结果简要解读

本次实验表明，视觉化 prompt ensemble 对 SAM3 的 Potsdam 零样本分割有明显帮助。尤其是 `low vegetation` 从几乎无法识别提升到 IoU 0.1310，说明 `grass`、`lawn`、`low plants`、`ground vegetation` 等提示比单一官方术语更容易被 SAM3 理解。

但结果仍然不足以支撑可靠的密集语义分割。预测结果中 `clutter/background` 仍占 69.38%，远高于真实标签中的 4.65%；`building` 和 `tree` 的 recall 仍然很低。该实验说明 prompt 工程有效，但还需要后续 mask 过滤、多模态输入和遥感先验约束共同改善结果。
