# 实验二C：SAM3 + Potsdam Prompt 集成与 Mask 过滤结果说明

## 1. 实验目的

本实验用于分析在 SAM3 + Potsdam 零样本语义分割中，引入固定的候选 mask 面积过滤规则后，是否能够进一步改善视觉化 prompt ensemble 的结果。

Exp2-B 已经证明：

```text
对 5 个前景类别使用视觉化 prompt ensemble，可以显著增加有效候选 mask 数量，并提升零样本分割指标。
```

Exp2-C 在 Exp2-B 基础上只增加一个变量：

```text
在 patch 级置信度融合之前，对 SAM3 生成的候选 mask 按类别面积比例进行固定阈值过滤。
```

实验希望回答的问题是：

```text
使用类别相关的 mask 尺度先验，能否过滤明显异常的过大或过小候选 mask，并进一步提升 Potsdam 零样本语义分割结果？
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

| 类别 ID | 类别名称 | 英文名称 | 标签颜色 | Exp2-C prompt 策略 |
|---:|---|---|---|---|
| 0 | 不透水表面 | impervious surface | 白色 `(255,255,255)` | 5 个 prompt 变体 + 面积过滤 |
| 1 | 建筑物 | building | 蓝色 `(0,0,255)` | 5 个 prompt 变体 + 面积过滤 |
| 2 | 低矮植被 | low vegetation | 青色 `(0,255,255)` | 5 个 prompt 变体 + 面积过滤 |
| 3 | 树木 | tree | 绿色 `(0,255,0)` | 5 个 prompt 变体 + 面积过滤 |
| 4 | 汽车 | car | 黄色 `(255,255,0)` | 5 个 prompt 变体 + 面积过滤 |
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

## 5. Mask 过滤规则

Exp2-C 在 Exp2-B 的候选 mask 生成之后、patch 级置信度融合之前加入面积过滤。面积比例定义为：

```text
mask_area_ratio = mask 非零像素数 / patch 像素总数
```

本次实验使用固定类别阈值：

| 类别 | 最小面积比例 | 最大面积比例 |
|---|---:|---:|
| impervious surface | 0.001 | 0.80 |
| building | 0.0005 | 0.60 |
| low vegetation | 0.001 | 0.80 |
| tree | 0.0003 | 0.50 |
| car | 0.00001 | 0.05 |

这些阈值作为 Potsdam 高分辨率影像中各类常见尺度的先验，不针对单张测试图像调参。

## 6. 推理流程

1. 读取 RGB 图像和 noBoundary 标签。
2. 将 RGB 大图切成 `1008 x 1008` patch。
3. 使用边缘对齐切片，避免图像右侧和下侧出现大面积补零 patch。
4. 对每个 patch 使用 25 个前景 prompt 调用 SAM3。
5. 不对 `clutter/background` 主动调用 SAM3。
6. 将同一类别下不同 prompt 变体产生的 mask 映射回同一个类别 ID。
7. 对候选 mask 应用类别相关面积比例过滤。
8. 对过滤后的 patch 内多个 mask 按 SAM3 score 进行置信度融合。
9. 未被任何有效 mask 覆盖的像素回退为 `clutter/background`。
10. 对重叠 patch 的预测按类别累计置信度，得到整图预测。
11. 将预测与标签在非 ignore 像素上计算指标。
12. 保存预测图、GT 可视化、对比图、单图指标和数据集总体指标。

## 7. 实验元数据

本次实验的元数据保存在：

```text
results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/metrics/experiment_metadata.json
```

核心字段如下：

| 字段 | 内容 |
|---|---|
| `experiment_name` | `Exp2-C: SAM3 Potsdam prompt ensemble with mask filter` |
| `variable_under_test` | `visual prompt ensemble with class-aware mask area filtering` |
| `prompt_class_ids` | `[0, 1, 2, 3, 4]` |
| `excluded_prompt_class_ids` | `[5]` |
| `fallback_class_id` | `5` |
| `fallback_class_name` | `clutter/background` |
| `score_threshold` | `0.5` |
| `prompt_calls_per_patch` | `25` |
| `mask_filter_enabled` | `true` |

## 8. Mask 过滤统计

本次运行中，SAM3 共生成 23066 个有效候选 mask；面积过滤后保留 22861 个，过滤 205 个，过滤比例约为 0.89%。

| 类别 | 输入 mask | 保留 mask | 过滤过小 | 过滤过大 | 过滤比例 |
|---|---:|---:|---:|---:|---:|
| impervious surface | 1506 | 1330 | 0 | 176 | 11.69% |
| building | 465 | 465 | 0 | 0 | 0.00% |
| low vegetation | 1267 | 1238 | 17 | 12 | 2.29% |
| tree | 6 | 6 | 0 | 0 | 0.00% |
| car | 19822 | 19822 | 0 | 0 | 0.00% |
| 合计 | 23066 | 22861 | 17 | 188 | 0.89% |

过滤主要发生在 `impervious surface` 的大面积 mask 上，其次是少量 `low vegetation` 过小或过大 mask。`building`、`tree`、`car` 在当前阈值下没有候选 mask 被过滤。

## 9. 指标定义

所有指标均在 `ground_truth != 255` 的有效像素上计算。

### 9.1 Overall Accuracy

Overall Accuracy，简称 OA，表示所有有效像素中预测正确的比例：

```text
OA = 正确分类像素数 / 有效像素总数
```

### 9.2 Per-class IoU 与 Mean IoU

每类 IoU 定义为：

```text
IoU_c = TP_c / (TP_c + FP_c + FN_c)
```

Mean IoU 是 6 个 Potsdam 类别 IoU 的宏平均：

```text
mIoU = mean(IoU_0, IoU_1, IoU_2, IoU_3, IoU_4, IoU_5)
```

虽然 Exp2-C 不主动 prompt `clutter/background`，但该类仍然参与 mIoU 计算。

### 9.3 Frequency Weighted IoU

Frequency Weighted IoU，简称 FWIoU，按真实标签中各类别像素频率对 IoU 加权：

```text
FWIoU = sum(freq_c * IoU_c)
freq_c = GT 类别 c 的有效像素数 / 全部有效像素数
```

### 9.4 Precision、Recall 与 F1

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

## 10. 推荐论文主结果

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

## 11. 输出文件说明

完整运行后，结果默认保存到 `results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/`：

| 输出路径 | 内容 |
|---|---|
| `results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/predictions/*_prediction.png` | 每张图的预测标签彩色图 |
| `results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/predictions/*_ground_truth.png` | 每张图的 GT 标签彩色图 |
| `results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/visualizations/*_comparison.png` | 原图、GT、预测、叠加图对比 |
| `results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/metrics/per_image_metrics.csv` | 每张图的指标 |
| `results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/metrics/overall_metrics.json` | 数据集级累计指标和逐图平均指标 |
| `results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/metrics/experiment_metadata.json` | Exp2-C 实验变量、prompt ensemble 和 mask 过滤统计 |
| `results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/metrics/config_snapshot.yaml` | 本次运行使用的配置快照 |
| `results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/metrics/run_context.json` | 自动发现样本、成功处理样本和跳过/失败样本记录 |
| `results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/logs/evaluation_log_*.txt` | 运行日志 |

本次运行共生成：

| 输出类型 | 数量 | 说明 |
|---|---:|---|
| predictions | 152 | 38 张图，每张包含彩色预测、彩色 GT、预测 ID、GT ID |
| visualizations | 38 | 每张图 1 张对比可视化 |
| metrics | 43 | 38 个单图 JSON + 5 个总体/配置/元数据文件 |
| logs | 1 | 本次运行日志 |

当前配置中 `metrics.save_confusion_matrix=false`，因此没有单独生成 `dataset_confusion_matrix.csv`；数据集级混淆矩阵保存在 `overall_metrics.json` 的 `dataset_confusion_matrix` 字段中。

## 12. 结果解读注意事项

1. 本实验是零样本评估，不使用 Potsdam 训练集微调 SAM3。
2. 当前输入只使用 RGB，不使用 DSM、nDSM、IRRG 或 RGBIR。
3. noBoundary 标签中的黑色像素作为 ignore，不参与指标计算。
4. `clutter/background` 不主动输入 SAM3，但仍作为评估类别。
5. 没有被 SAM3 任一有效前景 mask 覆盖的像素回退为 `clutter/background`。
6. 同一类别的多个 prompt 变体输出均映射回同一个类别 ID。
7. Exp2-C 的面积过滤发生在 patch 级融合之前。
8. mIoU、mean F1、mean precision、mean recall 均为 6 类宏平均。
9. `dataset_*` 指标由 38 张图的混淆矩阵累加后计算，适合作为论文主结果。
10. `average_*` 指标是逐图指标的算术平均，适合分析不同图像之间的波动。
11. `top_potsdam_4_12_RGB` 的 noBoundary 标签有效像素为 0，不影响数据集级累计指标，但会影响逐图平均。

## 13. 论文表述建议

可以在论文实验设置中写：

```text
Exp2-C evaluates a class-aware mask area filtering strategy on top of the visual prompt ensemble. SAM3 is queried with the same 25 foreground prompt variants as Exp2-B, while clutter/background is still used only as the residual fallback class. Before confidence-based patch fusion, candidate masks are filtered by fixed class-specific area-ratio thresholds to remove masks with implausible spatial scales. The thresholds are fixed a priori and are not tuned for individual test images.
```

如果论文使用中文表述，可以写：

```text
Exp2-C 在视觉化 prompt ensemble 的基础上评估类别相关的候选 mask 面积过滤策略。该实验沿用 Exp2-B 的 25 个前景 prompt 变体，并继续将 clutter/background 仅作为 residual fallback 类别。在基于置信度的 patch 融合之前，候选 mask 会根据固定的类别面积比例阈值进行过滤，以剔除空间尺度明显异常的候选区域。所有阈值预先固定，不针对单张测试图像调参。
```

