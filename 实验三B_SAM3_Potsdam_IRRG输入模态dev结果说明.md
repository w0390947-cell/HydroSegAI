# 实验三B：SAM3 + Potsdam IRRG 输入模态 dev 结果说明

## 1. 实验目的

本实验用于分析在固定当前较有效 prompt 策略后，IRRG 假彩色输入模态在 Potsdam development split 上的表现。

Exp3-A 已经建立了 RGB 参照组。Exp3-B 在此基础上只改变输入模态：

```text
RGB natural orthophoto input -> IRRG false-color orthophoto input
```

实验希望回答的问题是：

```text
在固定 Exp2-B 视觉 prompt ensemble 策略后，
IRRG 假彩色输入是否能利用近红外信息改善 SAM3 在 Potsdam 上的候选 mask 生成和语义分割结果？
```

尤其关注：

1. `low vegetation` 和 `tree` 是否从近红外信息中获益。
2. `building` 和 `car` 是否因假彩色外观偏离自然图像分布而下降。
3. 前景像素回退到 `clutter/background` 的比例是否降低。
4. IRRG 是否适合作为 RGB 的替代模态，还是更适合作为后续多视图融合的补充模态。

## 2. 数据设置

当前脚本默认使用以下数据：

| 项目 | 路径 | 说明 |
|---|---|---|
| 输入影像 | `Potsdam/3_Ortho_IRRG/3_Ortho_IRRG` | IRRG 假彩色正射影像 |
| 参考标签 | `Potsdam/5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary` | 24 张 participant noBoundary 标签 |
| 输出目录 | `results_exp3b_sam3_potsdam_irrg_modality_dev` | Exp3-B dev 结果目录 |

本实验使用 Exp3 预定义的 `dev` split，共 24 张图像：

```text
top_potsdam_2_10, top_potsdam_2_11, top_potsdam_2_12
top_potsdam_3_10, top_potsdam_3_11, top_potsdam_3_12
top_potsdam_4_10, top_potsdam_4_11, top_potsdam_4_12
top_potsdam_5_10, top_potsdam_5_11, top_potsdam_5_12
top_potsdam_6_7, top_potsdam_6_8, top_potsdam_6_9
top_potsdam_6_10, top_potsdam_6_11, top_potsdam_6_12
top_potsdam_7_7, top_potsdam_7_8, top_potsdam_7_9
top_potsdam_7_10, top_potsdam_7_11, top_potsdam_7_12
```

配置中通过以下字段控制划分：

```yaml
evaluation:
  split: "dev"
  test_images: []
```

当 `test_images` 为空时，脚本会根据 `split: "dev"` 自动使用上述 24 张 development 图像。

## 3. 类别定义

评估使用 Potsdam 标准 6 类语义定义：

| 类别 ID | 类别名称 | 英文名称 | 标签颜色 | Exp3-B prompt 策略 |
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

Exp3-B 固定使用 Exp2-B / Exp3-A 的 5 类前景视觉 prompt ensemble。每个 patch 共调用：

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

## 5. 输入模态设置

Exp3-B 的输入模态为 IRRG：

| 配置项 | 内容 |
|---|---|
| `input_modality.name` | `irrg` |
| `input_modality.description` | `Infrared-red-green orthophoto input.` |
| 图像目录 | `3_Ortho_IRRG/3_Ortho_IRRG` |
| 图像文件模板 | `{image_id}_IRRG.tif` |
| 标签文件模板 | `{image_id}_label_noBoundary.tif` |

本实验不使用以下信息：

```text
RGB 真彩色影像
RGBIR 四波段影像
NDVI 或其他派生指数
DSM / nDSM 高程信息
多视图融合
```

因此，Exp3-B 是 IRRG 作为单独三通道输入替代 RGB 的模态消融。

## 6. 推理流程

1. 根据 `evaluation.split=dev` 读取 24 张 development 图像 ID。
2. 读取 IRRG 影像和 participant noBoundary 标签。
3. 将 IRRG 大图切成 `1008 x 1008` patch。
4. 使用 stride `672` 的边缘对齐滑窗，避免边缘区域遗漏。
5. 对每个 patch 使用 25 个前景 prompt 调用 SAM3。
6. 不对 `clutter/background` 主动调用 SAM3。
7. 将同一类别下不同 prompt 变体产生的 mask 映射回同一个类别 ID。
8. 不启用 class-aware mask area filtering。
9. 对 patch 内多个 mask 按 SAM3 score 进行置信度融合。
10. 未被任何有效前景 mask 覆盖的像素回退为 `clutter/background`。
11. 对重叠 patch 的预测按类别累计置信度，得到整图预测。
12. 将预测与标签在非 ignore 像素上计算指标。
13. 保存预测图、GT 图、对比可视化、单图指标和数据集总体指标。

## 7. 实验元数据

本次实验的元数据保存在：

```text
results_exp3b_sam3_potsdam_irrg_modality_dev/metrics/experiment_metadata.json
```

核心字段如下：

| 字段 | 内容 |
|---|---|
| `experiment_name` | `Exp3-B: SAM3 Potsdam IRRG modality` |
| `variable_under_test` | `input remote-sensing modality under fixed Exp2-B prompt ensemble strategy` |
| `input_modality` | `irrg` |
| `rgbir_composite_mode` | `null` |
| `prompt_class_ids` | `[0, 1, 2, 3, 4]` |
| `excluded_prompt_class_ids` | `[5]` |
| `fallback_class_id` | `5` |
| `fallback_class_name` | `clutter/background` |
| `prompt_calls_per_patch` | `25` |
| `mask_filter_enabled` | `false` |
| `evaluation_split.split` | `dev` |
| `evaluation_split.num_split_images` | `24` |

推理健康状态保存在 `overall_metrics.json` 的 `inference_health` 字段中：

| 字段 | 数值 |
|---|---:|
| `patch_calls` | 1944 |
| `patch_failures` | 0 |
| `prompt_calls` | 48600 |
| `prompt_successes` | 48600 |
| `prompt_failures` | 0 |
| `empty_prompt_outputs` | 46228 |
| `valid_masks` | 8171 |
| `prompt_failure_rate` | 0.0 |

与 Exp3-A RGB dev 相比，IRRG 的有效候选 mask 数从 12402 降至 8171，空 prompt 输出从 44880 增加到 46228。这说明 IRRG 假彩色输入更难触发当前 prompt 下的有效 SAM3 mask。

## 8. 指标定义

所有指标均在 `ground_truth != 255` 的有效像素上计算。

### 8.1 Overall Accuracy

Overall Accuracy，简称 OA，表示所有有效像素中预测正确的比例：

```text
OA = 正确分类像素数 / 有效像素总数
```

### 8.2 Per-class IoU 与 Mean IoU

每类 IoU 定义为：

```text
IoU_c = TP_c / (TP_c + FP_c + FN_c)
```

Mean IoU 是 6 个 Potsdam 类别 IoU 的宏平均：

```text
mIoU = mean(IoU_0, IoU_1, IoU_2, IoU_3, IoU_4, IoU_5)
```

虽然 Exp3-B 不主动 prompt `clutter/background`，但该类仍然参与 mIoU 计算。

### 8.3 Frequency Weighted IoU

Frequency Weighted IoU，简称 FWIoU，按真实标签中各类别像素频率对 IoU 加权：

```text
FWIoU = sum(freq_c * IoU_c)
freq_c = GT 类别 c 的有效像素数 / 全部有效像素数
```

### 8.4 Precision、Recall 与 F1

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

## 9. 推荐论文主结果

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
| `dataset_precision_per_class` | 各类别 Precision |
| `dataset_recall_per_class` | 各类别 Recall |
| `dataset_confusion_matrix` | 数据集级混淆矩阵 |

逐图平均指标 `average_*` 可作为补充分析，不建议作为论文主结果。

本次 Exp3-B dev 的总体主结果为：

| 指标 | 数值 |
|---|---:|
| `dataset_overall_accuracy` | 0.1284 |
| `dataset_mean_iou` | 0.0629 |
| `dataset_frequency_weighted_iou` | 0.0548 |
| `dataset_mean_f1` | 0.1098 |
| `dataset_mean_precision` | 0.5159 |
| `dataset_mean_recall` | 0.1879 |

与 Exp3-A RGB dev 的主要对比如下：

| 指标 | Exp3-A RGB | Exp3-B IRRG | 变化 |
|---|---:|---:|---:|
| OA | 0.1677 | 0.1284 | -0.0393 |
| mIoU | 0.1032 | 0.0629 | -0.0404 |
| FWIoU | 0.1048 | 0.0548 | -0.0499 |
| Mean F1 | 0.1781 | 0.1098 | -0.0682 |
| Mean Precision | 0.6506 | 0.5159 | -0.1347 |
| Mean Recall | 0.2284 | 0.1879 | -0.0405 |

## 10. 输出文件说明

完整运行后，结果默认保存到：

```text
results_exp3b_sam3_potsdam_irrg_modality_dev/
```

| 输出路径 | 内容 |
|---|---|
| `results_exp3b_sam3_potsdam_irrg_modality_dev/predictions/*_prediction.png` | 每张图的预测标签彩色图 |
| `results_exp3b_sam3_potsdam_irrg_modality_dev/predictions/*_ground_truth.png` | 每张图的 GT 标签彩色图 |
| `results_exp3b_sam3_potsdam_irrg_modality_dev/predictions/*_prediction_id.png` | 每张图的预测类别 ID 图 |
| `results_exp3b_sam3_potsdam_irrg_modality_dev/predictions/*_ground_truth_id.png` | 每张图的 GT 类别 ID 图 |
| `results_exp3b_sam3_potsdam_irrg_modality_dev/visualizations/*_comparison.png` | 原图、GT、预测、叠加图对比 |
| `results_exp3b_sam3_potsdam_irrg_modality_dev/metrics/per_image_metrics.csv` | 每张图的指标 |
| `results_exp3b_sam3_potsdam_irrg_modality_dev/metrics/overall_metrics.json` | 数据集级累计指标和逐图平均指标 |
| `results_exp3b_sam3_potsdam_irrg_modality_dev/metrics/experiment_metadata.json` | Exp3-B 实验变量、prompt ensemble 和 split 信息 |
| `results_exp3b_sam3_potsdam_irrg_modality_dev/metrics/config_snapshot.yaml` | 本次运行使用的配置快照 |
| `results_exp3b_sam3_potsdam_irrg_modality_dev/metrics/run_context.json` | 期望样本、成功处理样本和跳过/失败样本记录 |
| `results_exp3b_sam3_potsdam_irrg_modality_dev/logs/evaluation_log_*.txt` | 运行日志 |

本次运行共生成：

| 输出类型 | 数量 | 说明 |
|---|---:|---|
| predictions | 96 | 24 张图，每张包含 4 个预测/标签文件 |
| visualizations | 24 | 每张图 1 张对比可视化 |
| metrics | 29 | 24 个单图 JSON + 5 个总体/配置/元数据文件 |
| logs | 1 | 本次运行日志 |

当前配置中 `metrics.save_confusion_matrix=false`，因此没有单独生成 `dataset_confusion_matrix.csv`；数据集级混淆矩阵保存在 `overall_metrics.json` 的 `dataset_confusion_matrix` 字段中。

## 11. 结果解读注意事项

1. 本实验是零样本评估，不使用 Potsdam 训练集微调 SAM3。
2. 本实验只使用 IRRG 输入，不使用 RGB、RGBIR、DSM 或 nDSM。
3. 本实验使用 24 张 development split，不是 Exp1/Exp2 的 38 张 full-reference 设置。
4. noBoundary 标签中的黑色像素作为 ignore，不参与指标计算。
5. `clutter/background` 不主动输入 SAM3，但仍作为评估类别。
6. 没有被 SAM3 任一有效前景 mask 覆盖的像素回退为 `clutter/background`。
7. 同一类别的多个 prompt 变体输出均映射回同一个类别 ID。
8. 不启用 Exp2-C 的 class-aware mask area filtering。
9. mIoU、Mean F1、Mean Precision、Mean Recall 均为 6 类宏平均。
10. `dataset_*` 指标由 24 张 dev 图像的混淆矩阵累加后计算，适合作为 Exp3 dev 模态比较主结果。
11. `average_*` 指标是逐图指标的算术平均，适合分析不同图像之间的波动。
12. IRRG 的物理信息优势不等同于 SAM3 开放词汇表征优势；需要结合类别级混淆矩阵解释。

## 12. 论文表述建议

可以在论文实验设置中写：

```text
Exp3-B evaluates IRRG false-color orthophotos as an alternative input modality. The prompt ensemble, foreground-only query policy, score threshold, patch inference strategy, disabled mask filtering, and clutter/background fallback rule are kept identical to Exp3-A. The experiment is conducted on the same 24-image development split.
```

中文可写为：

```text
Exp3-B 将 IRRG 假彩色正射影像作为替代输入模态进行评估。实验保持与 Exp3-A 完全一致的视觉 prompt ensemble、仅前景类别主动查询策略、score threshold、patch 推理方式、关闭 mask 过滤设置以及 clutter/background fallback 规则，并在相同的 24 张 development split 上进行。
```

结果表述可写为：

```text
Exp3-B 在 development split 上取得 0.0629 的数据集级 mIoU、0.1098 的 Mean F1 和 0.1284 的 OA，均低于 RGB 参照组。类别级结果显示，IRRG 并没有提升 low vegetation，反而使其 IoU 从 RGB 的 0.1270 降至 0.0001；building 和 car 也出现下降。虽然 clutter/background 的预测占比从 69.75% 降至 64.03%，但大量植被和背景像素被误归入 impervious surface，因此这不是有效的前景召回提升。
```

对后续实验的衔接可写为：

```text
该结果说明，IRRG 假彩色输入不能直接替代 RGB 作为 SAM3 的主输入模态。后续实验应进一步考察 RGBIR 派生通道或多视图融合策略，以保留 RGB 与通用视觉模型的自然图像对齐能力，同时尝试利用近红外信息提供类别补充。
```
