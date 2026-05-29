# 实验三C：SAM3 + Potsdam RGBIR 派生输入模态 dev 结果说明

## 1. 实验目的

本实验用于分析在固定当前较有效 prompt 策略后，从 Potsdam RGBIR 四波段影像派生出的三通道输入是否能改善 SAM3 在 development split 上的表现。

Exp3-A 使用 RGB 真彩色输入，Exp3-B 使用 IRRG 假彩色输入。Exp3-C 进一步使用 RGBIR 四波段影像，并在脚本中转换为 SAM3 可接受的三通道 composite：

```text
RGBIR -> NIR-R-G
```

实验希望回答的问题是：

```text
如果不直接使用单独的 IRRG 文件，而是从 RGBIR 四波段影像中显式取出 NIR、R、G 构成三通道输入，
是否能在保持 Exp2-B prompt ensemble 策略不变的情况下改善候选 mask 生成和最终语义分割结果？
```

重点关注：

1. `low vegetation` 和 `tree` 是否从 NIR 通道中受益。
2. `building` 和 `car` 是否继续因假彩色输入偏离自然图像分布而下降。
3. RGBIR-derived composite 是否优于 Exp3-B IRRG。
4. RGBIR-derived composite 是否能超过 Exp3-A RGB，成为后续 held-out 的候选主模态。

## 2. 数据设置

当前脚本默认使用以下数据：

| 项目 | 路径 | 说明 |
|---|---|---|
| 输入影像 | `Potsdam/4_Ortho_RGBIR/4_Ortho_RGBIR` | RGBIR 四波段正射影像 |
| 参考标签 | `Potsdam/5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary` | 24 张 participant noBoundary 标签 |
| 输出目录 | `results_exp3c_sam3_potsdam_rgbir_composite_dev` | Exp3-C dev 结果目录 |

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

| 类别 ID | 类别名称 | 英文名称 | 标签颜色 | Exp3-C prompt 策略 |
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

Exp3-C 固定使用 Exp2-B 的 5 类前景视觉 prompt ensemble。每个 patch 共调用：

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

所有 prompt 变体输出的 mask 都映射回对应 Potsdam 类别。`clutter/background` 不主动输入 SAM3，仅作为 fallback 类别。

## 5. 输入模态设置

Exp3-C 的输入模态为 RGBIR-derived composite：

| 配置项 | 内容 |
|---|---|
| `input_modality.name` | `rgbir_composite` |
| `input_modality.rgbir_composite_mode` | `nir-r-g` |
| `input_modality.description` | `Three-channel NIR-R-G composite derived from Potsdam RGBIR.` |
| 图像目录 | `4_Ortho_RGBIR/4_Ortho_RGBIR` |
| 图像文件模板 | `{image_id}_RGBIR.tif` |
| 标签文件模板 | `{image_id}_label_noBoundary.tif` |

RGBIR 原始影像包含 4 个波段：

```text
R, G, B, NIR
```

脚本将其转换为 SAM3 可接受的 3 通道输入：

```text
NIR, R, G
```

本实验不使用以下信息：

```text
DSM / nDSM 高程信息
NDVI 显式指数图
多视图融合
监督训练
```

因此，Exp3-C 是 RGBIR 四波段数据派生为三通道假彩色输入后的单模态消融。

## 6. 推理流程

1. 根据 `evaluation.split=dev` 读取 24 张 development 图像 ID。
2. 读取 RGBIR 四波段影像和 participant noBoundary 标签。
3. 将 RGBIR 影像转换为 `NIR-R-G` 三通道 composite。
4. 将 composite 大图切成 `1008 x 1008` patch。
5. 使用 stride `672` 的边缘对齐滑窗，避免边缘区域遗漏。
6. 对每个 patch 使用 25 个前景 prompt 调用 SAM3。
7. 不对 `clutter/background` 主动调用 SAM3。
8. 将同一类别下不同 prompt 变体产生的 mask 映射回同一个类别 ID。
9. 不启用 class-aware mask area filtering。
10. 对 patch 内多个 mask 按 SAM3 score 进行置信度融合。
11. 未被任何有效前景 mask 覆盖的像素回退为 `clutter/background`。
12. 对重叠 patch 的预测按类别累计置信度，得到整图预测。
13. 将预测与标签在非 ignore 像素上计算指标。
14. 保存预测图、GT 图、对比可视化、单图指标和数据集总体指标。

## 7. 实验元数据

本次实验的元数据保存在：

```text
results_exp3c_sam3_potsdam_rgbir_composite_dev/metrics/experiment_metadata.json
```

核心字段如下：

| 字段 | 内容 |
|---|---|
| `experiment_name` | `Exp3-C: SAM3 Potsdam RGBIR composite` |
| `variable_under_test` | `input remote-sensing modality under fixed Exp2-B prompt ensemble strategy` |
| `input_modality` | `rgbir_composite` |
| `rgbir_composite_mode` | `nir-r-g` |
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
| `empty_prompt_outputs` | 46336 |
| `valid_masks` | 7836 |
| `prompt_failure_rate` | 0.0 |

与 Exp3-A RGB dev 相比，Exp3-C 的有效候选 mask 数从 12402 降至 7836；与 Exp3-B IRRG 的 8171 也基本同级。这说明 NIR-R-G composite 比 RGB 更难触发当前 prompt 下的有效 SAM3 mask。

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

虽然 Exp3-C 不主动 prompt `clutter/background`，但该类仍然参与 mIoU 计算。

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

本次 Exp3-C dev 的总体主结果为：

| 指标 | 数值 |
|---|---:|
| `dataset_overall_accuracy` | 0.1296 |
| `dataset_mean_iou` | 0.0632 |
| `dataset_frequency_weighted_iou` | 0.0575 |
| `dataset_mean_f1` | 0.1104 |
| `dataset_mean_precision` | 0.5732 |
| `dataset_mean_recall` | 0.1909 |

与 Exp3-A / Exp3-B 的主要对比如下：

| 指标 | Exp3-A RGB | Exp3-B IRRG | Exp3-C RGBIR NIR-R-G |
|---|---:|---:|---:|
| OA | 0.1677 | 0.1284 | 0.1296 |
| mIoU | 0.1032 | 0.0629 | 0.0632 |
| FWIoU | 0.1048 | 0.0548 | 0.0575 |
| Mean F1 | 0.1781 | 0.1098 | 0.1104 |
| Mean Precision | 0.6506 | 0.5159 | 0.5732 |
| Mean Recall | 0.2284 | 0.1879 | 0.1909 |

## 10. 输出文件说明

完整运行后，结果默认保存到：

```text
results_exp3c_sam3_potsdam_rgbir_composite_dev/
```

| 输出路径 | 内容 |
|---|---|
| `results_exp3c_sam3_potsdam_rgbir_composite_dev/predictions/*_prediction.png` | 每张图的预测标签彩色图 |
| `results_exp3c_sam3_potsdam_rgbir_composite_dev/predictions/*_ground_truth.png` | 每张图的 GT 标签彩色图 |
| `results_exp3c_sam3_potsdam_rgbir_composite_dev/predictions/*_prediction_id.png` | 每张图的预测类别 ID 图 |
| `results_exp3c_sam3_potsdam_rgbir_composite_dev/predictions/*_ground_truth_id.png` | 每张图的 GT 类别 ID 图 |
| `results_exp3c_sam3_potsdam_rgbir_composite_dev/visualizations/*_comparison.png` | 原图、GT、预测、叠加图对比 |
| `results_exp3c_sam3_potsdam_rgbir_composite_dev/metrics/per_image_metrics.csv` | 每张图的指标 |
| `results_exp3c_sam3_potsdam_rgbir_composite_dev/metrics/overall_metrics.json` | 数据集级累计指标和逐图平均指标 |
| `results_exp3c_sam3_potsdam_rgbir_composite_dev/metrics/experiment_metadata.json` | Exp3-C 实验变量、prompt ensemble 和 split 信息 |
| `results_exp3c_sam3_potsdam_rgbir_composite_dev/metrics/config_snapshot.yaml` | 本次运行使用的配置快照 |
| `results_exp3c_sam3_potsdam_rgbir_composite_dev/metrics/run_context.json` | 期望样本、成功处理样本和跳过/失败样本记录 |
| `results_exp3c_sam3_potsdam_rgbir_composite_dev/logs/evaluation_log_*.txt` | 运行日志 |

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
2. 本实验使用 RGBIR 四波段影像派生出的 NIR-R-G 三通道输入。
3. 本实验使用 24 张 development split，不是 Exp1/Exp2 的 38 张 full-reference 设置。
4. noBoundary 标签中的黑色像素作为 ignore，不参与指标计算。
5. `clutter/background` 不主动输入 SAM3，但仍作为评估类别。
6. 没有被 SAM3 任一有效前景 mask 覆盖的像素回退为 `clutter/background`。
7. 同一类别的多个 prompt 变体输出均映射回同一个类别 ID。
8. 不启用 Exp2-C 的 class-aware mask area filtering。
9. mIoU、Mean F1、Mean Precision、Mean Recall 均为 6 类宏平均。
10. `dataset_*` 指标由 24 张 dev 图像的混淆矩阵累加后计算，适合作为 Exp3 dev 模态比较主结果。
11. RGBIR composite 的物理光谱信息优势不等同于 SAM3 开放词汇表征优势；需要结合类别级混淆矩阵解释。

## 12. 论文表述建议

可以在论文实验设置中写：

```text
Exp3-C evaluates a three-channel composite derived from the four-band RGBIR orthophotos. Specifically, the NIR, red, and green bands are stacked as an NIR-R-G input. All prompt variants, foreground-only query policy, score threshold, patch inference strategy, disabled mask filtering, and clutter/background fallback rule are kept identical to Exp3-A and Exp3-B.
```

中文可写为：

```text
Exp3-C 评估由 RGBIR 四波段正射影像派生出的三通道 composite。具体而言，实验将近红外、红光和绿光波段堆叠为 NIR-R-G 输入。该实验保持与 Exp3-A 和 Exp3-B 完全一致的 prompt variants、仅前景类别主动查询策略、score threshold、patch 推理方式、关闭 mask 过滤设置以及 clutter/background fallback 规则。
```

结果表述可写为：

```text
Exp3-C 在 development split 上取得 0.0632 的数据集级 mIoU、0.1104 的 Mean F1 和 0.1296 的 OA。该结果略高于 IRRG，但仍明显低于 RGB 参照组。类别级结果显示，RGBIR-derived NIR-R-G composite 并没有改善 low vegetation，其 IoU 仅为 0.0007；tree 虽然略高于 RGB 和 IRRG，但 IoU 也只有 0.0026，绝对效果仍很弱。因此，直接使用近红外相关三通道 composite 并不能有效提升当前 SAM3 开放词汇零样本语义分割表现。
```

对后续实验的衔接可写为：

```text
Exp3-B 和 Exp3-C 共同说明，近红外相关输入直接作为 SAM3 单输入模态时没有超过 RGB。后续若继续利用 NIR 信息，应优先考虑 NDVI / class-aware prior、多视图融合或监督模型学习，而不是直接用假彩色输入替换 RGB。
```
