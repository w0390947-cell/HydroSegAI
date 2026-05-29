# 实验三A：SAM3 + Potsdam RGB 输入模态 dev 结果分析

## 1. 本次运行概况

本次实验已完成 `exp3a_sam3_potsdam_rgb_modality.py` 的 development split 运行，结果保存于：

```text
results_exp3a_sam3_potsdam_rgb_modality_dev/
```

Exp3-A 的实验变量是：

```text
在固定 Exp2-B 视觉 prompt ensemble 策略后，仅使用 RGB 真彩色正射影像作为输入模态。
```

也就是说，本实验不是新的 prompt 消融，而是 Exp3 多源遥感输入模态实验中的 RGB 参照组。后续 Exp3-B 的 IRRG、Exp3-C 的 RGBIR 派生三通道输入、Exp3-D 的多视图融合，都应优先与本实验在同一 `dev` split 上比较。

本次运行使用的数据与配置如下：

| 项目 | 内容 |
|---|---|
| 输入影像 | `Potsdam/2_Ortho_RGB/2_Ortho_RGB` |
| 参考标签 | `Potsdam/5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary` |
| 评估划分 | `dev`，24 张 official participant-labeled tiles |
| 自动期望样本数 | 24 张 |
| 成功处理样本数 | 24 张 |
| 跳过样本数 | 0 张 |
| 失败样本数 | 0 张 |
| Patch 大小 | `1008 x 1008` |
| Patch stride | `672` |
| 每张图 patch 数 | 81 |
| 总 patch 数 | 1944 |
| 主动 prompt 类别 | 0-4，即 impervious surface、building、low vegetation、tree、car |
| 排除主动 prompt 类别 | 5，即 clutter/background |
| 每类 prompt 变体数 | 5 |
| 每个 patch 文本 prompt 调用数 | 25 |
| 总文本 prompt 调用数 | 48600 |
| 总有效候选 mask 数 | 12402 |
| mask 面积过滤 | 关闭 |
| fallback 类别 | `clutter/background` |
| 评估方式 | noBoundary 标签，黑色 `(0,0,0)` 映射为 `ignore_index=255` |

输出文件数量检查：

| 输出类型 | 数量 | 说明 |
|---|---:|---|
| `results_exp3a_sam3_potsdam_rgb_modality_dev/predictions` | 96 | 24 张图，每张包含彩色预测、彩色 GT、预测 ID、GT ID |
| `results_exp3a_sam3_potsdam_rgb_modality_dev/visualizations` | 24 | 每张图一张对比可视化 |
| `results_exp3a_sam3_potsdam_rgb_modality_dev/metrics` | 29 | 24 个单图 JSON + `overall_metrics.json`、`per_image_metrics.csv`、`experiment_metadata.json`、`config_snapshot.yaml`、`run_context.json` |
| `results_exp3a_sam3_potsdam_rgb_modality_dev/logs` | 1 | 本次运行日志 |

关键结果文件：

```text
results_exp3a_sam3_potsdam_rgb_modality_dev/metrics/overall_metrics.json
results_exp3a_sam3_potsdam_rgb_modality_dev/metrics/per_image_metrics.csv
results_exp3a_sam3_potsdam_rgb_modality_dev/metrics/experiment_metadata.json
results_exp3a_sam3_potsdam_rgb_modality_dev/metrics/run_context.json
results_exp3a_sam3_potsdam_rgb_modality_dev/metrics/config_snapshot.yaml
```

当前配置中 `metrics.save_confusion_matrix=false`，因此没有单独生成 `dataset_confusion_matrix.csv`；数据集级混淆矩阵保存在 `overall_metrics.json` 的 `dataset_confusion_matrix` 字段中。

## 2. 固定 Prompt 与模态设置

Exp3-A 固定继承 Exp2-B 的视觉 prompt ensemble：

| 类别 | Prompt variants |
|---|---|
| impervious surface | `impervious surface`, `road and pavement`, `asphalt road`, `paved area`, `parking lot` |
| building | `building`, `buildings`, `rooftop`, `building roof`, `house roof` |
| low vegetation | `low vegetation`, `grass`, `lawn`, `low plants`, `ground vegetation` |
| tree | `tree`, `trees`, `tree crown`, `tall tree`, `urban trees` |
| car | `car`, `cars`, `vehicle`, `small vehicle`, `parked car` |

`clutter/background` 不作为主动文本查询类别，只作为评估类别和未覆盖像素的 fallback 类别。

模态设置如下：

| 字段 | 内容 |
|---|---|
| `input_modality.name` | `rgb` |
| `input_modality.description` | `Natural RGB orthophoto input.` |
| 图像来源 | Potsdam RGB 正射影像 |
| 近红外信息 | 未使用 |
| DSM / nDSM | 未使用 |
| RGBIR 派生通道 | 未使用 |
| 多视图融合 | 未使用 |

因此，Exp3-A 的主要作用是给 Exp3-B/C/D 提供干净的 RGB dev 参照结果。

## 3. 总体结果

本次实验的推荐论文主结果应使用 `dataset_*` 指标，即先累计 24 张 dev 图像的混淆矩阵，再计算总体指标。

| 指标 | 数值 |
|---|---:|
| Overall Accuracy | 0.1677 |
| Mean IoU | 0.1032 |
| Mean F1 | 0.1781 |
| Mean Precision | 0.6506 |
| Mean Recall | 0.2284 |
| Frequency Weighted IoU | 0.1048 |

逐图平均指标如下，仅建议作为补充参考：

| 指标 | 数值 |
|---|---:|
| Average Overall Accuracy | 0.1673 |
| Average Mean IoU | 0.1057 |
| Average Mean F1 | 0.1756 |
| Average Mean Precision | 0.5090 |
| Average Mean Recall | 0.2398 |
| Average Frequency Weighted IoU | 0.1098 |

总体判断：

```text
在 development split 上，RGB 输入模态延续了 Exp2-B 的基本行为：视觉 prompt ensemble 能产生一定前景候选，但整体分割性能仍较低，预测结果仍严重依赖 clutter/background fallback。
```

需要注意，本结果使用 24 张 participant-labeled dev tiles；Exp1/Exp2 的主结果使用 38 张 full-reference tiles。二者标签目录和评估划分不同，因此不应把 Exp3-A dev 的数值与 Exp2-B full 的数值作为严格同表主对比。更合理的用法是：

```text
Exp3-A dev 作为 RGB 模态参照；
Exp3-B/C/D dev 与 Exp3-A dev 在同一划分上比较；
最终确定规则后，再在 held-out split 上评估。
```

## 4. 各类别结果

| 类别 | GT 占比 | 预测占比 | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 28.34% | 23.86% | 0.2032 | 0.3378 | 0.3695 | 0.3111 |
| building | 27.83% | 1.35% | 0.0460 | 0.0880 | 0.9541 | 0.0462 |
| low vegetation | 23.20% | 4.76% | 0.1270 | 0.2254 | 0.6618 | 0.1358 |
| tree | 14.49% | 0.01% | 0.0004 | 0.0009 | 0.8901 | 0.0004 |
| car | 1.36% | 0.27% | 0.1970 | 0.3292 | 0.9819 | 0.1977 |
| clutter/background | 4.78% | 69.75% | 0.0455 | 0.0871 | 0.0465 | 0.6789 |

类别级结果说明：

1. `impervious surface` 是 RGB 模态下表现最稳定的大面积前景类，IoU 为 0.2032，Recall 为 0.3111，但仍有大量硬化地表回退为背景。
2. `building` 的 precision 高达 0.9541，但 recall 仅 0.0462，说明预测到的建筑像素较可信，问题主要是覆盖严重不足。
3. `low vegetation` 的 IoU 为 0.1270，是 prompt ensemble 带来有效覆盖的类别之一，但预测占比只有 4.76%，远低于 GT 占比 23.20%。
4. `tree` 几乎没有被召回，IoU 仅 0.0004。这是 RGB 输入和当前 prompt 策略下最突出的类别短板。
5. `car` 作为小目标仍保持很高 precision，IoU 为 0.1970，但 recall 只有 0.1977，漏检仍明显。
6. `clutter/background` 的预测占比达到 69.75%，远高于真实占比 4.78%，说明大量前景像素没有被有效前景 mask 覆盖。

## 5. 混淆矩阵解读

数据集级混淆矩阵显示，主要错误方向仍是前景类别被预测为 `clutter/background`：

| 真实类别 | 误分为 clutter/background 的比例 |
|---|---:|
| impervious surface | 68.57% |
| building | 88.70% |
| low vegetation | 53.62% |
| tree | 61.76% |
| car | 73.51% |
| clutter/background | 67.89% |

这个结果说明，Exp3-A 的 RGB 输入并没有从根本上解决 SAM3 在 Potsdam 上的前景召回不足问题。虽然视觉 prompt ensemble 比官方类别名更容易触发有效 mask，但大量道路、屋顶、植被、树冠和车辆像素仍没有被主动前景 mask 覆盖，最终落入 fallback 类。

其中最需要关注的是：

```text
真实 clutter/background 只占 4.78%，但预测 clutter/background 占 69.75%。
```

这与 Exp1/Exp2 系列暴露出的核心问题一致：当前系统的瓶颈不是某个类别 precision 不够，而是 SAM3 候选 mask 对遥感密集前景区域的覆盖不足。

## 6. 单图结果分析

按 `mean_iou` 排序，本次实验表现最好的 5 张图为：

| 排名 | 图像 | Mean IoU | OA |
|---:|---|---:|---:|
| 1 | `top_potsdam_7_10_RGB` | 0.1897 | 0.4003 |
| 2 | `top_potsdam_3_11_RGB` | 0.1611 | 0.1930 |
| 3 | `top_potsdam_7_12_RGB` | 0.1355 | 0.2462 |
| 4 | `top_potsdam_7_11_RGB` | 0.1341 | 0.3187 |
| 5 | `top_potsdam_4_12_RGB` | 0.1328 | 0.1512 |

表现最差的 5 张图为：

| 排名 | 图像 | Mean IoU | OA |
|---:|---|---:|---:|
| 1 | `top_potsdam_7_8_RGB` | 0.0417 | 0.0608 |
| 2 | `top_potsdam_6_7_RGB` | 0.0608 | 0.0913 |
| 3 | `top_potsdam_5_12_RGB` | 0.0650 | 0.0952 |
| 4 | `top_potsdam_5_11_RGB` | 0.0725 | 0.1000 |
| 5 | `top_potsdam_6_12_RGB` | 0.0777 | 0.1018 |

与 Exp1/Exp2 full-reference 结果不同，本次 dev split 中 `top_potsdam_4_12_RGB` 使用 participant noBoundary 标签，具有有效评估像素，因此不再是全 ignore 样本。这也进一步说明，Exp3 dev 结果不应与 Exp1/Exp2 full 结果进行简单逐图等价对比。

## 7. 作为 Exp3 模态参照的意义

Exp3-A 的意义不在于证明 RGB 输入已经足够好，而在于建立后续模态实验的参照系：

| 后续实验 | 应重点比较的问题 |
|---|---|
| Exp3-B IRRG | 近红外假彩色输入是否提高 vegetation/tree 的召回，是否削弱 building/car |
| Exp3-C RGBIR 派生三通道 | NIR 或 NDVI 派生信息是否改善低矮植被、树木和硬化地表的类别区分 |
| Exp3-D 多视图融合 | 多模态候选是否减少前景回退背景，还是引入更多错误 mask |

从 Exp3-A dev 结果看，后续模态如果要证明有效，至少应关注以下信号：

1. `clutter/background` 预测占比是否低于 69.75%。
2. 真实前景误分为 `clutter/background` 的比例是否下降。
3. `low vegetation` 和 `tree` 的 recall 是否提高。
4. `building` 是否在保持高 precision 的同时增加 recall。
5. `car` 这类小目标是否在模态变化后出现明显退化。

## 8. 论文中如何表述这个结果

建议将 Exp3-A 定位为：

```text
多源遥感输入模态实验中的 RGB development reference。
```

英文可写为：

```text
Exp3-A fixes the visual prompt ensemble strategy from Exp2-B and evaluates the natural RGB orthophoto input on the 24-image development split. The dataset-level mIoU is 0.1032 and the mean F1 is 0.1781. Although RGB with visual prompt variants provides non-trivial foreground masks for impervious surfaces, low vegetation, and cars, 69.75% of pixels are still assigned to clutter/background. This confirms that RGB serves as a necessary modality reference, while the main limitation remains insufficient foreground mask coverage.
```

中文可写为：

```text
Exp3-A 固定采用 Exp2-B 的视觉 prompt ensemble 策略，并在 24 张 development split 上评估 RGB 真彩色正射影像输入。数据集级 mIoU 为 0.1032，Mean F1 为 0.1781。RGB 输入能够为不透水表面、低矮植被和车辆生成一定数量的有效前景 mask，但预测结果中仍有 69.75% 的像素被归入 clutter/background，说明主要瓶颈仍是前景候选 mask 覆盖不足。因此，Exp3-A 更适合作为后续 IRRG、RGBIR 派生输入和多视图融合实验的 RGB 参照组。
```

## 9. 推荐论文表格

### 9.1 Exp3 dev 总体指标表

| Method | Split | Input modality | Active prompts | Mask filter | OA | mIoU | FWIoU | Mean F1 | Mean Precision | Mean Recall |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Exp3-A | dev | RGB | 25 foreground prompts | disabled | 0.1677 | 0.1032 | 0.1048 | 0.1781 | 0.6506 | 0.2284 |

### 9.2 Exp3-A dev 类别指标表

| Class | GT % | Pred % | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 28.34 | 23.86 | 0.2032 | 0.3378 | 0.3695 | 0.3111 |
| building | 27.83 | 1.35 | 0.0460 | 0.0880 | 0.9541 | 0.0462 |
| low vegetation | 23.20 | 4.76 | 0.1270 | 0.2254 | 0.6618 | 0.1358 |
| tree | 14.49 | 0.01 | 0.0004 | 0.0009 | 0.8901 | 0.0004 |
| car | 1.36 | 0.27 | 0.1970 | 0.3292 | 0.9819 | 0.1977 |
| clutter/background | 4.78 | 69.75 | 0.0455 | 0.0871 | 0.0465 | 0.6789 |

## 10. 后续分析建议

后续写 Exp3-B dev 时，应在同一表中并列 Exp3-A RGB 与 Exp3-B IRRG，重点比较：

1. 总体 OA、mIoU、Mean F1、FWIoU 是否提升。
2. `low vegetation` 和 `tree` 的 IoU / Recall 是否受益于近红外。
3. `building`、`car` 是否因假彩色输入偏离 SAM3 自然图像预训练分布而下降。
4. `clutter/background` 预测占比是否下降。
5. 各真实前景类误分为 `clutter/background` 的比例是否下降。

只有当 IRRG 或 RGBIR 在这些类别级信号上表现出清晰差异时，才能进一步讨论 class-aware modality routing 或多视图融合的必要性。
