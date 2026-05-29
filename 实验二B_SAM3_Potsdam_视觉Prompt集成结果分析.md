# 实验二B：SAM3 + Potsdam 视觉 Prompt 集成结果分析

## 1. 本次运行概况

本次实验已完成 `exp2b_sam3_potsdam_visual_prompt_ensemble.py` 的全量运行，结果保存于：

```text
results_exp2b_sam3_potsdam_visual_prompt_ensemble/
```

Exp2-B 的实验变量是：

```text
对 5 个前景类别使用视觉化 prompt ensemble，并继续不主动 prompt clutter/background。
```

也就是说，`clutter/background` 仍然不作为主动文本查询类别，只作为评估类别和未覆盖像素的 fallback 类别；每个前景类别由 5 个更视觉化、更接近自然图像概念的 prompt 共同查询。

本次运行使用的数据与配置如下：

| 项目 | 内容 |
|---|---|
| 输入影像 | `Potsdam/2_Ortho_RGB/2_Ortho_RGB` |
| 参考标签 | `Potsdam/5_Labels_all_noBoundary` |
| 自动发现样本数 | 38 张 |
| 成功处理样本数 | 38 张 |
| 有有效评估像素的样本数 | 37 张 |
| 全 ignore 样本 | `top_potsdam_4_12_RGB` |
| Patch 大小 | `1008 x 1008` |
| Patch stride | `672` |
| 每张图 patch 数 | 81 |
| 总 patch 数 | 3078 |
| 主动 prompt 类别 | 0-4，即 impervious surface、building、low vegetation、tree、car |
| 排除主动 prompt 类别 | 5，即 clutter/background |
| 每类 prompt 变体数 | 5 |
| 每个 patch 文本 prompt 调用数 | 25 |
| 总文本 prompt 调用数 | 76950 |
| fallback 类别 | `clutter/background` |
| 评估方式 | noBoundary 标签，黑色 `(0,0,0)` 映射为 `ignore_index=255` |

输出文件数量检查：

| 输出类型 | 数量 | 说明 |
|---|---:|---|
| `results_exp2b_sam3_potsdam_visual_prompt_ensemble/predictions` | 152 | 38 张图，每张包含彩色预测、彩色 GT、预测 ID、GT ID |
| `results_exp2b_sam3_potsdam_visual_prompt_ensemble/visualizations` | 38 | 每张图一张对比可视化 |
| `results_exp2b_sam3_potsdam_visual_prompt_ensemble/metrics` | 43 | 38 个单图 JSON + `overall_metrics.json`、`per_image_metrics.csv`、`experiment_metadata.json`、`config_snapshot.yaml`、`run_context.json` |

关键结果文件：

```text
results_exp2b_sam3_potsdam_visual_prompt_ensemble/metrics/overall_metrics.json
results_exp2b_sam3_potsdam_visual_prompt_ensemble/metrics/per_image_metrics.csv
results_exp2b_sam3_potsdam_visual_prompt_ensemble/metrics/experiment_metadata.json
results_exp2b_sam3_potsdam_visual_prompt_ensemble/metrics/run_context.json
results_exp2b_sam3_potsdam_visual_prompt_ensemble/metrics/config_snapshot.yaml
```

当前配置中 `metrics.save_confusion_matrix=false`，因此没有单独生成 `dataset_confusion_matrix.csv`；数据集级混淆矩阵保存在 `overall_metrics.json` 的 `dataset_confusion_matrix` 字段中。

## 2. Prompt Ensemble 设置

本次实验使用的 prompt variants 如下：

| 类别 | Prompt variants |
|---|---|
| impervious surface | `impervious surface`, `road and pavement`, `asphalt road`, `paved area`, `parking lot` |
| building | `building`, `buildings`, `rooftop`, `building roof`, `house roof` |
| low vegetation | `low vegetation`, `grass`, `lawn`, `low plants`, `ground vegetation` |
| tree | `tree`, `trees`, `tree crown`, `tall tree`, `urban trees` |
| car | `car`, `cars`, `vehicle`, `small vehicle`, `parked car` |

所有同类 prompt 变体产生的 mask 都映射回原始 Potsdam 类别。例如 `road and pavement`、`asphalt road`、`parking lot` 都映射为 `impervious surface`。

## 3. 总体结果

本次实验的推荐论文主结果应使用 `dataset_*` 指标，即先累计 38 张图的混淆矩阵，再计算总体指标。

| 指标 | 数值 |
|---|---:|
| Overall Accuracy | 0.1659 |
| Mean IoU | 0.1052 |
| Mean F1 | 0.1806 |
| Mean Precision | 0.6647 |
| Mean Recall | 0.2231 |
| Frequency Weighted IoU | 0.1056 |

逐图平均指标如下，仅建议作为补充参考：

| 指标 | 数值 |
|---|---:|
| Average Overall Accuracy | 0.1611 |
| Average Mean IoU | 0.1031 |
| Average Mean F1 | 0.1715 |
| Average Mean Precision | 0.4793 |
| Average Mean Recall | 0.2323 |
| Average Frequency Weighted IoU | 0.1065 |

总体判断：

```text
视觉化 prompt ensemble 明显提升了 SAM3 在 Potsdam 上的零样本分割结果。
```

与 Exp1/Exp2-A 相比，Exp2-B 的数据集级 mIoU 从 0.0578 提升到 0.1052，Mean F1 从 0.1020 提升到 0.1806，OA 从 0.0892 提升到 0.1659。

## 4. 与 Exp1 / Exp2-A 的对比

| 实验 | Prompt 策略 | Prompt 调用数 | Valid masks | OA | mIoU | Mean F1 | FWIoU |
|---|---|---:|---:|---:|---:|---:|---:|
| Exp1 | 六类官方 prompt | 18468 | 6823 | 0.0892 | 0.0578 | 0.1020 | 0.0426 |
| Exp2-A | 5 类前景 prompt，无背景 prompt | 15390 | 6823 | 0.0892 | 0.0578 | 0.1020 | 0.0426 |
| Exp2-B | 5 类前景 prompt ensemble，无背景 prompt | 76950 | 23066 | 0.1659 | 0.1052 | 0.1806 | 0.1056 |

相对 Exp1，Exp2-B 的主要提升如下：

| 指标 | Exp1 | Exp2-B | 绝对提升 | 相对倍数 |
|---|---:|---:|---:|---:|
| OA | 0.0892 | 0.1659 | +0.0767 | 1.86x |
| mIoU | 0.0578 | 0.1052 | +0.0474 | 1.82x |
| Mean F1 | 0.1020 | 0.1806 | +0.0786 | 1.77x |
| Mean Recall | 0.1702 | 0.2231 | +0.0529 | 1.31x |
| FWIoU | 0.0426 | 0.1056 | +0.0630 | 2.48x |

从推理统计看：

```text
prompt_calls: 18468 -> 76950
valid_masks: 6823 -> 23066
empty_prompt_outputs: 16645 -> 70628
```

Exp2-B 显著增加了 prompt 查询次数，也生成了更多有效 mask。这说明 prompt ensemble 的作用主要是提高前景候选 mask 覆盖，而不是改变 fallback 规则。

## 5. 各类别结果

| 类别 | GT 占比 | 预测占比 | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 29.41% | 24.08% | 0.2014 | 0.3353 | 0.3724 | 0.3049 |
| building | 26.54% | 1.27% | 0.0458 | 0.0875 | 0.9614 | 0.0458 |
| low vegetation | 22.31% | 4.96% | 0.1310 | 0.2316 | 0.6364 | 0.1415 |
| tree | 15.66% | 0.00% | 0.0002 | 0.0004 | 0.9965 | 0.0002 |
| car | 1.41% | 0.31% | 0.2114 | 0.3490 | 0.9792 | 0.2124 |
| clutter/background | 4.65% | 69.38% | 0.0415 | 0.0796 | 0.0425 | 0.6336 |

类别级提升主要集中在 `impervious surface`、`low vegetation` 和 `car`：

| 类别 | Exp1 IoU | Exp2-B IoU | 绝对提升 |
|---|---:|---:|---:|
| impervious surface | 0.1197 | 0.2014 | +0.0817 |
| building | 0.0112 | 0.0458 | +0.0346 |
| low vegetation | 0.0007 | 0.1310 | +0.1303 |
| tree | 0.0001 | 0.0002 | +0.0001 |
| car | 0.1778 | 0.2114 | +0.0336 |
| clutter/background | 0.0375 | 0.0415 | +0.0039 |

其中 `low vegetation` 的提升最明显，说明 `grass`、`lawn`、`low plants`、`ground vegetation` 这类视觉化 prompt 比官方类名 `low vegetation` 更容易被 SAM3 理解。

## 6. 混淆矩阵解读

数据集级混淆矩阵显示，Exp2-B 仍然存在大量像素被预测为 `clutter/background`：

| 真实类别 | 主要误分方向 | 占该真实类别比例 |
|---|---|---:|
| impervious surface | clutter/background | 69.22% |
| building | clutter/background | 87.83% |
| low vegetation | clutter/background | 52.92% |
| tree | clutter/background | 63.30% |
| car | clutter/background | 73.08% |
| clutter/background | clutter/background | 63.36% |

与 Exp1 相比，这些比例已经下降。例如：

| 真实类别 | Exp1 误分为 clutter | Exp2-B 误分为 clutter | 变化 |
|---|---:|---:|---:|
| impervious surface | 81.85% | 69.22% | -12.63% |
| building | 94.13% | 87.83% | -6.30% |
| low vegetation | 63.78% | 52.92% | -10.86% |
| tree | 73.10% | 63.30% | -9.80% |
| car | 80.11% | 73.08% | -7.03% |

这说明 prompt ensemble 确实减少了前景区域回退到背景类的比例，但问题远未解决。预测结果中 `clutter/background` 仍占 69.38%，远高于真实占比 4.65%。

## 7. 单图结果分析

按 `mean_iou` 排序，本次实验表现最好的 5 张图为：

| 排名 | 图像 | Mean IoU | OA |
|---:|---|---:|---:|
| 1 | `top_potsdam_7_10_RGB` | 0.1897 | 0.4003 |
| 2 | `top_potsdam_3_11_RGB` | 0.1611 | 0.1930 |
| 3 | `top_potsdam_5_15_RGB` | 0.1406 | 0.2135 |
| 4 | `top_potsdam_4_14_RGB` | 0.1399 | 0.1587 |
| 5 | `top_potsdam_5_14_RGB` | 0.1377 | 0.1959 |

表现最差的 5 张图为：

| 排名 | 图像 | Mean IoU | OA |
|---:|---|---:|---:|
| 1 | `top_potsdam_4_12_RGB` | 0.0000 | 0.0000 |
| 2 | `top_potsdam_6_7_RGB` | 0.0361 | 0.0910 |
| 3 | `top_potsdam_7_8_RGB` | 0.0417 | 0.0608 |
| 4 | `top_potsdam_5_12_RGB` | 0.0650 | 0.0952 |
| 5 | `top_potsdam_5_11_RGB` | 0.0725 | 0.1000 |

其中 `top_potsdam_4_12_RGB` 的单图指标全为 0，是因为对应 noBoundary 标签中有效像素数为 0：

```text
valid_pixels = 0
ignored_pixels = 36000000
```

这张图不影响 `dataset_*` 累计指标，因为其混淆矩阵全为 0；但它会拉低 `average_*` 逐图平均指标。因此论文主结果应使用 `dataset_*`，不要使用 `average_*` 作为核心结论。

## 8. 结果意义

Exp2-B 说明 prompt 构造是 SAM3 遥感推理分割中的关键因素。

直接使用 Potsdam 官方类别名时，模型很难理解 `impervious surface`、`low vegetation` 这类遥感术语。将其拆成更接近日常视觉概念的 prompt 后，SAM3 能生成更多候选 mask，并显著提高 mIoU 和 Mean F1。

但 Exp2-B 也说明，仅靠 prompt ensemble 还不足以完成可靠的遥感密集语义分割：

1. `clutter/background` 仍然被严重过预测。
2. `building` recall 虽然提升到 0.0458，但仍然很低。
3. `tree` 几乎没有被有效召回，IoU 仍接近 0。
4. prompt 调用次数显著增加，计算代价更高。

因此，Exp2-B 更适合作为“prompt 工程有效但不充分”的证据，为后续 mask 过滤、多模态输入和遥感先验约束提供动机。

## 9. 论文中如何表述这个结果

建议将 Exp2-B 定位为：

```text
前景类别视觉 prompt ensemble 消融实验。
```

英文可写为：

```text
Exp2-B introduces a visual prompt ensemble for the five foreground Potsdam classes while keeping clutter/background as a residual fallback class. Compared with the six-prompt zero-shot baseline, the dataset-level mIoU increases from 0.0578 to 0.1052 and the mean F1 increases from 0.1020 to 0.1806. The number of valid masks also increases from 6823 to 23066, indicating that visual prompt variants improve foreground mask coverage. However, 69.38% of pixels are still predicted as clutter/background, showing that prompt engineering alone is insufficient for reliable dense remote-sensing semantic segmentation.
```

中文可写为：

```text
Exp2-B 对 Potsdam 五个前景类别引入视觉化 prompt ensemble，并继续将 clutter/background 作为 residual fallback 类别。与六类官方 prompt 的零样本基线相比，数据集级 mIoU 从 0.0578 提升到 0.1052，Mean F1 从 0.1020 提升到 0.1806，有效 mask 数量从 6823 增加到 23066。这说明视觉化 prompt 变体能够改善前景候选 mask 覆盖。然而，预测结果中仍有 69.38% 的像素被归入 clutter/background，表明仅依赖 prompt 工程仍不足以实现可靠的遥感密集语义分割。
```

## 10. 推荐论文表格

### 10.1 总体指标表

| Method | Active prompts | Fallback class | OA | mIoU | FWIoU | Mean F1 | Mean Precision | Mean Recall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SAM3 zero-shot | 6 official prompts | clutter/background | 0.0892 | 0.0578 | 0.0426 | 0.1020 | 0.5776 | 0.1702 |
| Exp2-A no background prompt | 5 foreground prompts | clutter/background | 0.0892 | 0.0578 | 0.0426 | 0.1020 | 0.5776 | 0.1702 |
| Exp2-B visual prompt ensemble | 25 foreground prompts | clutter/background | 0.1659 | 0.1052 | 0.1056 | 0.1806 | 0.6647 | 0.2231 |

### 10.2 Prompt 调用与 mask 统计表

| Method | Patch calls | Prompt calls | Empty prompt outputs | Valid masks |
|---|---:|---:|---:|---:|
| SAM3 zero-shot | 3078 | 18468 | 16645 | 6823 |
| Exp2-A no background prompt | 3078 | 15390 | 13567 | 6823 |
| Exp2-B visual prompt ensemble | 3078 | 76950 | 70628 | 23066 |

### 10.3 各类别 IoU 表

| Method | Impervious | Building | Low Veg. | Tree | Car | Clutter | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp2-B visual prompt ensemble | 0.2014 | 0.0458 | 0.1310 | 0.0002 | 0.2114 | 0.0415 | 0.1052 |

## 11. 本次结果的结论

本次实验完成了 SAM3 在 Potsdam RGB 数据上的视觉 prompt ensemble 消融。结果表明：

```text
视觉化 prompt ensemble 能显著改善 SAM3 的前景候选 mask 覆盖，并提升数据集级指标。
```

但同时：

```text
该策略仍无法解决遥感密集语义分割中的大面积背景回退和类别召回不足问题。
```

因此，Exp2-B 支撑了论文中的一个关键判断：prompt 工程是推理分割体系中的有效环节，但必须继续结合 mask 质量控制、多源遥感输入和领域先验，才能向更可靠的遥感/水利分割框架推进。
