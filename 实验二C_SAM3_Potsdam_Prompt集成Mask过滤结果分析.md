# 实验二C：SAM3 + Potsdam Prompt 集成与 Mask 过滤结果分析

## 1. 本次运行概况

本次实验已完成 `exp2c_sam3_potsdam_prompt_ensemble_mask_filter.py` 的全量运行，结果保存于：

```text
results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/
```

Exp2-C 的实验变量是：

```text
在 Exp2-B 的视觉化 prompt ensemble 基础上，加入类别相关的候选 mask 面积过滤。
```

也就是说，`clutter/background` 仍然不作为主动文本查询类别，只作为评估类别和未覆盖像素的 fallback 类别；每个前景类别仍由 5 个视觉化 prompt 共同查询。新增的 mask 过滤规则在 patch 融合前生效。

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
| 总有效候选 mask 数 | 23066 |
| 面积过滤后保留 mask 数 | 22861 |
| fallback 类别 | `clutter/background` |
| 评估方式 | noBoundary 标签，黑色 `(0,0,0)` 映射为 `ignore_index=255` |

输出文件数量检查：

| 输出类型 | 数量 | 说明 |
|---|---:|---|
| `results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/predictions` | 152 | 38 张图，每张包含彩色预测、彩色 GT、预测 ID、GT ID |
| `results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/visualizations` | 38 | 每张图一张对比可视化 |
| `results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/metrics` | 43 | 38 个单图 JSON + `overall_metrics.json`、`per_image_metrics.csv`、`experiment_metadata.json`、`config_snapshot.yaml`、`run_context.json` |
| `results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/logs` | 1 | 本次运行日志 |

关键结果文件：

```text
results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/metrics/overall_metrics.json
results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/metrics/per_image_metrics.csv
results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/metrics/experiment_metadata.json
results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/metrics/run_context.json
results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/metrics/config_snapshot.yaml
```

当前配置中 `metrics.save_confusion_matrix=false`，因此没有单独生成 `dataset_confusion_matrix.csv`；数据集级混淆矩阵保存在 `overall_metrics.json` 的 `dataset_confusion_matrix` 字段中。

## 2. Prompt Ensemble 与 Mask 过滤设置

本次实验使用的 prompt variants 如下：

| 类别 | Prompt variants |
|---|---|
| impervious surface | `impervious surface`, `road and pavement`, `asphalt road`, `paved area`, `parking lot` |
| building | `building`, `buildings`, `rooftop`, `building roof`, `house roof` |
| low vegetation | `low vegetation`, `grass`, `lawn`, `low plants`, `ground vegetation` |
| tree | `tree`, `trees`, `tree crown`, `tall tree`, `urban trees` |
| car | `car`, `cars`, `vehicle`, `small vehicle`, `parked car` |

面积过滤规则如下：

| 类别 | 最小面积比例 | 最大面积比例 |
|---|---:|---:|
| impervious surface | 0.001 | 0.80 |
| building | 0.0005 | 0.60 |
| low vegetation | 0.001 | 0.80 |
| tree | 0.0003 | 0.50 |
| car | 0.00001 | 0.05 |

实际过滤统计如下：

| 类别 | 输入 mask | 保留 mask | 过滤过小 | 过滤过大 | 过滤比例 |
|---|---:|---:|---:|---:|---:|
| impervious surface | 1506 | 1330 | 0 | 176 | 11.69% |
| building | 465 | 465 | 0 | 0 | 0.00% |
| low vegetation | 1267 | 1238 | 17 | 12 | 2.29% |
| tree | 6 | 6 | 0 | 0 | 0.00% |
| car | 19822 | 19822 | 0 | 0 | 0.00% |
| 合计 | 23066 | 22861 | 17 | 188 | 0.89% |

过滤行为比较集中：主要过滤了 `impervious surface` 的大面积 mask，少量过滤了 `low vegetation` 的异常尺度 mask；对 `building`、`tree`、`car` 没有实际影响。

## 3. 总体结果

本次实验的推荐论文主结果应使用 `dataset_*` 指标，即先累计 38 张图的混淆矩阵，再计算总体指标。

| 指标 | 数值 |
|---|---:|
| Overall Accuracy | 0.1518 |
| Mean IoU | 0.1013 |
| Mean F1 | 0.1753 |
| Mean Precision | 0.6660 |
| Mean Recall | 0.2265 |
| Frequency Weighted IoU | 0.0978 |

逐图平均指标如下，仅建议作为补充参考：

| 指标 | 数值 |
|---|---:|
| Average Overall Accuracy | 0.1475 |
| Average Mean IoU | 0.1006 |
| Average Mean F1 | 0.1685 |
| Average Mean Precision | 0.4807 |
| Average Mean Recall | 0.2337 |
| Average Frequency Weighted IoU | 0.0981 |

总体判断：

```text
Exp2-C 仍显著优于 Exp1/Exp2-A，但相对 Exp2-B 没有取得进一步提升；当前固定面积过滤规则带来了轻微负收益。
```

与 Exp1/Exp2-A 相比，Exp2-C 的数据集级 mIoU 从 0.0578 提升到 0.1013，Mean F1 从 0.1020 提升到 0.1753，OA 从 0.0892 提升到 0.1518。与 Exp2-B 相比，Exp2-C 的 mIoU 从 0.1052 降至 0.1013，OA 从 0.1659 降至 0.1518。

## 4. 与 Exp1 / Exp2-A / Exp2-B 的对比

| 实验 | Prompt / 后处理策略 | Prompt 调用数 | Valid masks | OA | mIoU | Mean F1 | FWIoU |
|---|---|---:|---:|---:|---:|---:|---:|
| Exp1 | 六类官方 prompt | 18468 | 6823 | 0.0892 | 0.0578 | 0.1020 | 0.0426 |
| Exp2-A | 5 类前景 prompt，无背景 prompt | 15390 | 6823 | 0.0892 | 0.0578 | 0.1020 | 0.0426 |
| Exp2-B | 5 类前景 prompt ensemble，无背景 prompt | 76950 | 23066 | 0.1659 | 0.1052 | 0.1806 | 0.1056 |
| Exp2-C | Exp2-B + 类别面积过滤 | 76950 | 23066 | 0.1518 | 0.1013 | 0.1753 | 0.0978 |

相对 Exp1，Exp2-C 的主要提升如下：

| 指标 | Exp1 | Exp2-C | 绝对提升 | 相对倍数 |
|---|---:|---:|---:|---:|
| OA | 0.0892 | 0.1518 | +0.0626 | 1.70x |
| mIoU | 0.0578 | 0.1013 | +0.0435 | 1.75x |
| Mean F1 | 0.1020 | 0.1753 | +0.0733 | 1.72x |
| Mean Recall | 0.1702 | 0.2265 | +0.0563 | 1.33x |
| FWIoU | 0.0426 | 0.0978 | +0.0552 | 2.30x |

相对 Exp2-B，Exp2-C 的变化如下：

| 指标 | Exp2-B | Exp2-C | 变化 |
|---|---:|---:|---:|
| OA | 0.1659 | 0.1518 | -0.0141 |
| mIoU | 0.1052 | 0.1013 | -0.0039 |
| Mean F1 | 0.1806 | 0.1753 | -0.0053 |
| Mean Precision | 0.6647 | 0.6660 | +0.0013 |
| Mean Recall | 0.2231 | 0.2265 | +0.0034 |
| FWIoU | 0.1056 | 0.0978 | -0.0078 |

从指标变化看，面积过滤略微提高了宏平均 precision 和 recall，但降低了 OA、mIoU、Mean F1 与 FWIoU。由于 OA 和 FWIoU 更受大面积类别影响，下降主要来自 `impervious surface` 覆盖损失。

## 5. 各类别结果

| 类别 | GT 占比 | 预测占比 | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 29.41% | 18.89% | 0.1723 | 0.2939 | 0.3759 | 0.2413 |
| building | 26.54% | 1.27% | 0.0458 | 0.0875 | 0.9614 | 0.0458 |
| low vegetation | 22.31% | 5.07% | 0.1340 | 0.2364 | 0.6384 | 0.1450 |
| tree | 15.66% | 0.00% | 0.0002 | 0.0004 | 0.9965 | 0.0002 |
| car | 1.41% | 0.31% | 0.2117 | 0.3495 | 0.9793 | 0.2127 |
| clutter/background | 4.65% | 74.47% | 0.0438 | 0.0840 | 0.0446 | 0.7140 |

与 Exp2-B 相比，类别变化如下：

| 类别 | Exp2-B IoU | Exp2-C IoU | 变化 |
|---|---:|---:|---:|
| impervious surface | 0.2014 | 0.1723 | -0.0291 |
| building | 0.0458 | 0.0458 | +0.0000 |
| low vegetation | 0.1310 | 0.1340 | +0.0030 |
| tree | 0.0002 | 0.0002 | +0.0000 |
| car | 0.2114 | 0.2117 | +0.0003 |
| clutter/background | 0.0415 | 0.0438 | +0.0023 |

类别级结果说明：

1. `impervious surface` 明显下降，是 Exp2-C 相对 Exp2-B 变差的主要来源。
2. `low vegetation`、`car` 和 `clutter/background` 有轻微提升，但幅度不足以抵消不透水表面的损失。
3. `building` 仍然保持高 precision、低 recall 的模式，说明预测到的建筑物较可信，但覆盖严重不足。
4. `tree` 仍几乎没有被有效召回，说明当前 prompt 和 RGB 输入下 SAM3 对树冠类别的候选生成能力很弱。

## 6. 混淆矩阵解读

数据集级混淆矩阵显示，Exp2-C 仍存在大量像素被预测为 `clutter/background`：

| 真实类别 | 主要误分方向 | 占该真实类别比例 |
|---|---|---:|
| impervious surface | clutter/background | 75.57% |
| building | clutter/background | 89.28% |
| low vegetation | clutter/background | 60.42% |
| tree | clutter/background | 68.23% |
| car | clutter/background | 74.27% |
| clutter/background | clutter/background | 71.40% |

与 Exp2-B 相比，预测为 `clutter/background` 的像素占比从 69.38% 增加到 74.47%。这说明面积过滤剔除的一部分前景 mask 并没有被其他正确前景类别补上，而是更多回退到了背景类。

其中最关键的是 `impervious surface`。Exp2-C 过滤掉了 176 个不透水表面的大面积候选 mask，使该类预测占比从 Exp2-B 的 24.08% 降至 18.89%，召回率也从 0.3049 降至 0.2413。由于不透水表面在 GT 中占比 29.41%，这类召回下降会直接拉低 OA 和 FWIoU。

## 7. 单图结果分析

按 `mean_iou` 排序，本次实验表现最好的 5 张图为：

| 排名 | 图像 | Mean IoU | OA |
|---:|---|---:|---:|
| 1 | `top_potsdam_7_10_RGB` | 0.1903 | 0.3991 |
| 2 | `top_potsdam_3_11_RGB` | 0.1554 | 0.1778 |
| 3 | `top_potsdam_5_14_RGB` | 0.1403 | 0.1912 |
| 4 | `top_potsdam_4_14_RGB` | 0.1394 | 0.1584 |
| 5 | `top_potsdam_7_12_RGB` | 0.1284 | 0.1859 |

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

Exp2-C 的结果说明：固定类别面积过滤在当前阈值下没有带来预期收益。

该实验原本希望过滤明显异常的候选 mask，减少错误覆盖。但从实际统计看，过滤比例只有 0.89%，且主要过滤的是 `impervious surface` 的大面积 mask。对于 Potsdam 这类高分辨率城市遥感影像，不透水表面本身就可能形成大范围连续区域，例如道路、广场、停车场和硬化地面。因此，简单按 patch 面积上限过滤，可能会误删本来有用的前景候选。

另一方面，`car` 类生成了 19822 个候选 mask，占候选 mask 总量的大多数，但当前面积阈值没有过滤任何汽车候选。这说明当前过滤规则并未作用于最密集的候选来源。即使汽车 IoU 较高，它在 GT 中只占 1.41%，无法主导总体指标。

因此，Exp2-C 更适合作为“朴素 mask 质量过滤不一定有效”的消融证据。它说明后续若要利用 mask 后处理，应避免只依赖固定面积阈值，而应考虑更稳健的质量信号，例如 mask 稳定性、边界一致性、类别竞争关系、图像纹理先验、DSM/nDSM 高程先验或 NDVI 植被先验。

## 9. 论文中如何表述这个结果

建议将 Exp2-C 定位为：

```text
视觉 prompt ensemble 后的类别面积先验过滤消融实验。
```

英文可写为：

```text
Exp2-C adds a fixed class-aware mask area filter on top of the visual prompt ensemble. Candidate masks are filtered before confidence-based patch fusion according to class-specific area-ratio thresholds. Compared with Exp2-B, the dataset-level mIoU decreases from 0.1052 to 0.1013 and OA decreases from 0.1659 to 0.1518, while mean precision changes only marginally from 0.6647 to 0.6660. The filter removes 205 out of 23066 candidate masks, mostly large impervious-surface masks. These results indicate that the simple fixed area prior does not provide a net benefit and may remove useful large-scale foreground masks in urban remote-sensing scenes.
```

中文可写为：

```text
Exp2-C 在视觉化 prompt ensemble 基础上加入固定的类别相关 mask 面积过滤，并在置信度融合之前剔除尺度异常的候选区域。与 Exp2-B 相比，数据集级 mIoU 从 0.1052 降至 0.1013，OA 从 0.1659 降至 0.1518，而 Mean Precision 仅从 0.6647 小幅变化到 0.6660。该过滤策略共剔除 23066 个候选 mask 中的 205 个，主要是不透水表面的大面积 mask。结果表明，简单固定面积先验没有带来净收益，并可能误删城市遥感场景中有用的大尺度前景候选。
```

## 10. 推荐论文表格

### 10.1 总体指标表

| Method | Active prompts | Post-processing | Fallback class | OA | mIoU | FWIoU | Mean F1 | Mean Precision | Mean Recall |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| SAM3 zero-shot | 6 official prompts | None | clutter/background | 0.0892 | 0.0578 | 0.0426 | 0.1020 | 0.5776 | 0.1702 |
| Exp2-A no background prompt | 5 foreground prompts | None | clutter/background | 0.0892 | 0.0578 | 0.0426 | 0.1020 | 0.5776 | 0.1702 |
| Exp2-B visual prompt ensemble | 25 foreground prompts | None | clutter/background | 0.1659 | 0.1052 | 0.1056 | 0.1806 | 0.6647 | 0.2231 |
| Exp2-C ensemble + mask filter | 25 foreground prompts | Class-aware area filter | clutter/background | 0.1518 | 0.1013 | 0.0978 | 0.1753 | 0.6660 | 0.2265 |

### 10.2 Mask 过滤统计表

| Class | Input masks | Kept masks | Removed small | Removed large | Removal ratio |
|---|---:|---:|---:|---:|---:|
| impervious surface | 1506 | 1330 | 0 | 176 | 11.69% |
| building | 465 | 465 | 0 | 0 | 0.00% |
| low vegetation | 1267 | 1238 | 17 | 12 | 2.29% |
| tree | 6 | 6 | 0 | 0 | 0.00% |
| car | 19822 | 19822 | 0 | 0 | 0.00% |
| Total | 23066 | 22861 | 17 | 188 | 0.89% |

### 10.3 类别 IoU 表

| Method | Impervious surface | Building | Low vegetation | Tree | Car | Clutter/background | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp2-B visual prompt ensemble | 0.2014 | 0.0458 | 0.1310 | 0.0002 | 0.2114 | 0.0415 | 0.1052 |
| Exp2-C ensemble + mask filter | 0.1723 | 0.0458 | 0.1340 | 0.0002 | 0.2117 | 0.0438 | 0.1013 |

## 11. 结论

Exp2-C 的主要结论可以概括为：

```text
视觉 prompt ensemble 是有效的，但当前固定面积过滤不是有效增益项。
```

具体来说，Exp2-C 相对 Exp1/Exp2-A 仍保持明显提升，说明 Exp2-B 引入的视觉化 prompt ensemble 仍然是主要有效因素；但相对 Exp2-B，新增的面积过滤没有改善总体结果，反而降低 OA、mIoU 和 FWIoU。其原因是过滤主要作用于大面积不透水表面 mask，导致前景覆盖下降，更多像素回退到 `clutter/background`。

因此，在论文中不建议把 Exp2-C 作为优于 Exp2-B 的方法，而应作为后处理消融：它证明朴素的固定面积过滤不足以解决 SAM3 零样本遥感分割中的背景过预测和类别召回不足问题，也为后续引入更强的多模态遥感先验提供动机。

