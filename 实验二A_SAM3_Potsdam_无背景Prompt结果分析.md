# 实验二A：SAM3 + Potsdam 无背景 Prompt 结果分析

## 1. 本次运行概况

本次实验已完成 `exp2a_sam3_potsdam_no_background_prompt.py` 的全量运行，结果保存于：

```text
results_exp2a_sam3_potsdam_no_background_prompt/
```

Exp2-A 的实验变量是：

```text
不主动向 SAM3 输入 clutter/background 文本 prompt。
```

需要注意的是，`clutter/background` 并没有从评估类别中删除。它仍然是 Potsdam 标准 6 类之一，也仍然作为未被任何有效 SAM3 mask 覆盖像素的 fallback 类别。

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
| 总文本 prompt 调用数 | 15390 |
| fallback 类别 | `clutter/background` |
| 评估方式 | noBoundary 标签，黑色 `(0,0,0)` 映射为 `ignore_index=255` |

输出文件数量检查：

| 输出类型 | 数量 | 说明 |
|---|---:|---|
| `results_exp2a_sam3_potsdam_no_background_prompt/predictions` | 152 | 38 张图，每张包含彩色预测、彩色 GT、预测 ID、GT ID |
| `results_exp2a_sam3_potsdam_no_background_prompt/visualizations` | 38 | 每张图一张对比可视化 |
| `results_exp2a_sam3_potsdam_no_background_prompt/metrics` | 43 | 38 个单图 JSON + `overall_metrics.json`、`per_image_metrics.csv`、`experiment_metadata.json`、`config_snapshot.yaml`、`run_context.json` |

关键结果文件：

```text
results_exp2a_sam3_potsdam_no_background_prompt/metrics/overall_metrics.json
results_exp2a_sam3_potsdam_no_background_prompt/metrics/per_image_metrics.csv
results_exp2a_sam3_potsdam_no_background_prompt/metrics/experiment_metadata.json
results_exp2a_sam3_potsdam_no_background_prompt/metrics/run_context.json
results_exp2a_sam3_potsdam_no_background_prompt/metrics/config_snapshot.yaml
```

当前配置中 `metrics.save_confusion_matrix=false`，因此没有单独生成 `dataset_confusion_matrix.csv`；数据集级混淆矩阵保存在 `overall_metrics.json` 的 `dataset_confusion_matrix` 字段中。

## 2. 总体结果

本次实验的推荐论文主结果应使用 `dataset_*` 指标，即先累计 38 张图的混淆矩阵，再计算总体指标。

| 指标 | 数值 |
|---|---:|
| Overall Accuracy | 0.0892 |
| Mean IoU | 0.0578 |
| Mean F1 | 0.1020 |
| Mean Precision | 0.5776 |
| Mean Recall | 0.1702 |
| Frequency Weighted IoU | 0.0426 |

逐图平均指标如下，仅建议作为补充参考：

| 指标 | 数值 |
|---|---:|
| Average Overall Accuracy | 0.0865 |
| Average Mean IoU | 0.0559 |
| Average Mean F1 | 0.0948 |
| Average Mean Precision | 0.3009 |
| Average Mean Recall | 0.1787 |
| Average Frequency Weighted IoU | 0.0449 |

总体判断：

```text
去掉 clutter/background 主动 prompt 后，Exp2-A 的数据集级指标与 Exp1 完全一致。
```

这说明在当前配置和 SAM3 输出行为下，`background clutter` 这个文本 prompt 没有为最终预测提供额外有效 mask。Exp2-A 的价值不是提升精度，而是验证“背景类更适合作为 residual/fallback 类别，而不是主动开放词汇查询类别”这一设计判断。

## 3. 与 Exp1 的对比

| 实验 | 主动 prompt 类别数 | Prompt 调用数 | Valid masks | Empty prompt outputs | OA | mIoU | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp1：六类官方 prompt | 6 | 18468 | 6823 | 16645 | 0.0892 | 0.0578 | 0.1020 |
| Exp2-A：不主动 prompt 背景 | 5 | 15390 | 6823 | 13567 | 0.0892 | 0.0578 | 0.1020 |

Exp2-A 相比 Exp1 少了 3078 次 prompt 调用，正好等于 38 张图乘以每张 81 个 patch：

```text
38 * 81 = 3078
```

这说明每个 patch 少调用了一次 `background clutter` prompt。与此同时：

```text
valid_masks 不变：6823
empty_prompt_outputs 减少：3078
```

因此可以推断，Exp1 中的 `background clutter` prompt 在每个 patch 上都没有产生有效 mask，或者至少没有产生通过当前阈值并参与最终融合的有效 mask。去掉它后，最终预测图、混淆矩阵和指标保持一致。

## 4. 各类别结果

| 类别 | GT 占比 | 预测占比 | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 29.41% | 20.47% | 0.1197 | 0.2138 | 0.2605 | 0.1813 |
| building | 26.54% | 0.31% | 0.0112 | 0.0221 | 0.9656 | 0.0112 |
| low vegetation | 22.31% | 0.07% | 0.0007 | 0.0014 | 0.2231 | 0.0007 |
| tree | 15.66% | 0.00% | 0.0001 | 0.0002 | 0.9955 | 0.0001 |
| car | 1.41% | 0.26% | 0.1778 | 0.3019 | 0.9823 | 0.1784 |
| clutter/background | 4.65% | 78.89% | 0.0375 | 0.0723 | 0.0383 | 0.6495 |

最重要的现象仍然是预测类别分布严重偏向 `clutter/background`：

```text
真实标签中 clutter/background 只占 4.65%
预测结果中 clutter/background 占 78.89%
```

但这个背景过预测不是由主动 `background clutter` prompt 直接造成的。Exp2-A 已经去掉了背景 prompt，预测仍然大量落入 `clutter/background`，说明主要原因是前景类别 mask 覆盖不足，未覆盖像素被 fallback 到背景类。

## 5. 混淆矩阵解读

数据集级混淆矩阵显示，主要错误方向都是被预测为 `clutter/background`：

| 真实类别 | 主要误分方向 | 占该真实类别比例 |
|---|---|---:|
| impervious surface | clutter/background | 81.85% |
| building | clutter/background | 94.13% |
| low vegetation | clutter/background | 63.78% |
| tree | clutter/background | 73.10% |
| car | clutter/background | 80.11% |
| clutter/background | clutter/background | 64.95% |

这说明 Exp2-A 的失败模式与 Exp1 一致：SAM3 没有为大多数真实前景区域生成足够覆盖的有效 mask，导致道路、建筑、植被、树木和车辆大量回退到背景类。

同时，`building`、`tree`、`car` 的 precision 很高但 recall 极低：

| 类别 | Precision | Recall | 解释 |
|---|---:|---:|---|
| building | 0.9656 | 0.0112 | 预测为 building 的像素较准，但几乎没有召回真实建筑 |
| tree | 0.9955 | 0.0001 | 几乎不预测 tree，因此 recall 接近 0 |
| car | 0.9823 | 0.1784 | car 的 precision 较高，但仍漏掉大量车辆 |
| clutter/background | 0.0383 | 0.6495 | 大量非背景像素被误归入 clutter，precision 极低 |

## 6. 单图结果分析

按 `mean_iou` 排序，本次实验表现最好的 5 张图为：

| 排名 | 图像 | Mean IoU |
|---:|---|---:|
| 1 | `top_potsdam_7_10_RGB` | 0.1075 |
| 2 | `top_potsdam_3_11_RGB` | 0.1001 |
| 3 | `top_potsdam_5_15_RGB` | 0.0876 |
| 4 | `top_potsdam_7_9_RGB` | 0.0864 |
| 5 | `top_potsdam_7_12_RGB` | 0.0856 |

表现最差的 5 张图为：

| 排名 | 图像 | Mean IoU |
|---:|---|---:|
| 1 | `top_potsdam_4_12_RGB` | 0.0000 |
| 2 | `top_potsdam_6_7_RGB` | 0.0171 |
| 3 | `top_potsdam_5_12_RGB` | 0.0211 |
| 4 | `top_potsdam_7_8_RGB` | 0.0284 |
| 5 | `top_potsdam_3_12_RGB` | 0.0307 |

其中 `top_potsdam_4_12_RGB` 的单图指标全为 0，是因为对应 noBoundary 标签中有效像素数为 0：

```text
valid_pixels = 0
ignored_pixels = 36000000
```

这张图不影响 `dataset_*` 累计指标，因为其混淆矩阵全为 0；但它会拉低 `average_*` 逐图平均指标。因此论文主结果应使用 `dataset_*`，不要使用 `average_*` 作为核心结论。

## 7. 结果意义

Exp2-A 的核心结论不是“性能提升”，而是“背景 prompt 的作用被排除了”。

本实验说明：

```text
在当前 SAM3 + Potsdam zero-shot 设置下，主动输入 background clutter 并不是背景过预测的主要原因。
```

背景过预测更可能来自：

1. 前景类别 prompt 无法充分召回遥感密集语义区域。
2. SAM3 输出偏对象/实例 mask，而 Potsdam 需要每个像素都有语义类别。
3. 未被有效 mask 覆盖的像素统一 fallback 到 `clutter/background`。
4. `impervious surface`、`low vegetation` 等遥感区域类不适合作为单一通用文本 prompt。

因此，后续改进不应继续围绕“是否 prompt 背景类”单点调整，而应重点转向前景 prompt ensemble、mask 过滤、多模态输入和遥感先验约束。

## 8. 论文中如何表述这个结果

建议将 Exp2-A 定位为：

```text
背景 prompt 策略消融实验。
```

英文可写为：

```text
In Exp2-A, the clutter/background category is removed from active text prompting and used only as a residual fallback class. The dataset-level results remain identical to the six-prompt baseline, with an mIoU of 0.0578 and a mean F1 of 0.1020. The number of prompt calls decreases from 18468 to 15390, while the number of valid masks remains unchanged. This indicates that the background prompt does not contribute effective masks under the current setting, and that the dominant background predictions are mainly caused by insufficient foreground mask coverage rather than active background querying.
```

中文可写为：

```text
在 Exp2-A 中，本文不再主动向 SAM3 输入 clutter/background 文本提示，而是将该类别仅作为未覆盖像素的 residual fallback 类别。实验结果与六类 prompt 基线完全一致，数据集级 mIoU 仍为 0.0578，Mean F1 仍为 0.1020。与此同时，prompt 调用次数由 18468 降至 15390，而有效 mask 数量保持 6823 不变。这说明在当前设置下，背景类 prompt 并未产生有效分割贡献，背景过预测主要源于前景类别 mask 覆盖不足，而不是主动背景查询本身。
```

## 9. 推荐论文表格

### 9.1 总体指标表

| Method | Active prompts | Fallback class | OA | mIoU | FWIoU | Mean F1 | Mean Precision | Mean Recall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SAM3 zero-shot | 6 classes | clutter/background | 0.0892 | 0.0578 | 0.0426 | 0.1020 | 0.5776 | 0.1702 |
| Exp2-A no background prompt | foreground 5 classes | clutter/background | 0.0892 | 0.0578 | 0.0426 | 0.1020 | 0.5776 | 0.1702 |

### 9.2 Prompt 调用与 mask 统计表

| Method | Patch calls | Prompt calls | Empty prompt outputs | Valid masks |
|---|---:|---:|---:|---:|
| SAM3 zero-shot | 3078 | 18468 | 16645 | 6823 |
| Exp2-A no background prompt | 3078 | 15390 | 13567 | 6823 |

### 9.3 各类别 IoU 表

| Method | Impervious | Building | Low Veg. | Tree | Car | Clutter | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp2-A no background prompt | 0.1197 | 0.0112 | 0.0007 | 0.0001 | 0.1778 | 0.0375 | 0.0578 |

## 10. 本次结果的结论

本次实验完成了 SAM3 在 Potsdam RGB 数据上的背景 prompt 消融。结果表明：

```text
去掉 clutter/background 主动 prompt 不会改变最终分割结果。
```

更重要的是：

```text
当前背景过预测主要不是 background prompt 本身造成的，而是前景 mask 召回不足和 fallback 机制共同导致的。
```

因此，该实验为后续 Exp2-B 的 prompt ensemble 提供了清晰动机：如果要改善结果，关键应是增强前景类别的候选 mask 覆盖，而不是继续调整背景类文本提示。
