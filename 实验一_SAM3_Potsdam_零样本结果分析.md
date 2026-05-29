# 实验一：SAM3 + Potsdam 零样本分割结果分析

## 1. 本次运行概况

本次实验已完成 `exp1_sam3_potsdam_zeroshot_baseline.py` 的全量运行，结果保存于：

```text
results_exp1_sam3_potsdam_zeroshot_baseline/
```

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
| 总文本 prompt 调用数 | 18468 |
| 评估方式 | noBoundary 标签，黑色 `(0,0,0)` 映射为 `ignore_index=255` |

输出文件数量检查：

| 输出类型 | 数量 | 说明 |
|---|---:|---|
| `results_exp1_sam3_potsdam_zeroshot_baseline/predictions` | 152 | 38 张图，每张包含彩色预测、彩色 GT、预测 ID、GT ID |
| `results_exp1_sam3_potsdam_zeroshot_baseline/visualizations` | 38 | 每张图一张对比可视化 |
| `results_exp1_sam3_potsdam_zeroshot_baseline/metrics` | 42 | 38 个单图 JSON + `overall_metrics.json`、`per_image_metrics.csv`、`config_snapshot.yaml`、`run_context.json` |

关键结果文件：

```text
results_exp1_sam3_potsdam_zeroshot_baseline/metrics/overall_metrics.json
results_exp1_sam3_potsdam_zeroshot_baseline/metrics/per_image_metrics.csv
results_exp1_sam3_potsdam_zeroshot_baseline/metrics/run_context.json
results_exp1_sam3_potsdam_zeroshot_baseline/metrics/config_snapshot.yaml
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
SAM3 在当前 Potsdam RGB 零样本设置下表现很弱。
主指标 mIoU 仅为 0.0578，Mean F1 仅为 0.1020。
```

这不是一个可以直接宣称“有效分割”的结果。它更适合作为论文中的负基线或动机实验，用来说明通用 SAM3 直接迁移到遥感语义分割时存在明显域差异和类别语义不匹配问题。

## 3. 各类别结果

| 类别 | GT 占比 | 预测占比 | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 29.41% | 20.47% | 0.1197 | 0.2138 | 0.2605 | 0.1813 |
| building | 26.54% | 0.31% | 0.0112 | 0.0221 | 0.9656 | 0.0112 |
| low vegetation | 22.31% | 0.07% | 0.0007 | 0.0014 | 0.2231 | 0.0007 |
| tree | 15.66% | 0.00% | 0.0001 | 0.0002 | 0.9955 | 0.0001 |
| car | 1.41% | 0.26% | 0.1778 | 0.3019 | 0.9823 | 0.1784 |
| clutter/background | 4.65% | 78.89% | 0.0375 | 0.0723 | 0.0383 | 0.6495 |

最重要的现象是预测类别分布严重偏向 `clutter/background`：

```text
真实标签中 clutter/background 只占 4.65%
预测结果中 clutter/background 占 78.89%
```

这说明当前 prompt 与后处理策略下，SAM3 对大量道路、建筑、植被、树木和车辆区域没有生成有效类别 mask，最终这些像素被回退到 `clutter/background`。

## 4. 混淆矩阵解读

数据集级混淆矩阵显示，主要错误方向都是被预测为 `clutter/background`：

| 真实类别 | 主要误分方向 | 占该真实类别比例 |
|---|---|---:|
| impervious surface | clutter/background | 81.85% |
| building | clutter/background | 94.13% |
| low vegetation | clutter/background | 63.78% |
| tree | clutter/background | 73.10% |
| car | clutter/background | 80.11% |
| clutter/background | clutter/background | 64.95% |

这解释了为什么总体 OA 和 mIoU 都很低。

同时也解释了为什么 `Mean Precision = 0.5776` 看起来相对较高：模型对 `building`、`tree`、`car` 的预测数量非常少，一旦预测出来通常较准，因此 precision 高；但这些类别的 recall 极低，说明绝大多数真实目标没有被召回。

典型例子：

| 类别 | Precision | Recall | 解释 |
|---|---:|---:|---|
| building | 0.9656 | 0.0112 | 预测为 building 的像素较准，但几乎没有召回真实建筑 |
| tree | 0.9955 | 0.0001 | 几乎不预测 tree，因此 recall 接近 0 |
| car | 0.9823 | 0.1784 | car 的 precision 较高，但仍漏掉大量车辆 |
| clutter/background | 0.0383 | 0.6495 | 大量非背景像素被误归入 clutter，precision 极低 |

因此论文中不应只看 precision。更合理的主结论应基于 mIoU、Mean F1、Recall 和混淆矩阵。

## 5. 单图结果分析

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

## 6. 结果低的主要原因

### 6.1 直接使用官方类别名 prompt 对通用模型不友好

当前 prompt 是：

```text
impervious surface
building
low vegetation
tree
car
background clutter
```

其中 `building`、`car`、`tree` 是通用视觉概念，SAM3 较容易理解；但 `impervious surface`、`low vegetation`、`background clutter` 更接近遥感数据集术语，对通用开放词汇模型不一定友好。

尤其是：

```text
impervious surface
low vegetation
background clutter
```

这些类别不是普通图像中的自然物体类别，而是遥感语义分割中的区域类别。SAM3 作为通用分割模型，很可能无法稳定生成覆盖整片道路、草地、低矮植被或杂类背景的 mask。

### 6.2 SAM3 输出偏实例/对象 mask，而 Potsdam 是密集语义分割

SAM3 更擅长根据文本定位对象或区域 mask。Potsdam 评估要求每个有效像素都归入 6 个语义类别之一。

当 SAM3 没有对某个像素生成有效 mask 时，脚本会将该像素回退为 `clutter/background`。本次结果显示大量像素都走到了这个回退路径，最终导致 `clutter/background` 被严重过预测。

### 6.3 遥感俯视视角与通用图像分布差异明显

Potsdam 是高分辨率航空影像，存在俯视视角、细粒度纹理、小目标车辆、屋顶/道路/植被光谱混淆等问题。通用 SAM3 未经遥感适配时，对这些类别的语义边界和区域定义不稳定。

### 6.4 RGB-only 输入缺少高度和近红外信息

本实验只使用 RGB 正射影像，没有使用 DSM、nDSM、IRRG 或 RGBIR。Potsdam 中树木与低矮植被、建筑与不透水表面之间常需要高度或近红外信息辅助区分。RGB-only 零样本设置本身难度很高。

## 7. 论文中如何表述这个结果

建议将该实验定位为：

```text
SAM3 直接零样本迁移到 Potsdam 遥感语义分割的基线实验。
```

不建议表述为“取得良好效果”。更准确的表述是：

```text
The direct zero-shot application of SAM3 to Potsdam RGB orthophotos yields limited semantic segmentation performance. The dataset-level mIoU is 0.0578 and the mean F1 is 0.1020. The confusion matrix shows that most pixels are assigned to clutter/background, indicating a substantial mismatch between SAM3's open-vocabulary mask generation and dense remote-sensing semantic segmentation requirements.
```

中文可写为：

```text
直接将 SAM3 应用于 Potsdam RGB 正射影像的零样本语义分割时，模型表现较弱。数据集级 mIoU 仅为 0.0578，Mean F1 为 0.1020。混淆矩阵显示，大量不透水表面、建筑物、低矮植被、树木和车辆像素被预测为 clutter/background，说明通用 SAM3 的开放词汇 mask 生成能力与遥感密集语义分割任务之间存在明显不匹配。
```

## 8. 推荐论文表格

### 8.1 总体指标表

| Method | Input | OA | mIoU | FWIoU | Mean F1 | Mean Precision | Mean Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| SAM3 zero-shot | RGB | 0.0892 | 0.0578 | 0.0426 | 0.1020 | 0.5776 | 0.1702 |

### 8.2 各类别 IoU 表

| Method | Impervious | Building | Low Veg. | Tree | Car | Clutter | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAM3 zero-shot | 0.1197 | 0.0112 | 0.0007 | 0.0001 | 0.1778 | 0.0375 | 0.0578 |

### 8.3 各类别 F1 表

| Method | Impervious | Building | Low Veg. | Tree | Car | Clutter | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAM3 zero-shot | 0.2138 | 0.0221 | 0.0014 | 0.0002 | 0.3019 | 0.0723 | 0.1020 |

## 9. 后续改进方向

如果后续要提升实验效果，可以考虑以下方向：

1. Prompt variants / prompt ensemble  
   用更视觉化的提示替代单一官方类名，例如 `road, pavement, parking lot, sidewalk` 对应不透水表面，`grass, lawn, low plants` 对应低矮植被。

2. 不主动预测 `clutter/background`  
   将 clutter/background 只作为未被其他类别覆盖的 fallback，避免模型用不明确的 `background clutter` prompt 生成大量背景响应。

3. 分类别后处理  
   对 car 使用小目标增强，对 building 使用面积或形状约束，对 vegetation/tree 使用颜色或纹理先验。

4. 引入 RGBIR 或 DSM/nDSM  
   用近红外和高度信息改善树木/低矮植被、建筑/道路的区分。

5. 遥感适配或轻量微调  
   在 Potsdam 或类似遥感数据上进行 prompt tuning、adapter tuning 或 mask 后处理学习。

## 10. 本次结果的结论

本次实验完成了 SAM3 在 Potsdam RGB 数据上的全量零样本评估。结果表明：

```text
直接使用 SAM3 + 官方类别名 prompt 进行 Potsdam 密集语义分割并不可靠。
```

其主要失败模式是：

```text
大量真实类别像素被预测为 clutter/background。
```

因此，该实验适合作为论文中的基础零样本基线和问题动机，为后续 prompt 改进、多模态输入或遥感适配方法提供对比依据。
