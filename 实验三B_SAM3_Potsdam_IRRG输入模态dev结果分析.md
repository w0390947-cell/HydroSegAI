# 实验三B：SAM3 + Potsdam IRRG 输入模态 dev 结果分析

## 1. 本次运行概况

本次实验已完成 `exp3b_sam3_potsdam_irrg_modality.py` 的 development split 运行，结果保存于：

```text
results_exp3b_sam3_potsdam_irrg_modality_dev/
```

Exp3-B 的实验变量是：

```text
在固定 Exp2-B 视觉 prompt ensemble 策略后，仅将输入模态从 RGB 真彩色影像改为 IRRG 假彩色影像。
```

因此，本实验与 Exp3-A dev 的主要区别只有输入影像目录和文件模板不同；prompt、patch 设置、score threshold、fallback 规则、mask 过滤开关和评估 split 均保持一致。

本次运行使用的数据与配置如下：

| 项目 | 内容 |
|---|---|
| 输入影像 | `Potsdam/3_Ortho_IRRG/3_Ortho_IRRG` |
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
| 总有效候选 mask 数 | 8171 |
| mask 面积过滤 | 关闭 |
| fallback 类别 | `clutter/background` |
| 评估方式 | noBoundary 标签，黑色 `(0,0,0)` 映射为 `ignore_index=255` |

输出文件数量检查：

| 输出类型 | 数量 | 说明 |
|---|---:|---|
| `results_exp3b_sam3_potsdam_irrg_modality_dev/predictions` | 96 | 24 张图，每张包含彩色预测、彩色 GT、预测 ID、GT ID |
| `results_exp3b_sam3_potsdam_irrg_modality_dev/visualizations` | 24 | 每张图一张对比可视化 |
| `results_exp3b_sam3_potsdam_irrg_modality_dev/metrics` | 29 | 24 个单图 JSON + `overall_metrics.json`、`per_image_metrics.csv`、`experiment_metadata.json`、`config_snapshot.yaml`、`run_context.json` |
| `results_exp3b_sam3_potsdam_irrg_modality_dev/logs` | 1 | 本次运行日志 |

关键结果文件：

```text
results_exp3b_sam3_potsdam_irrg_modality_dev/metrics/overall_metrics.json
results_exp3b_sam3_potsdam_irrg_modality_dev/metrics/per_image_metrics.csv
results_exp3b_sam3_potsdam_irrg_modality_dev/metrics/experiment_metadata.json
results_exp3b_sam3_potsdam_irrg_modality_dev/metrics/run_context.json
results_exp3b_sam3_potsdam_irrg_modality_dev/metrics/config_snapshot.yaml
```

当前配置中 `metrics.save_confusion_matrix=false`，因此没有单独生成 `dataset_confusion_matrix.csv`；数据集级混淆矩阵保存在 `overall_metrics.json` 的 `dataset_confusion_matrix` 字段中。

## 2. 固定 Prompt 与模态设置

Exp3-B 固定继承 Exp2-B / Exp3-A 的视觉 prompt ensemble：

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
| `input_modality.name` | `irrg` |
| `input_modality.description` | `Infrared-red-green orthophoto input.` |
| 图像来源 | Potsdam IRRG 假彩色正射影像 |
| 图像文件模板 | `{image_id}_IRRG.tif` |
| RGB 真彩色 | 未使用 |
| DSM / nDSM | 未使用 |
| RGBIR 派生通道 | 未使用 |
| 多视图融合 | 未使用 |

## 3. 总体结果

本次实验的推荐论文主结果应使用 `dataset_*` 指标，即先累计 24 张 dev 图像的混淆矩阵，再计算总体指标。

| 指标 | 数值 |
|---|---:|
| Overall Accuracy | 0.1284 |
| Mean IoU | 0.0629 |
| Mean F1 | 0.1098 |
| Mean Precision | 0.5159 |
| Mean Recall | 0.1879 |
| Frequency Weighted IoU | 0.0548 |

逐图平均指标如下，仅建议作为补充参考：

| 指标 | 数值 |
|---|---:|
| Average Overall Accuracy | 0.1280 |
| Average Mean IoU | 0.0638 |
| Average Mean F1 | 0.1072 |
| Average Mean Precision | 0.2642 |
| Average Mean Recall | 0.1953 |
| Average Frequency Weighted IoU | 0.0616 |

总体判断：

```text
在 development split 上，IRRG 输入模态明显低于 RGB 输入参照；近红外假彩色并没有在当前 SAM3 + 视觉 prompt ensemble 设置下转化为有效的植被类别增益。
```

## 4. 与 Exp3-A RGB dev 的对比

Exp3-A 与 Exp3-B 使用同一 dev split、同一 prompt ensemble、同一推理参数，因此可以作为模态敏感性实验直接比较。

| 实验 | Input modality | Valid masks | OA | mIoU | Mean F1 | Mean Precision | Mean Recall | FWIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Exp3-A | RGB | 12402 | 0.1677 | 0.1032 | 0.1781 | 0.6506 | 0.2284 | 0.1048 |
| Exp3-B | IRRG | 8171 | 0.1284 | 0.0629 | 0.1098 | 0.5159 | 0.1879 | 0.0548 |
| 变化 | IRRG - RGB | -4231 | -0.0393 | -0.0404 | -0.0682 | -0.1347 | -0.0405 | -0.0499 |

从总体指标看，IRRG 相对 RGB 全面下降：

1. mIoU 从 0.1032 降至 0.0629。
2. Mean F1 从 0.1781 降至 0.1098。
3. OA 从 0.1677 降至 0.1284。
4. FWIoU 从 0.1048 降至 0.0548。
5. 有效候选 mask 数从 12402 降至 8171，说明 IRRG 触发的 SAM3 有效 mask 更少。

这表明，虽然 IRRG 包含近红外信息，但它作为假彩色三通道图像输入给 SAM3 时，并不一定符合 SAM3 在自然图像上的视觉表征分布。

## 5. 各类别结果

| 类别 | GT 占比 | 预测占比 | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 28.34% | 35.57% | 0.1738 | 0.2961 | 0.2660 | 0.3339 |
| building | 27.83% | 0.12% | 0.0037 | 0.0074 | 0.8422 | 0.0037 |
| low vegetation | 23.20% | 0.04% | 0.0001 | 0.0002 | 0.0636 | 0.0001 |
| tree | 14.49% | 0.03% | 0.0019 | 0.0038 | 0.8956 | 0.0019 |
| car | 1.36% | 0.21% | 0.1514 | 0.2630 | 0.9808 | 0.1518 |
| clutter/background | 4.78% | 64.03% | 0.0462 | 0.0884 | 0.0475 | 0.6360 |

类别级结果说明：

1. `impervious surface` 的 recall 从 RGB 的 0.3111 提升到 0.3339，但 precision 从 0.3695 降至 0.2660，说明 IRRG 更容易把其他类别误吸收到不透水表面。
2. `building` 几乎崩塌，IoU 从 RGB 的 0.0460 降至 0.0037，预测占比仅 0.12%。
3. `low vegetation` 并没有从近红外输入中受益，IoU 从 RGB 的 0.1270 降至 0.0001，预测占比仅 0.04%。
4. `tree` 有极小提升，IoU 从 0.0004 增至 0.0019，但绝对值仍接近 0，不能视为有效召回。
5. `car` 从 0.1970 降至 0.1514，仍保持高 precision，但 recall 降低。
6. `clutter/background` 的预测占比从 RGB 的 69.75% 降至 64.03%，但这不是有效提升，因为减少的背景预测主要转移到了 `impervious surface`，而不是正确前景类别。

类别 IoU 与 RGB 对比如下：

| 类别 | RGB IoU | IRRG IoU | 变化 |
|---|---:|---:|---:|
| impervious surface | 0.2032 | 0.1738 | -0.0295 |
| building | 0.0460 | 0.0037 | -0.0423 |
| low vegetation | 0.1270 | 0.0001 | -0.1269 |
| tree | 0.0004 | 0.0019 | +0.0015 |
| car | 0.1970 | 0.1514 | -0.0456 |
| clutter/background | 0.0455 | 0.0462 | +0.0007 |

最关键的类别变化是 `low vegetation` 的大幅下降。这与“近红外有利于植被”的物理直觉相反，说明这里的主要问题不是数据中是否含有 NIR 信息，而是 SAM3 能否把 IRRG 假彩色外观与文本 prompt 中的 `grass`、`lawn`、`low plants` 等自然图像概念对齐。

## 6. 混淆矩阵解读

数据集级混淆矩阵显示，IRRG 的主要错误不只是回退到 `clutter/background`，还包括大量前景被吸收到 `impervious surface`。

| 真实类别 | 误分为 clutter/background 的比例 |
|---|---:|
| impervious surface | 66.60% |
| building | 88.58% |
| low vegetation | 39.64% |
| tree | 49.68% |
| car | 78.12% |
| clutter/background | 63.60% |

表面上看，`low vegetation` 和 `tree` 误分为背景的比例低于 RGB；但这不是好现象，因为大量植被没有被正确识别，而是被误预测为 `impervious surface`：

| 真实类别 | 误分为 impervious surface 的比例 |
|---|---:|
| building | 11.04% |
| low vegetation | 60.35% |
| tree | 49.90% |
| clutter/background | 35.96% |

因此，IRRG 降低背景预测占比的方式并不是增加正确前景覆盖，而是把更多非硬化地表区域错误归入 `impervious surface`。这解释了为什么 `clutter/background` 预测占比从 RGB 的 69.75% 降到 64.03%，但总体 OA、mIoU 和 FWIoU 仍显著下降。

## 7. 单图结果分析

按 `mean_iou` 排序，本次实验表现最好的 5 张图为：

| 排名 | 图像 | Mean IoU | OA |
|---:|---|---:|---:|
| 1 | `top_potsdam_7_10_IRRG` | 0.1360 | 0.3497 |
| 2 | `top_potsdam_7_9_IRRG` | 0.1026 | 0.2635 |
| 3 | `top_potsdam_3_11_IRRG` | 0.1026 | 0.1281 |
| 4 | `top_potsdam_2_11_IRRG` | 0.0975 | 0.1727 |
| 5 | `top_potsdam_7_12_IRRG` | 0.0897 | 0.2181 |

表现最差的 5 张图为：

| 排名 | 图像 | Mean IoU | OA |
|---:|---|---:|---:|
| 1 | `top_potsdam_4_10_IRRG` | 0.0262 | 0.0639 |
| 2 | `top_potsdam_3_12_IRRG` | 0.0285 | 0.0561 |
| 3 | `top_potsdam_7_8_IRRG` | 0.0294 | 0.0447 |
| 4 | `top_potsdam_6_7_IRRG` | 0.0341 | 0.0682 |
| 5 | `top_potsdam_7_7_IRRG` | 0.0378 | 0.0826 |

与 RGB 参照相比，IRRG 的最好单图 `top_potsdam_7_10_IRRG` 仍低于 RGB 对应结果 `top_potsdam_7_10_RGB` 的 Mean IoU 0.1897。这说明 IRRG 下降不是由个别异常图像造成，而是整体模态适配性更弱。

## 8. 结果意义

Exp3-B 说明，不能简单假设“近红外输入一定提升遥感语义分割”。对于当前 SAM3 开放词汇推理框架，IRRG 至少存在两个问题：

1. IRRG 假彩色改变了物体外观分布，可能削弱 SAM3 对 `building`、`car`、`grass`、`lawn` 等自然图像概念的匹配。
2. NIR 信息虽然增强了植被光谱差异，但当前模型和 prompt 并没有把这种差异转化为正确的 vegetation/tree mask，反而出现 `low vegetation` 大面积误归入 `impervious surface` 的现象。

因此，Exp3-B 的论文定位应是：

```text
IRRG 作为单独三通道输入并不能替代 RGB；在当前 SAM3 零样本开放词汇设置下，RGB 仍是更稳定的输入模态。
```

这也为后续 Exp3-C / Exp3-D 提供动机：与其直接用 IRRG 替换 RGB，不如考虑 RGBIR 派生通道或多视图融合，让 RGB 保留自然图像外观对齐能力，同时把 NIR 信息作为补充而不是完全替代。

## 9. 论文中如何表述这个结果

建议将 Exp3-B 定位为：

```text
IRRG 假彩色输入模态消融实验。
```

英文可写为：

```text
Exp3-B replaces the RGB orthophoto input with IRRG false-color imagery while keeping the same visual prompt ensemble and fallback strategy as Exp3-A. On the development split, IRRG decreases the dataset-level mIoU from 0.1032 to 0.0629 and the mean F1 from 0.1781 to 0.1098. The number of valid masks also drops from 12402 to 8171. Contrary to the expectation that near-infrared information may benefit vegetation classes, the IoU of low vegetation decreases from 0.1270 to 0.0001, and many vegetation pixels are confused with impervious surface. This suggests that IRRG false-color images are less aligned with SAM3's open-vocabulary visual-text representations than natural RGB images.
```

中文可写为：

```text
Exp3-B 在保持 Exp3-A 相同视觉 prompt ensemble 和 fallback 策略的基础上，将输入由 RGB 真彩色影像替换为 IRRG 假彩色影像。在 development split 上，IRRG 使数据集级 mIoU 从 0.1032 降至 0.0629，Mean F1 从 0.1781 降至 0.1098，有效 mask 数也从 12402 降至 8171。与近红外信息可能有利于植被类别的直觉不同，low vegetation 的 IoU 从 0.1270 降至 0.0001，大量植被像素被误分为 impervious surface。这说明在当前 SAM3 开放词汇零样本设置下，IRRG 假彩色图像与模型的视觉-文本表征对齐程度弱于自然 RGB 图像。
```

## 10. 推荐论文表格

### 10.1 Exp3 dev 总体指标表

| Method | Split | Input modality | Active prompts | Mask filter | Valid masks | OA | mIoU | FWIoU | Mean F1 | Mean Precision | Mean Recall |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Exp3-A | dev | RGB | 25 foreground prompts | disabled | 12402 | 0.1677 | 0.1032 | 0.1048 | 0.1781 | 0.6506 | 0.2284 |
| Exp3-B | dev | IRRG | 25 foreground prompts | disabled | 8171 | 0.1284 | 0.0629 | 0.0548 | 0.1098 | 0.5159 | 0.1879 |

### 10.2 Exp3-A / Exp3-B 类别 IoU 对比表

| Class | RGB IoU | IRRG IoU | Change |
|---|---:|---:|---:|
| impervious surface | 0.2032 | 0.1738 | -0.0295 |
| building | 0.0460 | 0.0037 | -0.0423 |
| low vegetation | 0.1270 | 0.0001 | -0.1269 |
| tree | 0.0004 | 0.0019 | +0.0015 |
| car | 0.1970 | 0.1514 | -0.0456 |
| clutter/background | 0.0455 | 0.0462 | +0.0007 |

### 10.3 Exp3-B dev 类别指标表

| Class | GT % | Pred % | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 28.34 | 35.57 | 0.1738 | 0.2961 | 0.2660 | 0.3339 |
| building | 27.83 | 0.12 | 0.0037 | 0.0074 | 0.8422 | 0.0037 |
| low vegetation | 23.20 | 0.04 | 0.0001 | 0.0002 | 0.0636 | 0.0001 |
| tree | 14.49 | 0.03 | 0.0019 | 0.0038 | 0.8956 | 0.0019 |
| car | 1.36 | 0.21 | 0.1514 | 0.2630 | 0.9808 | 0.1518 |
| clutter/background | 4.78 | 64.03 | 0.0462 | 0.0884 | 0.0475 | 0.6360 |

## 11. 后续分析建议

后续写 Exp3-C / Exp3-D 时，应避免把 IRRG 单独输入直接解释为“近红外无效”。更准确的结论是：

```text
IRRG 作为替代 RGB 的假彩色三通道输入，在当前 SAM3 零样本开放词汇设置下无效甚至负收益。
```

后续更值得验证的是：

1. RGBIR 派生三通道是否能在保留自然 RGB 对齐的同时引入 NIR 信息。
2. 多视图融合是否能让 RGB 负责 building/car 等自然外观类别，让 IRRG 或 RGBIR 补充 vegetation/tree。
3. 是否需要 class-aware modality routing，而不是全类别统一替换输入模态。
