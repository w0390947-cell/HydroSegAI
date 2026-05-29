# 实验三C：SAM3 + Potsdam RGBIR 派生输入模态 dev 结果分析

## 1. 本次运行概况

本次实验已完成 `exp3c_sam3_potsdam_rgbir_composite.py` 的 development split 运行，结果保存于：

```text
results_exp3c_sam3_potsdam_rgbir_composite_dev/
```

Exp3-C 的实验变量是：

```text
在固定 Exp2-B 视觉 prompt ensemble 策略后，使用 Potsdam RGBIR 四波段影像派生出的三通道 composite 作为输入。
```

本次配置中的 composite 方式为：

```text
rgbir_composite_mode = nir-r-g
```

也就是从 RGBIR 四波段影像中取：

```text
NIR -> 第 1 通道
R   -> 第 2 通道
G   -> 第 3 通道
```

因此，Exp3-C 可以理解为一种从 RGBIR 四波段数据派生的近红外假彩色输入实验。它与 Exp3-B IRRG 在通道语义上非常接近，但数据来源是 `4_Ortho_RGBIR` 四波段影像，而不是 `3_Ortho_IRRG` 三波段影像。

本次运行使用的数据与配置如下：

| 项目 | 内容 |
|---|---|
| 输入影像 | `Potsdam/4_Ortho_RGBIR/4_Ortho_RGBIR` |
| 输入模态 | `rgbir_composite` |
| composite mode | `nir-r-g` |
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
| 总有效候选 mask 数 | 7836 |
| mask 面积过滤 | 关闭 |
| fallback 类别 | `clutter/background` |
| 评估方式 | noBoundary 标签，黑色 `(0,0,0)` 映射为 `ignore_index=255` |

输出文件数量检查：

| 输出类型 | 数量 | 说明 |
|---|---:|---|
| `results_exp3c_sam3_potsdam_rgbir_composite_dev/predictions` | 96 | 24 张图，每张包含彩色预测、彩色 GT、预测 ID、GT ID |
| `results_exp3c_sam3_potsdam_rgbir_composite_dev/visualizations` | 24 | 每张图一张对比可视化 |
| `results_exp3c_sam3_potsdam_rgbir_composite_dev/metrics` | 29 | 24 个单图 JSON + `overall_metrics.json`、`per_image_metrics.csv`、`experiment_metadata.json`、`config_snapshot.yaml`、`run_context.json` |
| `results_exp3c_sam3_potsdam_rgbir_composite_dev/logs` | 1 | 本次运行日志 |

## 2. 固定 Prompt 与模态设置

Exp3-C 固定继承 Exp2-B / Exp3-A / Exp3-B 的视觉 prompt ensemble：

| 类别 | Prompt variants |
|---|---|
| impervious surface | `impervious surface`, `road and pavement`, `asphalt road`, `paved area`, `parking lot` |
| building | `building`, `buildings`, `rooftop`, `building roof`, `house roof` |
| low vegetation | `low vegetation`, `grass`, `lawn`, `low plants`, `ground vegetation` |
| tree | `tree`, `trees`, `tree crown`, `tall tree`, `urban trees` |
| car | `car`, `cars`, `vehicle`, `small vehicle`, `parked car` |

`clutter/background` 不作为主动文本查询类别，只作为评估类别和未覆盖像素的 fallback 类别。Exp3-C 同样不启用 Exp2-C 的 class-aware mask area filtering。

模态设置如下：

| 字段 | 内容 |
|---|---|
| `input_modality.name` | `rgbir_composite` |
| `input_modality.rgbir_composite_mode` | `nir-r-g` |
| `input_modality.description` | `Three-channel NIR-R-G composite derived from Potsdam RGBIR.` |
| 图像文件模板 | `{image_id}_RGBIR.tif` |
| 原始波段 | R、G、B、NIR |
| 实际输入 SAM3 的三通道 | NIR、R、G |
| DSM / nDSM | 未使用 |
| 多视图融合 | 未使用 |

## 3. 总体结果

本次实验的推荐论文主结果应使用 `dataset_*` 指标，即先累计 24 张 dev 图像的混淆矩阵，再计算总体指标。

| 指标 | 数值 |
|---|---:|
| Overall Accuracy | 0.1296 |
| Mean IoU | 0.0632 |
| Mean F1 | 0.1104 |
| Mean Precision | 0.5732 |
| Mean Recall | 0.1909 |
| Frequency Weighted IoU | 0.0575 |

逐图平均指标如下，仅建议作为补充参考：

| 指标 | 数值 |
|---|---:|
| Average Overall Accuracy | 0.1290 |
| Average Mean IoU | 0.0637 |
| Average Mean F1 | 0.1066 |
| Average Mean Precision | 0.2796 |
| Average Mean Recall | 0.1969 |
| Average Frequency Weighted IoU | 0.0638 |

总体判断：

```text
Exp3-C 的 RGBIR-derived NIR-R-G composite 结果与 Exp3-B IRRG 非常接近，略高于 IRRG，但仍显著低于 Exp3-A RGB。该输入方式没有恢复 RGB 与 SAM3 自然图像视觉-文本表征之间的对齐优势，也没有带来有效的植被类别增益。
```

## 4. 与 Exp3-A / Exp3-B 的对比

Exp3-A、Exp3-B、Exp3-C 使用同一 dev split、同一 prompt ensemble、同一推理参数和同一 fallback 规则，因此可以作为输入模态敏感性实验直接比较。

| 实验 | Input modality | Valid masks | OA | mIoU | Mean F1 | Mean Precision | Mean Recall | FWIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Exp3-A | RGB | 12402 | 0.1677 | 0.1032 | 0.1781 | 0.6506 | 0.2284 | 0.1048 |
| Exp3-B | IRRG | 8171 | 0.1284 | 0.0629 | 0.1098 | 0.5159 | 0.1879 | 0.0548 |
| Exp3-C | RGBIR composite `nir-r-g` | 7836 | 0.1296 | 0.0632 | 0.1104 | 0.5732 | 0.1909 | 0.0575 |

相对 RGB，Exp3-C 的变化如下：

| 指标 | RGB | RGBIR composite | 变化 |
|---|---:|---:|---:|
| OA | 0.1677 | 0.1296 | -0.0382 |
| mIoU | 0.1032 | 0.0632 | -0.0400 |
| Mean F1 | 0.1781 | 0.1104 | -0.0676 |
| Mean Precision | 0.6506 | 0.5732 | -0.0774 |
| Mean Recall | 0.2284 | 0.1909 | -0.0374 |
| FWIoU | 0.1048 | 0.0575 | -0.0473 |

相对 IRRG，Exp3-C 只有极小提升：

| 指标 | IRRG | RGBIR composite | 变化 |
|---|---:|---:|---:|
| OA | 0.1284 | 0.1296 | +0.0011 |
| mIoU | 0.0629 | 0.0632 | +0.0004 |
| Mean F1 | 0.1098 | 0.1104 | +0.0006 |
| Mean Precision | 0.5159 | 0.5732 | +0.0573 |
| Mean Recall | 0.1879 | 0.1909 | +0.0030 |
| FWIoU | 0.0548 | 0.0575 | +0.0027 |

这说明 `nir-r-g` composite 与 IRRG 的行为高度相似，不能视为相对 RGB 的有效改进。

## 5. 各类别结果

| 类别 | GT 占比 | 预测占比 | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 28.34% | 32.82% | 0.1823 | 0.3084 | 0.2873 | 0.3327 |
| building | 27.83% | 0.12% | 0.0041 | 0.0083 | 0.9567 | 0.0041 |
| low vegetation | 23.20% | 0.06% | 0.0007 | 0.0014 | 0.2859 | 0.0007 |
| tree | 14.49% | 0.04% | 0.0026 | 0.0051 | 0.8784 | 0.0026 |
| car | 1.36% | 0.20% | 0.1435 | 0.2511 | 0.9837 | 0.1439 |
| clutter/background | 4.78% | 66.76% | 0.0463 | 0.0884 | 0.0474 | 0.6616 |

类别 IoU 与 RGB / IRRG 对比如下：

| 类别 | RGB IoU | IRRG IoU | RGBIR composite IoU | 相对 RGB | 相对 IRRG |
|---|---:|---:|---:|---:|---:|
| impervious surface | 0.2032 | 0.1738 | 0.1823 | -0.0209 | +0.0085 |
| building | 0.0460 | 0.0037 | 0.0041 | -0.0419 | +0.0004 |
| low vegetation | 0.1270 | 0.0001 | 0.0007 | -0.1263 | +0.0006 |
| tree | 0.0004 | 0.0019 | 0.0026 | +0.0021 | +0.0006 |
| car | 0.1970 | 0.1514 | 0.1435 | -0.0534 | -0.0078 |
| clutter/background | 0.0455 | 0.0462 | 0.0463 | +0.0007 | +0.0000 |

类别级结果说明：

1. `impervious surface` 相对 IRRG 略有改善，但仍低于 RGB；预测占比 32.82%，高于 GT 占比 28.34%，存在吸收其他类别的倾向。
2. `building` 与 IRRG 一样几乎无法召回，IoU 仅 0.0041。虽然 precision 高，但预测占比只有 0.12%，说明覆盖严重不足。
3. `low vegetation` 仍未从 NIR 信息中受益，IoU 仅 0.0007，远低于 RGB 的 0.1270。
4. `tree` 是唯一相对 RGB 略有提升的类别，但 IoU 只有 0.0026，绝对值仍然太低，不能认为形成了有效树冠识别。
5. `car` 低于 RGB 和 IRRG，说明小目标在该 composite 下没有受益。
6. `clutter/background` 预测占比为 66.76%，低于 RGB 的 69.75%，但这仍远高于真实占比 4.78%，且下降并不代表前景类别被正确召回。

## 6. 混淆矩阵解读

Exp3-C 的主要错误模式与 Exp3-B 一致：大量前景像素仍被预测为 `clutter/background`，同时大量植被和背景像素被误吸收到 `impervious surface`。

| 真实类别 | 误分为 clutter/background 的比例 | 误分为 impervious surface 的比例 |
|---|---:|---:|
| impervious surface | 66.71% | 33.27% |
| building | 90.96% | 8.61% |
| low vegetation | 44.87% | 55.05% |
| tree | 54.60% | 44.88% |
| car | 77.93% | 7.67% |
| clutter/background | 66.16% | 33.80% |

对 `low vegetation` 而言，问题尤为明显：

```text
真实 low vegetation 中只有 0.07% 被正确召回；
44.87% 回退到 clutter/background；
55.05% 被误分为 impervious surface。
```

这说明 NIR-R-G composite 并没有让 SAM3 把植被 prompt 与植被区域对齐，反而让大量植被区域被道路/硬化地表相关 prompt 吸收。

因此，Exp3-C 与 Exp3-B 支持同一个结论：

```text
近红外相关三通道输入作为 SAM3 单输入模态时，并不能直接带来植被类开放词汇分割收益。
```

更准确地说，这不是“近红外无用”，而是：

```text
当前 SAM3 的开放词汇视觉-文本表征更适配自然 RGB 外观；
NIR-R-G / IRRG 假彩色外观与 grass、lawn、tree crown、building roof 等文本概念的对齐较弱。
```

## 7. 单图结果分析

按 `mean_iou` 排序，本次实验表现最好的 5 张图为：

| 排名 | 图像 | Mean IoU | OA |
|---:|---|---:|---:|
| 1 | `top_potsdam_7_10_RGBIR` | 0.1382 | 0.3505 |
| 2 | `top_potsdam_3_11_RGBIR` | 0.1110 | 0.1599 |
| 3 | `top_potsdam_7_9_RGBIR` | 0.1020 | 0.2624 |
| 4 | `top_potsdam_7_12_RGBIR` | 0.0964 | 0.2563 |
| 5 | `top_potsdam_2_11_RGBIR` | 0.0930 | 0.1593 |

表现最差的 5 张图为：

| 排名 | 图像 | Mean IoU | OA |
|---:|---|---:|---:|
| 1 | `top_potsdam_7_8_RGBIR` | 0.0230 | 0.0291 |
| 2 | `top_potsdam_4_10_RGBIR` | 0.0268 | 0.0674 |
| 3 | `top_potsdam_7_7_RGBIR` | 0.0300 | 0.0738 |
| 4 | `top_potsdam_3_12_RGBIR` | 0.0373 | 0.0565 |
| 5 | `top_potsdam_6_7_RGBIR` | 0.0413 | 0.0668 |

最好的单图 `top_potsdam_7_10_RGBIR` 的 Mean IoU 为 0.1382，高于 IRRG 对应图像的 0.1360，但仍明显低于 RGB 对应图像的 0.1897。这进一步说明，RGBIR-derived NIR-R-G composite 并未超过 RGB 参照。

## 8. 结果意义

Exp3-C 的主要意义是验证了一个重要边界：

```text
即使从 RGBIR 四波段影像中显式引入 NIR 通道，并构造 NIR-R-G 三通道输入，也不能在当前 SAM3 零样本开放词汇框架下超过 RGB 输入。
```

这对后续实验路线有两个启示：

1. 如果继续做 Exp3-D 多视图融合，应谨慎期待 IRRG / RGBIR composite 带来整体增益，因为单模态结果没有显示明显类别互补性。
2. 如果要利用近红外信息，更可能需要 class-aware prior、NDVI 规则、DSM/nDSM 高度约束，或监督模型学习，而不是直接把假彩色图像交给 SAM3 文本 prompt。

因此，Exp3-C 更适合作为“直接多光谱派生输入并不充分”的负向证据，为后续 SegFormer 监督基线、DSM/NDVI 先验或更结构化的融合方法提供动机。

## 9. 论文中如何表述这个结果

建议将 Exp3-C 定位为：

```text
RGBIR-derived NIR-R-G composite 输入模态消融实验。
```

英文可写为：

```text
Exp3-C converts the four-band RGBIR orthophoto into a three-channel NIR-R-G composite and evaluates it with the same visual prompt ensemble and fallback strategy as Exp3-A/B. On the development split, the RGBIR-derived composite obtains a dataset-level mIoU of 0.0632 and a mean F1 of 0.1104, which are close to IRRG but substantially lower than RGB. In particular, low vegetation remains almost unrecognized, with an IoU of only 0.0007, and many vegetation pixels are confused with impervious surface. These results indicate that directly feeding NIR-based false-color composites into SAM3 does not improve open-vocabulary dense segmentation under the current setting.
```

中文可写为：

```text
Exp3-C 将 Potsdam RGBIR 四波段影像转换为 NIR-R-G 三通道 composite，并在与 Exp3-A/B 相同的视觉 prompt ensemble 和 fallback 策略下进行评估。在 development split 上，RGBIR-derived composite 的数据集级 mIoU 为 0.0632，Mean F1 为 0.1104，结果接近 IRRG，但明显低于 RGB。尤其是 low vegetation 的 IoU 仅为 0.0007，大量植被像素被误分为 impervious surface。该结果说明，在当前设置下，直接将近红外相关假彩色 composite 输入 SAM3 并不能改善开放词汇密集语义分割。
```

## 10. 推荐论文表格

### 10.1 Exp3 dev 总体指标表

| Method | Split | Input modality | Composite | Valid masks | OA | mIoU | FWIoU | Mean F1 | Mean Precision | Mean Recall |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Exp3-A | dev | RGB | - | 12402 | 0.1677 | 0.1032 | 0.1048 | 0.1781 | 0.6506 | 0.2284 |
| Exp3-B | dev | IRRG | - | 8171 | 0.1284 | 0.0629 | 0.0548 | 0.1098 | 0.5159 | 0.1879 |
| Exp3-C | dev | RGBIR-derived | NIR-R-G | 7836 | 0.1296 | 0.0632 | 0.0575 | 0.1104 | 0.5732 | 0.1909 |

### 10.2 Exp3-A/B/C 类别 IoU 对比表

| Class | RGB | IRRG | RGBIR NIR-R-G |
|---|---:|---:|---:|
| impervious surface | 0.2032 | 0.1738 | 0.1823 |
| building | 0.0460 | 0.0037 | 0.0041 |
| low vegetation | 0.1270 | 0.0001 | 0.0007 |
| tree | 0.0004 | 0.0019 | 0.0026 |
| car | 0.1970 | 0.1514 | 0.1435 |
| clutter/background | 0.0455 | 0.0462 | 0.0463 |

### 10.3 Exp3-C dev 类别指标表

| Class | GT % | Pred % | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 28.34 | 32.82 | 0.1823 | 0.3084 | 0.2873 | 0.3327 |
| building | 27.83 | 0.12 | 0.0041 | 0.0083 | 0.9567 | 0.0041 |
| low vegetation | 23.20 | 0.06 | 0.0007 | 0.0014 | 0.2859 | 0.0007 |
| tree | 14.49 | 0.04 | 0.0026 | 0.0051 | 0.8784 | 0.0026 |
| car | 1.36 | 0.20 | 0.1435 | 0.2511 | 0.9837 | 0.1439 |
| clutter/background | 4.78 | 66.76 | 0.0463 | 0.0884 | 0.0474 | 0.6616 |

## 11. 后续分析建议

基于 Exp3-A/B/C dev，目前不建议把最终方案改成 IRRG 或 RGBIR composite 单输入。更合理的下一步是：

1. 将 RGB 保持为 SAM3 的主输入模态。
2. 如果继续做 Exp3-D，多视图融合应被定位为探索性实验，而不是预期强增益方法。
3. 将近红外信息更多用于 NDVI / class-aware prior，而不是直接替换 RGB 外观。
4. 并行推进 SegFormer 监督基线，让模型通过标签学习多源遥感特征，而不是依赖 SAM3 零样本文本对齐。
