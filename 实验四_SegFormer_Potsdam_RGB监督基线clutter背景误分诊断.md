# 实验四：SegFormer RGB 监督基线 clutter/background 误分诊断

## 1. 诊断目的

在 SegFormer RGB supervised baseline 的 full-tile 评估中，validation 与 heldout 的总体结果如下：

| Split | OA | mIoU | Mean F1 | FWIoU |
|---|---:|---:|---:|---:|
| validation | 0.8932 | 0.7856 | 0.8762 | 0.8130 |
| heldout | 0.9021 | 0.7687 | 0.8535 | 0.8251 |

heldout 的 OA 和 FWIoU 高于 validation，但 mIoU 和 Mean F1 略低。逐类对比显示，下降主要来自 `clutter/background`：

| 类别 | Val IoU | Heldout IoU | 变化 |
|---|---:|---:|---:|
| impervious surface | 0.8557 | 0.8576 | +0.0018 |
| building | 0.8962 | 0.9248 | +0.0286 |
| low vegetation | 0.6729 | 0.7767 | +0.1038 |
| tree | 0.7481 | 0.7860 | +0.0379 |
| car | 0.8936 | 0.9034 | +0.0098 |
| clutter/background | 0.6470 | 0.3635 | -0.2834 |

因此，本诊断聚焦于：

```text
heldout 中真实 clutter/background 主要被 SegFormer 错分成哪些类别？
这些错误是否集中在少数 tile？
这对后续 SAM3 refinement 有什么启示？
```

## 2. Heldout clutter/background 总体误分方向

在 heldout 上，真实 `clutter/background` 像素总数为：

```text
19,920,368
```

这些像素的预测分布如下：

| 真实类别 | 预测类别 | 像素数 | 占真实 clutter/background 比例 |
|---|---|---:|---:|
| clutter/background | clutter/background | 8,567,058 | 43.01% |
| clutter/background | impervious surface | 5,952,210 | 29.88% |
| clutter/background | low vegetation | 2,890,824 | 14.51% |
| clutter/background | building | 1,915,168 | 9.61% |
| clutter/background | tree | 382,991 | 1.92% |
| clutter/background | car | 212,117 | 1.06% |

结论：

```text
heldout 中 clutter/background 的主要问题是召回不足。
真实 clutter/background 中只有 43.01% 被正确预测，
其余主要被吸收到 impervious surface、low vegetation 和 building。
```

其中最主要的误分方向是：

```text
clutter/background -> impervious surface
clutter/background -> low vegetation
clutter/background -> building
```

这说明模型并不是把 clutter/background 随机分散到所有类别，而是倾向于把复杂背景解释为更常见、更稳定的地物类别。

## 3. 与 validation 的对比

Validation 中真实 `clutter/background` 的预测分布为：

| 真实类别 | 预测类别 | 占真实 clutter/background 比例 |
|---|---|---:|
| clutter/background | clutter/background | 78.56% |
| clutter/background | impervious surface | 9.53% |
| clutter/background | building | 5.88% |
| clutter/background | low vegetation | 4.15% |
| clutter/background | tree | 1.60% |
| clutter/background | car | 0.28% |

对比可见：

| 误分方向 | Validation | Heldout | 变化 |
|---|---:|---:|---:|
| clutter/background -> clutter/background | 78.56% | 43.01% | -35.55% |
| clutter/background -> impervious surface | 9.53% | 29.88% | +20.35% |
| clutter/background -> low vegetation | 4.15% | 14.51% | +10.36% |
| clutter/background -> building | 5.88% | 9.61% | +3.73% |

这说明 validation 与 heldout 中 `clutter/background` 的外观分布存在明显差异。Heldout 中的杂乱/背景区域更容易被模型解释为硬化地表、低矮植被或建筑。

## 4. Heldout 逐图 clutter/background 表现

按 `clutter/background` recall 从低到高排列：

| 图像 | GT clutter 像素数 | 误分为 impervious | 误分为 building | 误分为 low vegetation | 误分为 tree | 误分为 car | clutter recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| top_potsdam_4_13_RGB | 2,016,369 | 54.10% | 9.19% | 16.29% | 7.49% | 1.07% | 11.86% |
| top_potsdam_4_15_RGB | 796,094 | 14.19% | 23.42% | 34.21% | 1.78% | 0.48% | 25.92% |
| top_potsdam_6_13_RGB | 616,456 | 56.93% | 2.55% | 9.58% | 2.66% | 0.62% | 27.65% |
| top_potsdam_6_14_RGB | 700,587 | 27.22% | 21.12% | 19.50% | 1.42% | 2.15% | 28.59% |
| top_potsdam_4_14_RGB | 3,353,621 | 45.50% | 4.28% | 18.76% | 0.88% | 0.31% | 30.26% |
| top_potsdam_3_13_RGB | 606,188 | 32.35% | 13.43% | 18.91% | 1.52% | 0.74% | 33.05% |
| top_potsdam_2_14_RGB | 86,364 | 12.68% | 30.45% | 14.41% | 6.39% | 2.48% | 33.60% |
| top_potsdam_5_15_RGB | 2,177,487 | 41.67% | 11.73% | 9.18% | 1.23% | 1.10% | 35.09% |
| top_potsdam_5_13_RGB | 849,116 | 22.63% | 22.69% | 9.93% | 0.97% | 4.93% | 38.85% |
| top_potsdam_3_14_RGB | 845,108 | 21.86% | 18.74% | 14.04% | 2.19% | 0.69% | 42.49% |
| top_potsdam_5_14_RGB | 2,264,710 | 22.99% | 14.37% | 10.88% | 2.95% | 0.96% | 47.86% |
| top_potsdam_2_13_RGB | 473,669 | 10.94% | 21.31% | 9.27% | 2.96% | 4.55% | 50.96% |
| top_potsdam_7_13_RGB | 3,153,689 | 14.78% | 0.21% | 12.36% | 0.28% | 0.52% | 71.86% |
| top_potsdam_6_15_RGB | 1,980,910 | 7.61% | 4.51% | 12.89% | 0.20% | 0.98% | 73.81% |

最严重的 tile 是：

```text
top_potsdam_4_13_RGB
top_potsdam_4_15_RGB
top_potsdam_6_13_RGB
top_potsdam_6_14_RGB
top_potsdam_4_14_RGB
```

这些图像中 `clutter/background` recall 均低于 31%。其中 `top_potsdam_4_13_RGB` 的 clutter recall 只有 11.86%，并且 54.10% 的真实 clutter 被预测为 `impervious surface`。

## 5. 其他类别的主要混淆方向

Heldout 数据集级混淆矩阵显示，各真实类别的前三个非本类误分方向如下：

| 真实类别 | 主要误分方向 |
|---|---|
| impervious surface | low vegetation 2.68%，tree 1.23%，building 1.06% |
| building | impervious surface 2.59%，clutter/background 0.70%，low vegetation 0.53% |
| low vegetation | tree 5.08%，impervious surface 3.01%，clutter/background 0.95% |
| tree | low vegetation 10.44%，impervious surface 2.68%，building 0.42% |
| car | clutter/background 1.46%，impervious surface 1.43%，tree 1.16% |
| clutter/background | impervious surface 29.88%，low vegetation 14.51%，building 9.61% |

除 `clutter/background` 外，最值得注意的是：

```text
tree -> low vegetation: 10.44%
low vegetation -> tree: 5.08%
```

这是植被内部的正常混淆，属于遥感语义分割中的常见问题。相比之下，`clutter/background -> impervious surface` 的 29.88% 更异常，也更值得后续针对。

## 6. 预测类别纯度

从预测类别角度看，各预测类中真实来源占比如下：

| 预测类别 | 最大真实来源 | 次要污染来源 |
|---|---|---|
| impervious surface | impervious surface 90.75% | clutter/background 3.90%，building 1.98%，low vegetation 1.90% |
| building | building 96.25% | clutter/background 1.65%，impervious surface 1.35% |
| low vegetation | low vegetation 84.64% | tree 8.10%，impervious surface 3.84%，clutter/background 2.81% |
| tree | tree 90.22% | low vegetation 6.43%，impervious surface 2.38% |
| car | car 94.33% | clutter/background 2.81%，tree 1.60% |
| clutter/background | clutter/background 70.14% | impervious surface 12.21%，low vegetation 7.53%，building 6.72% |

这说明：

1. `impervious surface` 的预测整体仍然很干净，90.75% 来自真实不透水表面。
2. 但由于 `impervious surface` 是大面积强势类，即使只有 3.90% 的预测污染来自 clutter/background，也对应大量 clutter 像素被吞并。
3. `clutter/background` 的预测纯度为 70.14%，并不算极差；真正的问题是召回不足，而不是预测出来的 clutter 大量错误。

因此，后续应优先提高 `clutter/background` recall，而不是简单抬高 precision。

## 7. 对 SAM3 refinement 的启示

本诊断不支持“让 SAM3 重新做全图六类分割”。更合理的 refinement 方向是：

```text
SegFormer 保持为主分割器；
SAM3 只参与 SegFormer 容易混淆的局部区域；
重点关注 clutter/background 与 impervious surface / low vegetation / building 的边界和低置信区域。
```

优先级最高的 refinement 目标：

1. `clutter/background -> impervious surface`
2. `clutter/background -> low vegetation`
3. `clutter/background -> building`
4. `tree <-> low vegetation`

建议的下一步实验不是直接融合 SAM3 mask，而是先导出 SegFormer 的不确定性图：

```text
softmax confidence
entropy
top-1 / top-2 margin
class boundary map
```

然后检查：

1. 被错分的 clutter/background 是否集中在低置信区域。
2. 被错分为 impervious surface 的 clutter 是否靠近类别边界。
3. SAM3 在这些局部区域能否生成更合理的对象/区域 mask。

如果上述条件成立，再设计：

```text
SegFormer coarse prediction + uncertainty-guided SAM3 local refinement
```

而不是：

```text
SAM3 full-image prompt ensemble replacement
```

## 8. 建议的后续实验设计

可以按三个阶段推进：

### 8.1 Exp5-A：SegFormer uncertainty 导出与错误区域分析

目标：

```text
在 val / heldout 上导出 confidence、entropy、margin，
并统计错误像素与低置信区域的重合程度。
```

预期产物：

```text
confidence map
entropy map
margin map
error map
class boundary map
per-class uncertainty statistics
```

### 8.2 Exp5-B：低置信区域 SAM3 局部重分割

目标：

```text
只在 SegFormer 低置信区域或类别边界附近调用 SAM3，
避免全图高成本 prompt ensemble。
```

重点关注：

```text
clutter/background 与 impervious surface / low vegetation / building 的边界区域。
```

### 8.3 Exp5-C：规则化融合

目标：

```text
设计简单、可解释的融合规则，
只在 SegFormer 低置信且 SAM3 mask 质量较高时允许修正。
```

可能规则：

1. 高置信 SegFormer 区域不修改。
2. 低置信区域允许 SAM3 候选 mask 覆盖。
3. 只允许修正指定混淆对。
4. 对小目标 car 保持 SegFormer 预测，不交给 SAM3 大范围改写。

一句话总结：

```text
Heldout 结果表明 SegFormer 已经足够稳定，后续 SAM3 的价值不在替代 SegFormer，而在有选择地处理 SegFormer 对 clutter/background 和类别边界不确定的局部区域。
```

