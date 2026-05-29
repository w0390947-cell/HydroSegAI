# 实验五A：SegFormer Potsdam 不确定性与错误区域分析

## 1. 实验目的

本实验基于 Exp4 的 SegFormer RGB supervised baseline，进一步分析：

```text
SegFormer 的错误区域是否集中在低置信、高 entropy、低 margin 区域？
clutter/background 的 heldout 欠召回是否可以通过不确定性图定位？
```

本实验使用的评估命令为：

```bash
python eval_segformer_potsdam.py \
  --config segformer_potsdam_rgb.yaml \
  --split heldout \
  --save-uncertainty \
  --output-dir results_segformer_potsdam_rgb/eval_heldout_uncertainty
```

输出目录为：

```text
results_segformer_potsdam_rgb/eval_heldout_uncertainty/
```

该目录在常规 full-tile evaluation 输出基础上，额外保存：

```text
uncertainty/*_confidence.png
uncertainty/*_entropy.png
uncertainty/*_margin.png
uncertainty/*_error.png
```

## 2. 输出完整性

本次 uncertainty evaluation 成功处理 14 张 heldout 图像，指标与上一轮 heldout full-tile evaluation 完全一致：

| 指标 | 数值 |
|---|---:|
| Overall Accuracy | 0.9021 |
| Mean IoU | 0.7687 |
| Mean F1 | 0.8535 |
| Frequency Weighted IoU | 0.8251 |

逐类 IoU：

| 类别 | IoU |
|---|---:|
| impervious surface | 0.8576 |
| building | 0.9248 |
| low vegetation | 0.7767 |
| tree | 0.7860 |
| car | 0.9034 |
| clutter/background | 0.3635 |

输出文件检查：

| 输出类型 | 数量 | 说明 |
|---|---:|---|
| `predictions` | 56 | 14 张图，每张包含预测、GT、ID 图 |
| `visualizations` | 14 | 每张图一张 comparison |
| `metrics` | 18 | 14 个单图 JSON + overall/per-image/run context/config |
| `uncertainty` | 56 | 每张图 4 张 uncertainty/error 图 |

## 3. 不确定性指标定义

本实验从滑窗融合后的整图 logits 计算不确定性：

| 图 | 含义 | 越大表示 |
|---|---|---|
| `confidence` | top-1 softmax probability | 模型越自信 |
| `entropy` | 归一化 softmax entropy | 模型越不确定 |
| `margin` | top-1 probability - top-2 probability | top-1 与 top-2 越分离，模型越确定 |
| `error` | prediction 是否等于 ground truth | 白色为错误，黑色为正确，灰色为 ignore |

注意：

```text
confidence 与 margin 越低，越不确定；
entropy 越高，越不确定。
```

## 4. 正确像素与错误像素对比

Heldout 全部有效像素中，正确像素与错误像素的不确定性统计如下：

| 像素组 | 像素数 | confidence mean | entropy mean | margin mean |
|---|---:|---:|---:|---:|
| correct | 422,042,960 | 0.9592 | 0.0590 | 0.9275 |
| error | 45,819,895 | 0.7708 | 0.2951 | 0.5863 |

分位数对比如下：

| 像素组 | confidence P25/P50/P75 | entropy P25/P50/P75 | margin P25/P50/P75 |
|---|---|---|---|
| correct | 0.9843 / 0.9961 / 0.9961 | 0.0000 / 0.0039 / 0.0431 | 0.9725 / 0.9961 / 0.9961 |
| error | 0.6078 / 0.7882 / 0.9451 | 0.1255 / 0.3216 / 0.4196 | 0.2784 / 0.6118 / 0.9059 |

结论：

```text
整体上，错误像素确实显著低置信、高 entropy、低 margin。
因此，SegFormer uncertainty 可以作为错误区域定位的重要线索。
```

但是，错误像素的 confidence P75 仍达到 0.9451，说明并非所有错误都是低置信错误。存在一部分高置信错分。

## 5. Clutter/background 相关错误的不确定性

Heldout 中 `clutter/background` 主要问题是召回不足。这里重点比较真实 clutter 的正确预测与错误预测：

| 像素组 | 像素数 | confidence mean | entropy mean | margin mean |
|---|---:|---:|---:|---:|
| clutter true positive | 8,567,058 | 0.8835 | 0.1678 | 0.7949 |
| clutter false negative | 11,353,310 | 0.8269 | 0.2417 | 0.7027 |

真实 clutter 被错分后的主要方向：

| 错分方向 | 像素数 | confidence mean | entropy mean | margin mean |
|---|---:|---:|---:|---:|
| clutter -> impervious surface | 5,952,210 | 0.8600 | 0.1951 | 0.7582 |
| clutter -> low vegetation | 2,890,824 | 0.7968 | 0.2771 | 0.6435 |
| clutter -> building | 1,915,168 | 0.8028 | 0.2883 | 0.6750 |

结论：

```text
clutter false negative 的不确定性高于 clutter true positive，
但差距没有普通 correct/error 对比那么大。
尤其 clutter -> impervious surface 的错分 confidence mean = 0.8600，
entropy mean = 0.1951，属于相对高置信错分。
```

这意味着：

```text
仅靠 entropy 或 confidence 阈值，不能完整捕获 clutter/background 欠召回问题。
```

## 6. Top 不确定区域的错误覆盖率

为了评估不确定性图是否能作为 SAM3 refinement 的候选区域，本实验统计了：

```text
选取每张图中最高 entropy / 最低 confidence / 最低 margin 的前 X% 有效像素，
这些区域能覆盖多少错误像素和 clutter false negative 像素。
```

结果如下：

| 选区比例 | high entropy 覆盖 error | low confidence 覆盖 error | low margin 覆盖 error | high entropy 覆盖 clutter FN | low confidence 覆盖 clutter FN | low margin 覆盖 clutter FN |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | 25.47% | 26.30% | 25.93% | 24.23% | 19.33% | 17.50% |
| 10% | 45.73% | 45.91% | 45.71% | 35.95% | 33.41% | 32.20% |
| 20% | 70.70% | 70.70% | 70.62% | 55.07% | 54.54% | 54.07% |
| 30% | 83.46% | 83.42% | 83.41% | 67.35% | 67.07% | 66.84% |

这一结果很有价值：

1. 对总体错误而言，前 20% 高不确定区域可以覆盖约 70.7% 的错误像素。
2. 对 clutter false negative 而言，前 20% 高不确定区域只能覆盖约 55.1%。
3. 前 30% 高不确定区域可以覆盖约 67.4% 的 clutter false negative，但选区已经较大。

因此：

```text
uncertainty threshold 适合做第一层候选区域筛选，
但不足以单独解决 clutter/background 欠召回。
```

## 7. 逐图不确定性观察

按 `clutter false negative entropy mean` 从高到低排序：

| 图像 | error rate | correct confidence | error confidence | correct entropy | error entropy | clutter FN confidence | clutter FN entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| top_potsdam_2_14_RGB | 0.0889 | 0.9551 | 0.7542 | 0.0632 | 0.2933 | 0.6767 | 0.4550 |
| top_potsdam_2_13_RGB | 0.0906 | 0.9532 | 0.7289 | 0.0692 | 0.3446 | 0.7225 | 0.3949 |
| top_potsdam_5_13_RGB | 0.0863 | 0.9571 | 0.7316 | 0.0619 | 0.3477 | 0.7285 | 0.3826 |
| top_potsdam_3_13_RGB | 0.0913 | 0.9547 | 0.7398 | 0.0667 | 0.3376 | 0.7496 | 0.3569 |
| top_potsdam_3_14_RGB | 0.0960 | 0.9526 | 0.7425 | 0.0688 | 0.3229 | 0.7611 | 0.3318 |
| top_potsdam_5_14_RGB | 0.1129 | 0.9567 | 0.7780 | 0.0629 | 0.2915 | 0.7876 | 0.3038 |
| top_potsdam_4_15_RGB | 0.0908 | 0.9609 | 0.7601 | 0.0567 | 0.3150 | 0.7939 | 0.3016 |
| top_potsdam_6_13_RGB | 0.0653 | 0.9675 | 0.7595 | 0.0470 | 0.3092 | 0.7952 | 0.2909 |
| top_potsdam_6_14_RGB | 0.0661 | 0.9712 | 0.7689 | 0.0412 | 0.2961 | 0.8006 | 0.2797 |
| top_potsdam_6_15_RGB | 0.0868 | 0.9617 | 0.7694 | 0.0554 | 0.2939 | 0.7937 | 0.2714 |
| top_potsdam_4_13_RGB | 0.1043 | 0.9647 | 0.7718 | 0.0520 | 0.3018 | 0.8135 | 0.2597 |
| top_potsdam_7_13_RGB | 0.1170 | 0.9619 | 0.8018 | 0.0534 | 0.2587 | 0.8416 | 0.2044 |
| top_potsdam_5_15_RGB | 0.1150 | 0.9622 | 0.7984 | 0.0543 | 0.2639 | 0.8556 | 0.2020 |
| top_potsdam_4_14_RGB | 0.1602 | 0.9483 | 0.8230 | 0.0757 | 0.2299 | 0.9244 | 0.1053 |

这里有一个重要现象：

```text
top_potsdam_4_14_RGB 的 error rate 最高，为 16.02%，
但 clutter FN entropy mean 只有 0.1053，confidence mean 高达 0.9244。
```

这说明某些 tile 上的 clutter 错分是模型非常自信的系统性错误，而不是简单的“不确定错误”。这类错误很难只靠 entropy threshold 捕获。

## 8. 对 SAM3 refinement 的启示

本实验支持使用 uncertainty 做 refinement 候选区域，但不能只靠 uncertainty。

更合理的策略是：

```text
uncertainty gating + class-pair gating + boundary/local region gating
```

具体来说：

1. 使用 high entropy / low confidence / low margin 找到大部分普通错误区域。
2. 对 `clutter -> impervious surface` 这类高置信错分，需要额外引入类别对规则。
3. 对 `tree <-> low vegetation`，可结合类别边界和局部纹理区域。
4. SAM3 不应全图运行，而应只在候选区域或候选区域周边 patch 中运行。

建议下一步 Exp5-B 的候选区域定义为：

```text
candidate_region =
    high_entropy_top20
    OR low_margin_top20
    OR boundary_band(predicted classes in {impervious, low vegetation, building, clutter})
    OR predicted impervious/low vegetation/building regions with local texture/shape anomaly
```

其中前两项由 uncertainty 提供，第三项由 SegFormer 预测边界提供，第四项可以先做简单启发式，后续再考虑 SAM3 mask 质量。

## 9. 当前结论

一句话总结：

```text
SegFormer uncertainty 能有效定位大部分普通错误，
但 heldout 的 clutter/background 欠召回包含大量高置信错分；
因此后续 SAM3 refinement 不能只按不确定性阈值触发，
必须结合类别混淆方向和边界局部区域。
```

