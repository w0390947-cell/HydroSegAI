# 实验五B：SegFormer + Potsdam SAM3 局部细化候选区域结果分析

## 1. 实验目的

Exp5-A 表明：

```text
SegFormer uncertainty 能覆盖大部分普通错误，
但 heldout 的 clutter/background 欠召回包含一部分高置信错分，
不能只靠 entropy 或 confidence 阈值定位。
```

因此，Exp5-B 先不直接调用 SAM3，而是构建一个可解释的局部 refinement 候选区域：

```text
candidate_region =
    high_entropy_top20
    OR low_margin_top20
    OR boundary_band(predicted classes in {impervious, building, low vegetation, clutter})
```

本实验要回答：

```text
如果只在 candidate_region 内考虑后续 SAM3 局部细化，
它能覆盖多少 SegFormer 错误像素？
它能覆盖多少 clutter/background false negative？
候选区域是否过大？
```

## 2. 运行命令

候选区域生成脚本为：

```text
exp5b_segformer_candidate_regions.py
```

运行命令：

```bash
python exp5b_segformer_candidate_regions.py \
  --eval-dir results_segformer_potsdam_rgb/eval_heldout_uncertainty \
  --output-dir results_exp5b_segformer_candidate_regions_heldout \
  --entropy-fraction 0.20 \
  --margin-fraction 0.20 \
  --boundary-radius 8 \
  --boundary-classes 0,1,2,5
```

其中：

| 参数 | 含义 |
|---|---|
| `entropy-fraction=0.20` | 每张图中 entropy 最高的 20% 有效像素 |
| `margin-fraction=0.20` | 每张图中 margin 最低的 20% 有效像素 |
| `boundary-radius=8` | 预测类别边界向外膨胀 8 像素 |
| `boundary-classes=0,1,2,5` | 只关注 impervious、building、low vegetation、clutter 的预测边界 |

输出目录：

```text
results_exp5b_segformer_candidate_regions_heldout/
```

## 3. 输出文件

每张 heldout 图输出：

| 输出 | 说明 |
|---|---|
| `masks/*_candidate.png` | 三类候选条件合并后的最终候选区域 |
| `masks/*_high_entropy.png` | high entropy top20 区域 |
| `masks/*_low_margin.png` | low margin top20 区域 |
| `masks/*_boundary_band.png` | 指定类别边界带 |
| `overlays/*_candidate_overlay.png` | 预测语义图上叠加红色 candidate region |
| `metrics/candidate_region_summary.json` | 数据集级统计 |
| `metrics/candidate_region_per_image.csv` | 逐图统计 |

本次输出总大小约：

```text
70 MB
```

## 4. 数据集级结果

候选区域总体统计如下：

| 指标 | 数值 |
|---|---:|
| 有效像素总数 | 467,862,855 |
| candidate 像素数 | 145,556,699 |
| candidate 占有效像素比例 | 31.11% |
| 总错误像素数 | 45,819,895 |
| candidate 覆盖错误像素数 | 36,869,626 |
| 总错误覆盖率 | 80.47% |
| clutter false negative 总数 | 11,353,310 |
| candidate 覆盖 clutter false negative 数 | 7,477,482 |
| clutter false negative 覆盖率 | 65.86% |
| candidate 内错误像素比例 | 25.33% |
| candidate 内 clutter false negative 比例 | 5.14% |

总体判断：

```text
第一版 candidate region 用 31.11% 的有效像素区域覆盖了 80.47% 的总错误，
以及 65.86% 的 clutter/background false negative。
```

这说明该候选区域足以作为 SAM3 局部细化的第一版搜索空间，但候选区域仍偏大。如果直接在全部 candidate 内密集调用 SAM3，成本可能仍然较高，需要进一步切 patch 或连通域筛选。

## 5. 不同组件的贡献

各候选组件单独覆盖率如下：

| 组件 | 区域占比 | error 覆盖率 | clutter FN 覆盖率 | clutter->impervious 覆盖率 | clutter->low vegetation 覆盖率 | clutter->building 覆盖率 | 区域内错误比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| high entropy top20 | 20.00% | 70.70% | 55.07% | 45.10% | 64.99% | 62.29% | 34.62% |
| low margin top20 | 20.00% | 70.62% | 54.07% | 44.56% | 64.34% | 59.55% | 34.58% |
| boundary band | 21.12% | 52.93% | 47.44% | 40.97% | 51.78% | 75.75% | 24.54% |
| merged candidate | 31.11% | 80.47% | 65.86% | 54.48% | 73.10% | 84.78% | 25.33% |

解释：

1. `high entropy top20` 和 `low margin top20` 行为非常接近，二者都能覆盖约 70.7% 总错误。
2. `boundary band` 单独覆盖总错误较少，但对 `clutter -> building` 很有效，覆盖率达到 75.75%。
3. 合并后 candidate 区域从 20% 扩大到 31.11%，总错误覆盖率从约 70.7% 提升到 80.47%。
4. 对 `clutter -> impervious surface` 的覆盖率仍只有 54.48%，说明这类高置信错分仍然是最难处理的部分。

## 6. 逐类 clutter false negative 覆盖

Candidate 对三类主要 clutter 错分方向的覆盖如下：

| 错分方向 | 总像素数 | candidate 覆盖像素数 | 覆盖率 |
|---|---:|---:|---:|
| clutter -> impervious surface | 5,952,210 | 3,242,617 | 54.48% |
| clutter -> low vegetation | 2,890,824 | 2,113,104 | 73.10% |
| clutter -> building | 1,915,168 | 1,623,586 | 84.78% |

这说明：

```text
candidate region 对 clutter->building 和 clutter->low vegetation 的定位较好；
对 clutter->impervious surface 的定位仍然不足。
```

这与 Exp5-A 的不确定性分析一致：

```text
clutter->impervious surface 往往是相对高置信错分，
不一定会落在 high entropy / low margin 区域。
```

## 7. 逐图结果

逐图 candidate 覆盖率如下：

| 图像 | candidate 占比 | error 覆盖率 | clutter FN 覆盖率 | candidate 内错误比例 |
|---|---:|---:|---:|---:|
| top_potsdam_2_13_RGB | 32.27% | 89.37% | 93.64% | 25.09% |
| top_potsdam_2_14_RGB | 29.62% | 88.29% | 92.30% | 26.48% |
| top_potsdam_3_13_RGB | 32.89% | 88.64% | 87.02% | 24.62% |
| top_potsdam_3_14_RGB | 32.06% | 87.82% | 85.24% | 26.31% |
| top_potsdam_4_13_RGB | 31.15% | 79.71% | 71.01% | 26.70% |
| top_potsdam_4_14_RGB | 33.37% | 59.02% | 26.94% | 28.34% |
| top_potsdam_4_15_RGB | 33.00% | 85.93% | 80.51% | 23.65% |
| top_potsdam_5_13_RGB | 30.41% | 87.93% | 93.46% | 24.95% |
| top_potsdam_5_14_RGB | 32.62% | 79.89% | 84.36% | 27.66% |
| top_potsdam_5_15_RGB | 32.33% | 74.32% | 59.21% | 26.43% |
| top_potsdam_6_13_RGB | 28.08% | 85.54% | 84.31% | 19.90% |
| top_potsdam_6_14_RGB | 29.16% | 86.63% | 82.83% | 19.62% |
| top_potsdam_6_15_RGB | 32.07% | 84.93% | 73.97% | 23.00% |
| top_potsdam_7_13_RGB | 26.81% | 72.59% | 65.38% | 31.68% |

最值得注意的是 `top_potsdam_4_14_RGB`：

```text
candidate 占比 = 33.37%
error 覆盖率 = 59.02%
clutter FN 覆盖率 = 26.94%
```

该图在 Exp5-A 中已经表现为高置信 clutter 错分。Exp5-B 再次确认：这类错误不容易被 uncertainty + 边界带捕获。

## 8. 对 SAM3 局部细化的意义

第一版 candidate region 的价值在于：

```text
它把 SAM3 可能需要处理的区域从整图 100% 缩小到约 31.11%，
同时保留了约 80.47% 的错误像素。
```

这已经比全图 SAM3 prompt ensemble 更合理。

但它也暴露出两个限制：

1. 候选区域仍偏大，需要进一步转成 patch/连通域级任务。
2. `clutter -> impervious surface` 高置信错分覆盖不足，不能只依赖 uncertainty 和普通边界。

因此，下一步不建议直接对所有 candidate 像素调用 SAM3，而应先生成候选 patch：

```text
candidate mask -> connected components / patch proposals -> SAM3 local prompt
```

推荐策略：

1. 将 candidate mask 切成局部 patch，例如 512×512 或 768×768。
2. 只保留 candidate 像素占比超过阈值的 patch。
3. 对高优先级混淆方向设置不同 prompt：
   - `clutter/background`
   - `road and pavement` / `impervious surface`
   - `low vegetation`
   - `building`
4. SAM3 只在候选 patch 内生成 mask，再通过规则决定是否覆盖 SegFormer 预测。

## 9. 当前结论

一句话总结：

```text
Exp5-B 的 candidate region 能以 31.11% 的区域覆盖 80.47% 总错误，
适合作为 SAM3 局部细化的第一版候选池；
但 clutter->impervious surface 的高置信错分仍然覆盖不足，
下一步需要从 candidate region 进一步生成局部 patch/proposal，
并设计类别受限的 SAM3 修正规则。
```

