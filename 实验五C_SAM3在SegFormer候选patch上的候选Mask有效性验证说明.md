# 实验五C：SAM3 在 SegFormer candidate patches 上的候选 mask 有效性验证说明

## 1. 实验目的

Exp5-C 的目标不是直接证明融合方法已经有效，而是先验证一个更基础的问题：

```text
当 SegFormer 已经给出疑难区域后，SAM3 在这些局部区域内生成的候选 mask，
是否能够覆盖 SegFormer 的错误像素、clutter/background 漏检像素，或边界不稳定区域？
```

也就是说，本实验把 SAM3 从“整图六类密集分类器”调整为“局部候选 mask 生成器”。如果 SAM3 的候选 mask 在这些区域内确实能够较好覆盖错误区域，那么后续才有必要继续设计 SegFormer + SAM3 的 refinement 或 fusion 规则。

## 2. 实验输入

本实验依赖前两个阶段的输出：

1. SegFormer heldout full-tile 评估及不确定性输出：

```text
results_segformer_potsdam_rgb/eval_heldout_uncertainty
```

其中包含：

```text
predictions/*_prediction_id.png
predictions/*_ground_truth_id.png
uncertainty/*_entropy.png
uncertainty/*_margin.png
uncertainty/*_error.png
```

2. Exp5-B 生成的候选区域：

```text
results_exp5b_segformer_candidate_regions_heldout
```

其中核心文件为：

```text
masks/*_candidate.png
```

候选区域来自：

```text
high entropy top 20%
OR low margin top 20%
OR selected-class boundary band
```

## 3. 实验方法

Exp5-C 的流程为：

1. 读取 heldout 图像、SegFormer 预测、GT 标签和 Exp5-B candidate mask。
2. 按 `1008 x 1008` patch、`672` stride 生成候选 patch。
3. 只保留 candidate 像素占比不低于 `0.08` 的 patch。
4. 每张图最多选 `6` 个 patch。
5. 全 heldout 再按 `candidate_ratio` 排序，最多保留 `60` 个 patch。
6. 对每个候选 patch 运行 Exp2-B 的 SAM3 prompt ensemble。
7. 将 SAM3 输出 mask 限制在 candidate region 内评价。

当前使用的 prompt class 为：

```text
0 impervious surface
1 building
2 low vegetation
3 tree
4 car
```

仍然不主动 prompt `clutter/background`，因为前面 Exp2/Exp3 已经表明直接让 SAM3 承担背景类别的密集分类并不稳定。这里更关心 SAM3 生成的前景候选 mask 是否能帮助解释或修正 SegFormer 在局部区域的错误。

## 4. 运行命令

配置文件为：

```text
exp5c_sam3_candidate_patch_validation.yaml
```

正式运行：

```bash
python exp5c_sam3_candidate_patch_validation.py \
  --config exp5c_sam3_candidate_patch_validation.yaml
```

如果只想检查候选 patch，不运行 SAM3：

```bash
python exp5c_sam3_candidate_patch_validation.py \
  --config exp5c_sam3_candidate_patch_validation.yaml \
  --dry-run \
  --output-dir results_exp5c_sam3_candidate_patch_validation_dryrun_full
```

## 5. 当前 dry-run 检查结果

已完成一次默认 60 个 patch 的 dry-run：

```text
输出目录：
results_exp5c_sam3_candidate_patch_validation_dryrun_full
```

候选 patch 概况：

```text
selected patches: 60
covered heldout images: 13 / 14
aggregate candidate ratio: 0.4817
aggregate SegFormer error ratio: 0.1894
aggregate clutter FN ratio: 0.0555
candidate_ratio range: 0.4375 - 0.5930
error_ratio range: 0.0999 - 0.4214
clutter_fn_ratio range: 0.0000 - 0.3802
```

这说明当前选出的 patch 不是随机区域，而是候选区域密度较高、同时包含较多 SegFormer 错误的局部区域，适合用于验证 SAM3 的局部 mask 有效性。

## 6. 输出文件

正式运行后，默认输出目录为：

```text
results_exp5c_sam3_candidate_patch_validation_heldout
```

核心输出包括：

```text
metrics/candidate_patch_proposals.csv
metrics/sam_patch_metrics.csv
metrics/sam_mask_metrics.csv
metrics/sam_candidate_validation_summary.json
patch_visualizations/*.png
```

其中：

```text
candidate_patch_proposals.csv
```

记录被选中的候选 patch 位置、candidate 比例、SegFormer 错误比例和 clutter FN 比例。

```text
sam_patch_metrics.csv
```

记录每个 patch 上 SAM3 mask union 对错误区域、clutter FN 区域和 candidate 区域的覆盖情况。

```text
sam_mask_metrics.csv
```

记录每一个 SAM3 mask 的类别、score、面积、错误区域重叠、clutter FN 重叠和 GT 类别精度。

```text
sam_candidate_validation_summary.json
```

记录全实验聚合指标。

## 7. 重点观察指标

最重要的指标不是常规 mIoU，而是：

```text
sam_error_coverage
sam_clutter_fn_coverage
sam_error_precision
sam_clutter_fn_precision
sam_candidate_overlap_ratio
```

解释如下：

```text
sam_error_coverage:
SAM3 mask union 覆盖了多少 SegFormer 错误像素。

sam_clutter_fn_coverage:
SAM3 mask union 覆盖了多少 clutter/background 漏检像素。

sam_error_precision:
SAM3 mask union 中有多少比例确实落在 SegFormer 错误区域。

sam_clutter_fn_precision:
SAM3 mask union 中有多少比例确实落在 clutter/background 漏检区域。

sam_candidate_overlap_ratio:
candidate region 中有多少比例被 SAM3 mask 覆盖。
```

如果 `sam_error_coverage` 较高但 `sam_error_precision` 很低，说明 SAM3 mask 很大，容易“扫到”错误区域，但不够精确。

如果 `sam_error_precision` 较高但 `sam_error_coverage` 很低，说明 SAM3 只捕捉到少量错误区域，不足以支撑大范围 refinement。

比较理想的结果是：

```text
SAM3 对错误区域有可观覆盖；
同时 mask 不只是大面积泛化覆盖；
在建筑、道路、车辆、植被边界附近能形成较清晰的实例或对象级候选。
```

## 8. 对后续实验的作用

Exp5-C 的结果会决定后续融合路线：

```text
如果 SAM3 mask 对错误区域覆盖有效：
继续做 Exp5-D，设计 SegFormer + SAM3 局部修正规则。

如果 SAM3 mask 覆盖错误区域有限：
说明 SAM3 更适合作为可视化解释或少量对象边界辅助，不适合作为主要 refinement 模块。

如果 SAM3 主要覆盖正确前景而不覆盖错误区域：
说明 candidate patch 选择需要调整，例如从 candidate_ratio 排序改为 error-likelihood、clutter-risk 或边界类别组合排序。
```

因此，Exp5-C 是 SegFormer + SAM3 融合前的必要验证实验。
