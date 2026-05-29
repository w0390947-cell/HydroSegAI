# 实验五D：SAM3 候选 mask 保守细化验证说明

## 1. 实验定位

Exp5-D 的目标不是追求 Potsdam heldout 分数的最大化，而是验证一个可迁移的推理分割问题：

```text
在 coarse semantic segmenter 已经给出 dense prediction，
并由不确定性 / 边界 / 低置信区域生成 candidate region 后，
开放词汇 mask proposer 生成的局部 masks 是否可以经过保守筛选，
用于有限、可解释、低风险的 refinement？
```

这里：

```text
SegFormer = coarse semantic segmenter
SAM3 = open-vocabulary mask proposer
candidate region = refinement search space
conservative rules = mask acceptance / rejection gates
```

Potsdam 只用于评价这套流程是否有效，不应成为规则被写死的对象。

## 2. 与 Exp5-C 的区别

Exp5-C 回答的是：

```text
SAM3 在 candidate patches 上生成的 masks 是否覆盖错误区域？
```

Exp5-D 进一步回答：

```text
如果只接受较可信、较小、位于不确定候选区域内的 SAM3 masks，
是否能实际改善 coarse segmentation？
```

因此 Exp5-D 会生成 refined prediction，并与原始 SegFormer prediction 对比。

## 3. 泛化约束

Exp5-D 遵守以下原则：

```text
GT 标签只用于评价，不参与 refinement 决策；
所有筛选规则均写入 yaml 配置；
不使用 Potsdam 图块编号、固定错误位置或 heldout 标签信息；
不把 SAM3 直接当成最终语义分类器；
不采用“SAM3 覆盖哪里就改哪里”的激进策略。
```

这使得该实验更接近一种可迁移框架验证，而不是 Potsdam 专属后处理。

## 4. 当前保守规则

配置文件：

```text
exp5d_sam3_conservative_refinement.yaml
```

当前规则：

```yaml
refinement:
  accepted_class_ids: [1, 4]
  score_threshold: 0.55
  min_mask_area: 16
  max_mask_area_fraction: 0.08
  require_candidate_region: true
  require_uncertainty_gate: true
  apply_only_where_prediction_differs: true
  mask_priority: "score"

  uncertainty_gate:
    entropy_min: 0.15
    margin_max: 0.80
    confidence_max: 0.98
    mode: "any"
```

解释：

```text
accepted_class_ids:
  当前先只允许较适合作为对象候选的类别进入 refinement。
  这不是写死规则，后续可以替换为其他遥感数据集的目标类别。

score_threshold:
  过滤低置信 SAM3 masks。

min_mask_area / max_mask_area_fraction:
  过滤极小噪声 mask 和超大泛化 mask。

require_candidate_region:
  只在 SegFormer candidate region 内改动。

require_uncertainty_gate:
  只在 SegFormer 低置信、高熵或低 margin 区域内改动。

apply_only_where_prediction_differs:
  如果 SegFormer 已经预测为同一类别，则不重复改动。
```

## 5. 运行命令

正式运行：

```bash
python exp5d_sam3_conservative_refinement.py \
  --config exp5d_sam3_conservative_refinement.yaml
```

只检查候选 patch，不运行 SAM3：

```bash
python exp5d_sam3_conservative_refinement.py \
  --config exp5d_sam3_conservative_refinement.yaml \
  --dry-run \
  --output-dir results_exp5d_sam3_conservative_refinement_dryrun
```

## 6. 输出目录

默认输出：

```text
results_exp5d_sam3_conservative_refinement_heldout
```

核心文件：

```text
metrics/refinement_summary.json
metrics/refinement_patch_metrics.csv
metrics/refinement_mask_metrics.csv
metrics/refinement_per_image_metrics.csv
predictions/*_refined_id.png
predictions/*_refined.png
change_maps/*_change_map.png
patch_visualizations/*.png
```

## 7. 关键评价指标

最重要的不是只看 mIoU 是否上升，而是同时看：

```text
delta_mean_iou
delta_overall_accuracy
changed_pixels
improved_pixels
degraded_pixels
change_precision
degradation_rate
accepted_mask_count
```

解释：

```text
delta_mean_iou:
  refinement 对整体语义分割指标的影响。

changed_pixels:
  规则实际改动了多少像素。

improved_pixels:
  原来错、改后对的像素数量。

degraded_pixels:
  原来对、改后错的像素数量。

change_precision:
  被改动像素中真正改对的比例。

degradation_rate:
  被改动像素中被改坏的比例。
```

如果 mIoU 略升但 degraded_pixels 很高，说明规则不够安全。

如果 changed_pixels 很少但 change_precision 很高，说明规则保守但可靠，可以考虑逐步放宽。

如果 changed_pixels 很多但 mIoU 下降，说明 SAM3 mask 不能直接用于 refinement，需要更强筛选或只作为解释输出。

## 8. 预期解读

Exp5-D 的理想结果不是一次性大幅提升，而是找到一种可信的融合边界：

```text
哪些 SAM3 masks 可以安全进入最终结果？
哪些 SAM3 masks 只能作为候选解释？
哪些类别或区域不应由 SAM3 修正？
```

如果当前保守规则有效，后续可以做 Exp5-E：

```text
多策略 refinement ablation：
  object-only
  boundary-only
  uncertainty-only
  class-specific gates
  background residual inference
```

如果当前保守规则无效，结论也有价值：

```text
SAM3 更适合作为候选解释和人工交互提示模块，
而不是自动 semantic refinement 模块。
```

这仍然服务于论文主线：构建一套可解释、可迁移、可验证的遥感推理分割技术体系。
