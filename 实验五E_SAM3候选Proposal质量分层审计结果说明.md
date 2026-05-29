# 实验五E：SAM3 候选 proposal 质量分层审计结果说明

## 1. 实验名称

```text
Exp5-E: SAM3 proposal quality audit
```

中文名称：

```text
SAM3 候选 proposal 质量分层审计
```

## 2. 实验目的

本实验用于回答：

```text
SAM3 candidate masks 更适合作为自动 refinement，
还是更适合作为对象候选、错误解释和交互式复核 proposal？
```

该实验不重新运行 SAM3，而是复用 Exp5-C 与 Exp5-D 的 mask 级结果进行审计。

## 3. 运行命令

```bash
python exp5e_sam3_proposal_quality_audit.py
```

## 4. 输入文件

```text
results_exp5c_sam3_candidate_patch_validation_heldout/metrics/sam_mask_metrics.csv
results_exp5d_sam3_conservative_refinement_heldout/metrics/refinement_mask_metrics.csv
```

## 5. 输出目录

```text
results_exp5e_sam3_proposal_quality_audit_heldout
```

输出文件：

```text
metrics/proposal_quality_summary.json
metrics/proposal_mask_merged.csv
metrics/proposal_quality_by_class.csv
metrics/proposal_quality_by_class_score.csv
metrics/proposal_quality_by_class_area.csv
metrics/proposal_quality_by_tier.csv
metrics/proposal_quality_by_reject_reason.csv
metrics/proposal_recommendations.csv
```

## 6. 总体结果

```text
mask_count: 465
accepted_mask_count in Exp5-D: 41
eval_area: 8,235,378
weighted GT precision: 0.1562
weighted error precision: 0.3282
weighted clutter FN precision: 0.0751
```

## 7. 按类别结果

```text
impervious surface:
  masks: 30
  eval_area: 7,707,872
  GT precision: 0.1285
  error precision: 0.3312

building:
  masks: 13
  eval_area: 91,208
  GT precision: 0.8140
  error precision: 0.0906

low vegetation:
  masks: 19
  eval_area: 395,766
  GT precision: 0.4691
  error precision: 0.3398

car:
  masks: 403
  eval_area: 40,532
  GT precision: 0.8862
  error precision: 0.1868
```

## 8. 质量分层结果

```text
object_proposal_high_precision:
  masks: 230
  eval_area: 75,607
  GT precision: 0.9928

object_proposal_medium_precision:
  masks: 24
  eval_area: 434,740
  GT precision: 0.6707

error_region_explanation:
  masks: 44
  eval_area: 3,034,341
  error precision: 0.4126
  clutter FN precision: 0.1810

large_surface_risky:
  masks: 14
  eval_area: 4,524,730
  GT precision: 0.1508

low_confidence_or_ambiguous:
  masks: 108
  eval_area: 165,960

empty_or_unusable:
  masks: 45
```

## 9. 推荐解释

Exp5-E 生成的推荐结论：

```text
impervious surface:
  do_not_auto_refine
  大面积地表 mask 语义精度低，只适合作为风险/解释层。

building:
  candidate_object_layer
  精度较高，适合作为候选对象或复核 proposal。

low vegetation:
  needs_additional_prior
  需要额外先验后才能进入自动流程。

car:
  candidate_object_layer
  精度较高，适合作为候选对象或复核 proposal。

object_proposal_high_precision:
  use_as_review_or_interactive_proposals
  可作为交互式或人工复核候选，不宜直接自动覆盖。
```

## 10. 结果结论

本实验说明：

```text
SAM3 proposals 中存在高精度对象候选，
但最可靠的使用方式不是自动写入语义图，
而是作为 object proposal layer、risk explanation layer 和 interactive refinement layer。
```

这为后续论文框架提供了更稳妥的方向：

```text
coarse semantic prediction
+ uncertainty / risk map
+ SAM3 object proposals
+ optional human / prior-guided refinement
```

而不是：

```text
coarse semantic prediction + SAM3 automatic overwrite
```
