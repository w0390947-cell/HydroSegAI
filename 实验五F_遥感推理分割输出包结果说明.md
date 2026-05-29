# 实验五F：遥感推理分割输出包结果说明

## 1. 实验名称

```text
Exp5-F: reasoning segmentation package
```

中文名称：

```text
遥感推理分割输出包
```

## 2. 运行命令

```bash
python exp5f_reasoning_segmentation_package.py \
  --config exp5f_reasoning_segmentation_package.yaml
```

## 3. 配置文件

```text
exp5f_reasoning_segmentation_package.yaml
```

关键配置：

```yaml
package:
  final_prediction_source: "segformer"
  include_ground_truth_metrics: true

review_priority:
  candidate_weight: 0.35
  entropy_weight: 0.30
  low_margin_weight: 0.20
  boundary_weight: 0.15
  high_priority_threshold: 0.60
  medium_priority_threshold: 0.35
```

说明：

```text
final prediction 不由 SAM3 自动覆盖；
SAM3 proposal 审计结果用于解释和复核建议；
GT 只用于评价 review priority 覆盖错误的能力。
```

## 4. 输入目录

```text
results_segformer_potsdam_rgb/eval_heldout_uncertainty
results_exp5b_segformer_candidate_regions_heldout
results_exp5e_sam3_proposal_quality_audit_heldout
```

## 5. 输出目录

```text
results_exp5f_reasoning_segmentation_package_heldout
```

输出结构：

```text
final_predictions/
review_priority/
candidate_components/
overlays/
manifests/
metrics/
```

## 6. 输出文件说明

```text
final_predictions/*_final_id.png
```

最终类别 ID 图，当前等于 SegFormer coarse prediction。

```text
final_predictions/*_final.png
```

彩色最终语义图。

```text
review_priority/*_priority.png
```

连续复核优先级图。

```text
review_priority/*_high_priority.png
```

高优先级复核区域。

```text
review_priority/*_medium_priority.png
```

中优先级复核区域，当前对应 candidate region 级别。

```text
candidate_components/
```

保存 candidate、high entropy、low margin、boundary band 等组成部分。

```text
overlays/*_priority_overlay.png
```

复核优先级叠加可视化。

```text
manifests/*_manifest.json
```

每张图的解释说明和指标。

```text
manifests/package_manifest.json
```

整个输出包的总 manifest。

## 7. 总体指标

```text
num_images: 14
valid pixels: 467,862,855
candidate pixels: 145,556,699
high priority pixels: 52,163,786
medium priority pixels: 145,556,699
error pixels: 45,819,895
high priority error pixels: 22,769,011
medium priority error pixels: 36,869,626
```

比例：

```text
candidate ratio: 0.3111
high priority ratio: 0.1115
medium priority ratio: 0.3111
high priority error coverage: 0.4969
medium priority error coverage: 0.8047
```

## 8. 结果解释

高优先级区域：

```text
占有效像素 11.15%，覆盖错误像素 49.69%。
```

中优先级 / candidate 区域：

```text
占有效像素 31.11%，覆盖错误像素 80.47%。
```

这说明：

```text
review priority layer 可以有效浓缩错误风险区域；
candidate region 可以作为更完整的复核搜索空间；
final prediction 保持稳定 coarse semantic result。
```

## 9. SAM3 proposal 建议

Package manifest 中保留了 Exp5-E 的 proposal audit 结论：

```text
impervious surface:
  do_not_auto_refine

building:
  candidate_object_layer

low vegetation:
  needs_additional_prior

car:
  candidate_object_layer

object_proposal_high_precision:
  use_as_review_or_interactive_proposals
```

## 10. 当前结论

Exp5-F 形成了一个阶段性的推理分割系统输出：

```text
stable final semantic prediction
+ review priority map
+ candidate region components
+ SAM3 proposal recommendations
+ per-image explanation manifest
```

它比自动 refinement 更稳健，也更符合后续迁移到其他遥感或水利数据集的目标。
