# 实验五D：SAM3 候选 mask 保守细化验证结果说明

## 1. 实验名称

```text
Exp5-D: conservative SAM3 refinement
```

中文名称：

```text
SAM3 候选 mask 保守细化验证
```

## 2. 运行命令

```bash
python exp5d_sam3_conservative_refinement.py \
  --config exp5d_sam3_conservative_refinement.yaml
```

## 3. 输出目录

```text
results_exp5d_sam3_conservative_refinement_heldout
```

主要输出：

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

## 4. 实验设置

本次 refinement 规则：

```yaml
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

说明：

```text
GT 标签只用于评价；
refinement 决策只依赖 SegFormer prediction、uncertainty maps、candidate region 和 SAM3 masks；
没有使用 Potsdam 的标签信息进行规则选择。
```

## 5. mask 筛选结果

SAM3 输出：

```text
total masks: 465
accepted masks: 41
accepted ratio: 8.82%
```

拒绝原因：

```text
area_too_small: 228
score_below_threshold: 147
class_not_accepted: 49
accepted: 41
```

类别分布：

```text
all masks:
  class 0 impervious surface: 30
  class 1 building: 13
  class 2 low vegetation: 19
  class 4 car: 403

accepted masks:
  class 1 building: 4
  class 4 car: 37
```

## 6. refinement 改动结果

```text
changed pixels: 6,420
improved pixels: 810
degraded pixels: 4,928
still wrong changed pixels: 682
change precision: 0.1262
degradation rate: 0.7676
```

解释：

```text
只有 12.62% 的改动是真正修正；
76.76% 的改动是误伤；
10.62% 的改动是从一种错误变成另一种错误。
```

## 7. processed image 口径指标

当前输出 summary 中 processed image 为 13 张图，因为 1 张 heldout 图没有入选 candidate patch。

```text
baseline OA: 0.901566
refined OA: 0.901557
delta OA: -0.000009

baseline mIoU: 0.768540
refined mIoU: 0.768535
delta mIoU: -0.000005
```

## 8. 完整 heldout 口径指标

将未入选 candidate patch 的图保持原预测不变，并计入完整 14 张 heldout 后：

```text
baseline OA: 0.902066
refined OA: 0.902057
delta OA: -0.000009

baseline mIoU: 0.768670
refined mIoU: 0.768666
delta mIoU: -0.000005
```

完整 heldout 各类 IoU：

```text
impervious surface:
  0.857596 -> 0.857571

building:
  0.924765 -> 0.924727

low vegetation:
  0.776710 -> 0.776713

tree:
  0.786032 -> 0.786033

car:
  0.903402 -> 0.903438

clutter/background:
  0.363517 -> 0.363512
```

## 9. 图像级结果

发生实际改动的图像：

```text
top_potsdam_5_13:
  changed: 96
  improved: 96
  degraded: 0

top_potsdam_4_14:
  changed: 651
  improved: 405
  degraded: 222

top_potsdam_5_14:
  changed: 67
  improved: 19
  degraded: 48

top_potsdam_6_15:
  changed: 5,606
  improved: 290
  degraded: 4,658
```

`top_potsdam_6_15` 是主要负贡献来源。

## 10. 结果判断

本次实验结果不支持直接自动 refinement：

```text
即使经过较保守筛选，SAM3 masks 自动写入 refined prediction 后仍带来净负收益。
```

因此，当前阶段应将 SAM3 定位为：

```text
candidate object layer
explanation layer
interactive refinement proposal
```

而不是：

```text
automatic semantic correction layer
```

## 11. 脚本修正说明

本次结果分析后，已对 `exp5d_sam3_conservative_refinement.py` 做了两个统计口径修正：

```text
1. 后续运行会将没有 candidate patch 的 heldout 图也纳入最终指标；
2. patch-level changed pixels 在重叠 patch 情况下只统计当前 patch 新引入的改动。
```

这两个修正不改变本次核心结论，但会让后续复现实验的统计更严格。

## 12. 下一步

建议后续转向：

```text
Exp5-E: SAM3 candidate proposal 质量分层与解释层构建
```

或引入更强的通用先验后再尝试 fusion：

```text
coarse model probability consistency
boundary consistency
object shape prior
multi-source remote-sensing prior
water-conservancy spatial prior
```

不建议继续只通过调节 Potsdam 上的 `score_threshold`、`area_threshold` 来追求小幅指标提升。
