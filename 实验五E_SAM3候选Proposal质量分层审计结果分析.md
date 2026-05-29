# 实验五E：SAM3 候选 proposal 质量分层审计结果分析

## 1. 实验目的

Exp5-D 表明，即使采用保守规则，直接把 SAM3 masks 写入最终语义图仍然存在较高误伤风险。

因此 Exp5-E 不再继续做自动 refinement，而是转向一个更符合论文主线的问题：

```text
SAM3 生成的 candidate masks 到底适合承担什么角色？
```

具体来说，本实验对 Exp5-C 和 Exp5-D 的 mask 级指标进行合并审计，将 SAM3 proposal 分为：

```text
高精度对象候选；
中等精度对象候选；
错误区域解释候选；
大面积地表风险候选；
低置信或歧义候选；
不可用候选。
```

这一步的目标不是提升 Potsdam 分数，而是为最终“遥感推理分割技术体系”明确模块分工。

## 2. 输入与输出

输入：

```text
results_exp5c_sam3_candidate_patch_validation_heldout/metrics/sam_mask_metrics.csv
results_exp5d_sam3_conservative_refinement_heldout/metrics/refinement_mask_metrics.csv
```

运行脚本：

```bash
python exp5e_sam3_proposal_quality_audit.py
```

输出目录：

```text
results_exp5e_sam3_proposal_quality_audit_heldout
```

主要输出：

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

## 3. 总体结果

Exp5-E 共审计：

```text
mask_count: 465
accepted_mask_count in Exp5-D: 41
eval_area: 8,235,378
weighted GT precision: 0.1562
weighted error precision: 0.3282
weighted clutter FN precision: 0.0751
```

这说明：

```text
从面积加权角度看，SAM3 proposals 的语义类别精度并不高；
但它们与 SegFormer 错误区域的重叠更明显。
```

换句话说：

```text
SAM3 masks 更像“疑难/错误区域解释信号”，
而不是可以直接覆盖 coarse prediction 的“语义标签信号”。
```

这与 Exp5-D 的负结果一致。

## 4. 按类别审计

按类别统计：

```text
class 0 impervious surface:
  mask_count: 30
  eval_area: 7,707,872
  GT precision: 0.1285
  error precision: 0.3312
  clutter FN precision: 0.0794

class 1 building:
  mask_count: 13
  eval_area: 91,208
  GT precision: 0.8140
  error precision: 0.0906
  clutter FN precision: 0.0351

class 2 low vegetation:
  mask_count: 19
  eval_area: 395,766
  GT precision: 0.4691
  error precision: 0.3398
  clutter FN precision: 0.0078

class 4 car:
  mask_count: 403
  eval_area: 40,532
  GT precision: 0.8862
  error precision: 0.1868
  clutter FN precision: 0.0000
```

这里出现了非常清晰的分工：

```text
building / car:
  语义精度较高，适合作为 candidate object layer 或人工复核 proposal。

impervious surface:
  面积巨大、语义精度低，但与错误区域重叠较高，适合作为 risk/explanation layer。

low vegetation:
  有一定错误区域解释能力，但类别精度不足，需要更强先验才能进入自动 refinement。
```

## 5. 按 proposal 质量层级审计

质量分层结果如下：

```text
object_proposal_high_precision:
  mask_count: 230
  eval_area: 75,607
  GT precision: 0.9928
  error precision: 0.0849
  accepted_mask_count: 28

object_proposal_medium_precision:
  mask_count: 24
  eval_area: 434,740
  GT precision: 0.6707
  error precision: 0.2536
  accepted_mask_count: 7

error_region_explanation:
  mask_count: 44
  eval_area: 3,034,341
  GT precision: 0.0722
  error precision: 0.4126
  clutter FN precision: 0.1810

large_surface_risky:
  mask_count: 14
  eval_area: 4,524,730
  GT precision: 0.1508
  error precision: 0.2887

low_confidence_or_ambiguous:
  mask_count: 108
  eval_area: 165,960
  GT precision: 0.1098
  error precision: 0.1703

empty_or_unusable:
  mask_count: 45
```

最重要的发现是：

```text
确实存在高精度对象 proposal。
```

`object_proposal_high_precision` 层级中：

```text
230 个 masks
GT precision = 0.9928
```

这说明 SAM3 对某些小目标或对象级区域有很强候选能力。

但与此同时：

```text
这些高精度 proposal 的总面积只有 75,607 像素，
主要适合作为对象候选层，而不是大范围语义图修正层。
```

## 6. 为什么 Exp5-D 自动 refinement 失败

Exp5-E 能解释 Exp5-D 的失败原因。

Exp5-D 试图将一部分 SAM3 masks 写入最终语义图，但 Exp5-E 显示：

```text
高精度 proposal 面积小，适合对象候选；
大面积 proposal 语义精度低，适合错误解释；
中等精度 proposal 覆盖面积较大，但误伤风险也高。
```

因此，自动 refinement 的困难在于：

```text
真正可靠的 proposal 太小；
真正覆盖大面积疑难区域的 proposal 又不够语义可靠。
```

这就是为什么 Exp5-D 会出现：

```text
changed pixels 很少；
但 degraded pixels 明显多于 improved pixels。
```

## 7. 对最终体系的意义

Exp5-E 给出了比“继续调阈值”更重要的结论：

```text
SAM3 在遥感推理分割体系中不应只有一个角色。
```

它至少可以拆成三种角色：

```text
1. object proposal layer
   用于 building、car 等对象性较强的候选区域。

2. error/risk explanation layer
   用于提示 coarse model 的疑难或错误风险区域。

3. interactive refinement proposal
   用于人工确认、交互式标注或半自动修正。
```

这比“自动覆盖 SegFormer prediction”更稳，也更符合可迁移遥感推理分割框架。

## 8. 对 Potsdam 泛化风险的控制

Exp5-E 没有引入新的 Potsdam 专属规则。

它使用的是通用审计维度：

```text
mask class
mask score
mask area
candidate overlap
error overlap
GT precision for evaluation only
Exp5-D acceptance/rejection status
```

在迁移到其他遥感数据集时，这些维度仍然成立。

需要变化的只是：

```text
类别名称；
prompt 词表；
coarse model；
领域空间先验。
```

因此，Exp5-E 比继续调 Potsdam 后处理阈值更接近论文最终目标。

## 9. 当前结论

一句话总结：

```text
Exp5-E 表明，SAM3 candidate masks 中确实存在高质量对象 proposals，
但其最可靠的价值不是自动语义覆盖，而是对象候选、错误解释和交互式 refinement。
```

对后续体系设计的直接影响是：

```text
最终方法不应追求 SAM3 自动改完整语义图；
而应输出 coarse prediction + candidate object layer + uncertainty/risk layer + optional refinement suggestions。
```

这使论文从“融合后处理”转向更合理的：

```text
可解释遥感推理分割体系。
```
