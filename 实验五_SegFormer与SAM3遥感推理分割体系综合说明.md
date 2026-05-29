# 实验五：SegFormer 与 SAM3 遥感推理分割体系综合说明

## 1. 实验五定位

实验五用于验证：

```text
如何将监督语义模型、开放词汇分割模型、不确定性分析和候选区域解释结合，
构建面向遥感图像的推理分割技术体系。
```

它不是 Potsdam 专属后处理实验。

## 2. 实验序列

| 实验 | 文件/目录 | 作用 |
|---|---|---|
| Exp5-A | `eval_heldout_uncertainty` | 分析 SegFormer 不确定性与错误关系 |
| Exp5-B | `results_exp5b_segformer_candidate_regions_heldout` | 生成 refinement / review candidate region |
| Exp5-C | `results_exp5c_sam3_candidate_patch_validation_heldout` | 验证 SAM3 在 candidate patches 上的 mask 有效性 |
| Exp5-D | `results_exp5d_sam3_conservative_refinement_heldout` | 验证保守自动 refinement 是否可行 |
| Exp5-E | `results_exp5e_sam3_proposal_quality_audit_heldout` | 审计 SAM3 proposal 适合承担的角色 |
| Exp5-F | `results_exp5f_reasoning_segmentation_package_heldout` | 生成阶段性推理分割输出包 |

## 3. 核心结果

Exp5-A：

```text
错误像素比正确像素更低 confidence、更高 entropy、更低 margin。
high entropy top20 可覆盖 70.70% 总错误。
```

Exp5-B：

```text
candidate ratio: 31.11%
error coverage: 80.47%
clutter FN coverage: 65.86%
```

Exp5-C：

```text
SAM3 masks: 465
sam_error_coverage: 0.2366
sam_error_precision: 0.3281
sam_clutter_fn_coverage: 0.1988
```

Exp5-D：

```text
changed pixels: 6,420
improved pixels: 810
degraded pixels: 4,928
change precision: 0.1262
degradation rate: 0.7676
```

Exp5-E：

```text
object_proposal_high_precision:
  masks: 230
  GT precision: 0.9928

error_region_explanation:
  masks: 44
  error precision: 0.4126

large_surface_risky:
  masks: 14
  GT precision: 0.1508
```

Exp5-F：

```text
high priority ratio: 0.1115
high priority error coverage: 0.4969
candidate ratio: 0.3111
candidate error coverage: 0.8047
```

## 4. 最终判断

实验五最终说明：

```text
SAM3 不适合直接作为 SegFormer 的自动标签修正器；
SAM3 更适合作为开放词汇对象候选与错误解释模块；
最终输出应从单一语义图扩展为语义图 + 风险图 + 候选图 + 解释 manifest。
```

## 5. 阶段性产物

当前最重要的阶段性产物是：

```text
results_exp5f_reasoning_segmentation_package_heldout
```

该目录包含：

```text
final_predictions/
review_priority/
candidate_components/
overlays/
manifests/
metrics/
```

这代表了当前推理分割体系的输出雏形。

## 6. 后续方向

后续不建议继续围绕 Potsdam 阈值微调。

建议转向：

```text
1. 整理论文方法框架；
2. 将 Exp5-F 抽象为通用遥感推理分割系统；
3. 设计水利场景迁移方案；
4. 引入 DEM、河网、历史水体等水利先验；
5. 如果获得水利数据，再验证 review priority 与 proposal layer 的实际应用价值。
```
