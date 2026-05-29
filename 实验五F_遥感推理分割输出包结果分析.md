# 实验五F：遥感推理分割输出包结果分析

## 1. 实验目的

Exp5-F 的目标是把前面 Exp4-Exp5E 的结果组装成一个更接近最终技术体系的输出包。

它不再尝试直接让 SAM3 自动改写语义图，而是输出：

```text
稳定语义分割结果；
不确定性 / 风险区域；
候选复核区域；
SAM3 proposal 审计建议；
每张图的解释 manifest。
```

这一步更符合论文最终目标：

```text
构建一套可迁移、可解释、可扩展的遥感推理分割技术体系。
```

## 2. 方法定位

Exp5-F 的核心策略是：

```text
最终语义图 = SegFormer coarse prediction
SAM3 = candidate object / risk explanation / interactive proposal source
candidate region = 复核与解释区域
review priority = 面向人工、先验或后续模块的优先级图
```

因此，本实验不采用：

```text
SAM3 mask 覆盖哪里就改哪里。
```

而采用：

```text
coarse semantic result + review priority + proposal recommendations
```

这避免了 Exp5-D 中出现的高误伤问题。

## 3. 输出目录

```text
results_exp5f_reasoning_segmentation_package_heldout
```

主要输出包括：

```text
final_predictions/*_final_id.png
final_predictions/*_final.png
review_priority/*_priority.png
review_priority/*_high_priority.png
review_priority/*_medium_priority.png
candidate_components/*_candidate.png
candidate_components/*_high_entropy.png
candidate_components/*_low_margin.png
candidate_components/*_boundary_band.png
overlays/*_priority_overlay.png
manifests/*_manifest.json
manifests/package_manifest.json
metrics/reasoning_package_summary.json
metrics/reasoning_package_per_image.csv
```

这些文件共同构成一个可解释推理分割输出包。

## 4. 总体结果

在 14 张 heldout 图上：

```text
valid pixels: 467,862,855
candidate pixels: 145,556,699
high priority pixels: 52,163,786
medium priority pixels: 145,556,699
error pixels: 45,819,895
high priority error pixels: 22,769,011
medium priority error pixels: 36,869,626
```

对应比例：

```text
candidate ratio: 0.3111
high priority ratio: 0.1115
medium priority ratio: 0.3111
high priority error coverage: 0.4969
medium priority error coverage: 0.8047
```

最重要的结果是：

```text
高优先级复核区域只占 11.15% 有效像素，
但覆盖了 49.69% 的 SegFormer 错误像素。
```

这说明 review priority layer 是有效的错误聚焦工具。

## 5. 与 Exp5-B 的关系

Exp5-B 的 candidate region 覆盖：

```text
candidate ratio: 31.11%
error coverage: 80.47%
```

Exp5-F 在此基础上进一步生成 high priority 区域：

```text
high priority ratio: 11.15%
high priority error coverage: 49.69%
```

这意味着：

```text
candidate region 是广义复核区域；
high priority region 是更紧凑的重点复核区域。
```

两者可以形成分层工作流：

```text
快速复核：先看 high priority；
完整风险审计：再看 candidate / medium priority；
自动输出：仍保留 SegFormer final prediction。
```

## 6. 与 Exp5-D 的关系

Exp5-D 直接自动 refinement 的结果为：

```text
changed pixels: 6,420
improved pixels: 810
degraded pixels: 4,928
change precision: 0.1262
degradation rate: 0.7676
```

这说明自动写入风险较高。

Exp5-F 的转向是：

```text
不自动改标签；
保留稳定 coarse prediction；
将风险区域显式输出；
将 SAM3 proposal 审计建议写入 manifest。
```

因此，Exp5-F 不是失败后的退让，而是更稳健的体系化设计：

```text
模型负责预测；
系统负责解释；
候选模块负责提示；
后续先验或人工交互负责确认。
```

## 7. SAM3 proposal 在输出包中的角色

Exp5-F 将 Exp5-E 的推荐写入 package manifest：

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

这体现了 SAM3 的合理定位：

```text
不是直接语义覆盖器；
而是对象候选、风险解释和交互式复核 proposal 来源。
```

## 8. 对可迁移性的意义

Exp5-F 避免了 Potsdam 专属调参风险。

它使用的是通用模块：

```text
coarse prediction；
confidence / entropy / margin；
candidate region；
boundary band；
proposal recommendation；
per-image manifest。
```

迁移到其他遥感数据集时，需要替换的是：

```text
coarse model；
类别定义；
prompt 词表；
领域空间先验；
可选的人工或规则确认模块。
```

但整体输出结构仍然成立：

```text
final semantic map
+ uncertainty/risk map
+ candidate review map
+ object proposal layer
+ explanation manifest
```

这比单纯报告 mIoU 更接近“推理分割技术体系”。

## 9. 当前结论

Exp5-F 支持以下结论：

```text
在当前阶段，最合理的最终产物不是 SAM3 自动修正后的语义图，
而是以 SegFormer 为稳定 dense prediction，
以 uncertainty/candidate region 为风险定位，
以 SAM3 proposal audit 为对象候选和解释建议的推理分割输出包。
```

一句话概括：

```text
Exp5-F 将前面实验从“模型融合尝试”推进为“可解释遥感推理分割系统雏形”。
```

## 10. 后续方向

下一步建议不要继续围绕 Potsdam 做阈值微调，而是做两件事：

```text
1. 写实验五综合分析，明确从 Exp5-A 到 Exp5-F 的逻辑链；
2. 设计面向水利场景的迁移版本：
   coarse water/land prediction
   + water-related prompts
   + DEM/河网/历史水体先验
   + review priority / explanation package。
```

这样论文主线会从 Potsdam 验证自然过渡到水利遥感推理分割框架。
