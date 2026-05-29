# 实验五：SegFormer 与 SAM3 遥感推理分割体系综合分析

## 1. 实验五的总体问题

实验五不是单纯为了提高 Potsdam heldout mIoU，也不是为了证明 SAM3 可以替代 SegFormer。

它要回答的是一个更接近论文核心的问题：

```text
当开放词汇分割模型无法独立完成高质量遥感密集语义分割时，
能否将其与监督语义模型、不确定性分析、候选区域定位和解释机制结合，
形成一套可迁移的遥感推理分割技术体系？
```

因此，实验五的主线不是：

```text
SegFormer + SAM3 = 更高分数
```

而是：

```text
SegFormer 提供稳定 dense semantic prediction；
uncertainty / candidate region 定位风险区域；
SAM3 提供开放词汇 object proposal 和解释信号；
系统输出 final prediction + review priority + proposal recommendations。
```

## 2. 实验五的逻辑链

实验五可以分为六个阶段：

| 实验 | 核心问题 | 结论 |
|---|---|---|
| Exp5-A | SegFormer 错误是否能由不确定性定位？ | 大部分错误可由高 entropy / 低 margin 捕获，但存在高置信错分 |
| Exp5-B | 能否构建后续 SAM3 的候选区域？ | candidate region 用 31.11% 区域覆盖 80.47% 错误 |
| Exp5-C | SAM3 在候选 patch 上是否命中错误？ | 能命中一部分错误，但覆盖与精度不足以直接修正 |
| Exp5-D | 保守筛选后能否自动 refinement？ | 自动写入带来净负收益，误伤高于修正 |
| Exp5-E | SAM3 proposal 到底适合什么角色？ | 适合作为对象候选、错误解释、交互式 proposal |
| Exp5-F | 能否形成体系化输出？ | 可输出 final semantic map + review priority + proposal recommendations |

这条链条说明：

```text
实验五从“尝试融合”推进到了“明确模块分工”。
```

这比单纯调阈值提升 Potsdam 指标更有论文价值。

## 3. SegFormer 的角色：稳定 coarse semantic prediction

Exp4 已经证明，SegFormer supervised baseline 在 Potsdam 上能提供稳定的 dense semantic prediction。

heldout full-tile 指标为：

```text
OA: 0.9021
mIoU: 0.7687
Mean F1: 0.8535
FWIoU: 0.8251
```

逐类 IoU 中：

```text
building: 0.9248
car: 0.9034
impervious surface: 0.8576
tree: 0.7860
low vegetation: 0.7767
clutter/background: 0.3635
```

这说明：

```text
监督语义模型适合作为最终语义图的主体来源。
```

它的短板不是整体语义框架，而是：

```text
clutter/background recall 偏低；
局部边界或高混淆区域存在错误；
部分错误具有高置信特征。
```

因此后续模块应围绕：

```text
错误定位、风险解释、局部复核、候选对象发现
```

而不是试图整体替代 SegFormer。

## 4. 不确定性与候选区域：有效的错误定位工具

Exp5-A 表明，SegFormer 错误像素整体呈现：

```text
低 confidence
高 entropy
低 margin
```

正确像素与错误像素的均值对比为：

```text
correct:
  confidence mean = 0.9592
  entropy mean = 0.0590
  margin mean = 0.9275

error:
  confidence mean = 0.7708
  entropy mean = 0.2951
  margin mean = 0.5863
```

高 entropy top20 区域可以覆盖：

```text
70.70% 总错误；
55.07% clutter/background false negative。
```

Exp5-B 进一步合并：

```text
high entropy top20
OR low margin top20
OR selected-class boundary band
```

得到 candidate region：

```text
candidate ratio: 31.11%
error coverage: 80.47%
clutter FN coverage: 65.86%
```

这说明：

```text
uncertainty + boundary candidate region 是有效的风险区域定位模块。
```

但也暴露出限制：

```text
clutter -> impervious surface 等高置信错分仍难以完整捕获。
```

因此，candidate region 适合作为：

```text
后续复核、解释、SAM3 局部 proposal 的搜索空间。
```

而不是最终修正结果。

## 5. SAM3 的角色：候选 proposal，而非自动语义修正器

Exp5-C 在 candidate patches 上运行 SAM3，结果为：

```text
patches: 60
SAM3 masks: 465
sam_error_coverage: 0.2366
sam_clutter_fn_coverage: 0.1988
sam_error_precision: 0.3281
sam_clutter_fn_precision: 0.0808
sam_candidate_overlap_ratio: 0.2835
```

这说明：

```text
SAM3 可以命中部分 SegFormer 错误区域，
但覆盖不足、精度不足、类别不均衡。
```

进一步看 mask 形态：

```text
少数超大 impervious-like masks 主导面积；
大量 car-like masks 面积极小但精度较高；
building / car 更像对象候选；
large surface masks 更像风险解释，而不是可直接采纳的语义标签。
```

因此，SAM3 在该体系中的合理定位是：

```text
open-vocabulary object / mask proposal generator
```

而不是：

```text
dense semantic correction module
```

## 6. 为什么自动 refinement 不成立

Exp5-D 尝试了较保守的自动 refinement：

```text
只接受 building / car；
要求 score >= 0.55；
要求位于 candidate region；
要求满足 uncertainty gate；
限制 mask 面积；
只在与 SegFormer 预测不同的位置改动。
```

结果为：

```text
accepted masks: 41 / 465
changed pixels: 6,420
improved pixels: 810
degraded pixels: 4,928
change precision: 0.1262
degradation rate: 0.7676
```

完整 heldout 口径：

```text
baseline mIoU: 0.768670
refined mIoU: 0.768666
delta mIoU: -0.000005
```

这说明：

```text
即使非常保守，自动写入 SAM3 masks 仍然风险较高。
```

失败原因不是规则太保守，而是：

```text
被接受的 SAM3 masks 仍不能可靠判断是否应该覆盖 coarse prediction。
```

因此，继续在 Potsdam 上调阈值并不是理想方向。

## 7. Proposal 审计给出的新定位

Exp5-E 对 SAM3 proposals 进行质量分层。

总体上：

```text
mask_count: 465
weighted GT precision: 0.1562
weighted error precision: 0.3282
weighted clutter FN precision: 0.0751
```

面积加权语义精度低，说明 SAM3 masks 不适合直接作为类别覆盖。

但质量分层显示：

```text
object_proposal_high_precision:
  masks: 230
  eval_area: 75,607
  GT precision: 0.9928

error_region_explanation:
  masks: 44
  eval_area: 3,034,341
  error precision: 0.4126

large_surface_risky:
  masks: 14
  eval_area: 4,524,730
  GT precision: 0.1508
```

这给出非常清晰的模块分工：

```text
高精度小面积 masks:
  适合作为 object proposal / review proposal。

大面积低语义精度 masks:
  适合作为 risk explanation，不适合自动改标签。

中等精度 masks:
  需要额外先验，例如边界一致性、形态规则、多源遥感信息。
```

这一步把 SAM3 从“自动修正器”重新定位为：

```text
对象候选层 + 错误解释层 + 交互式复核建议。
```

## 8. Exp5-F：阶段性体系输出

Exp5-F 将前面结果组装为一个推理分割输出包：

```text
final_predictions/
review_priority/
candidate_components/
overlays/
manifests/
metrics/
```

输出逻辑为：

```text
final prediction = SegFormer coarse prediction
review priority = candidate / entropy / low margin / boundary 综合风险图
SAM3 proposal audit = 写入 manifest 的对象候选与风险解释建议
```

总体效果：

```text
candidate ratio: 0.3111
candidate error coverage: 0.8047

high priority ratio: 0.1115
high priority error coverage: 0.4969
```

也就是说：

```text
高优先级复核区域只占 11.15% 有效像素，
但覆盖了 49.69% 的错误像素。
```

这说明：

```text
review priority layer 可以有效浓缩错误风险区域。
```

因此，Exp5-F 形成了一个比硬融合更稳健的阶段性系统产物：

```text
stable semantic map
+ risk / review priority map
+ candidate region components
+ SAM3 proposal recommendations
+ per-image explanation manifest
```

## 9. 对论文主线的意义

实验五证明的不是：

```text
SAM3 能直接提升 SegFormer 的语义分割指标。
```

而是：

```text
开放词汇分割模型可以作为遥感推理分割体系中的候选 proposal 与解释模块；
监督语义模型负责稳定 dense prediction；
不确定性和边界先验负责定位复核区域；
最终系统输出不应只有 mask，而应包含风险、解释和候选信息。
```

这使论文主线从：

```text
单模型分割性能比较
```

推进到：

```text
可解释、可交互、可迁移的遥感推理分割技术体系。
```

这是比单纯追求 Potsdam mIoU 更合理的研究贡献。

## 10. 对泛化性的约束

为了避免演化成 Potsdam 专属微调，实验五中保留的应是通用机制：

```text
coarse semantic prediction；
confidence / entropy / margin；
candidate region；
boundary band；
open-vocabulary object proposal；
proposal quality audit；
review priority map；
explanation manifest。
```

不应保留为最终方法核心的是：

```text
针对 Potsdam 某张 tile 的特殊规则；
针对 heldout 标签调出来的阈值；
直接把 SAM3 mask 写入语义图的硬规则；
只服务于 Potsdam 六类标签的不可迁移后处理。
```

这一区分非常重要。

Potsdam 在这里的作用是：

```text
提供高分辨率遥感语义分割验证平台；
帮助判断各模块是否有价值；
不应成为最终方法的绑定对象。
```

## 11. 阶段性结论

实验五的最终结论可以概括为：

```text
SegFormer + SAM3 的合理协同方式不是自动标签覆盖，
而是“稳定语义图 + 风险定位 + 候选对象 + 解释 manifest”的推理分割输出体系。
```

更具体地说：

```text
SegFormer:
  输出稳定 dense semantic prediction。

Uncertainty / boundary:
  定位错误风险和复核区域。

SAM3:
  生成对象候选、开放词汇 proposal 和错误解释信号。

Fusion / reasoning layer:
  不直接硬改标签，而是输出 review priority、proposal recommendation 和可解释 manifest。
```

这说明当前最合理的研究方向是：

```text
从“模型融合提升分数”转向“推理分割系统构建”。
```

## 12. 下一步建议

下一步建议进入两个方向。

第一，写论文中的方法框架：

```text
Remote-Sensing Reasoning Segmentation Framework
  Coarse Semantic Segmenter
  Uncertainty & Candidate Region Generator
  Open-Vocabulary Proposal Generator
  Proposal Quality Auditor
  Review Priority & Explanation Packager
```

第二，设计水利场景迁移版本：

```text
coarse water/land or land-cover prediction；
SAM3 water / river / flood / levee prompts；
DEM / river network / historical water extent priors；
review priority map；
water-conservancy explanation manifest。
```

在没有水利数据集的当前阶段，可以先写出方法框架和迁移方案，再寻找或构造小规模水利案例进行验证。
