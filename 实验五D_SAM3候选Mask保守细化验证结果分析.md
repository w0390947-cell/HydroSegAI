# 实验五D：SAM3 候选 mask 保守细化验证结果分析

## 1. 实验目的回顾

Exp5-D 的目标是验证：

```text
SAM3 在 SegFormer candidate patches 上生成的候选 masks，
经过保守筛选后，是否可以安全地用于局部 refinement？
```

这个实验不是为了对 Potsdam 做专门调参，而是为了判断一种可迁移的推理分割组件是否成立：

```text
coarse semantic prediction
  + candidate region
  + open-vocabulary mask proposal
  + conservative gates
  -> refined semantic prediction
```

其中，GT 标签只用于评价，不参与 refinement 决策。

## 2. 当前规则

本次 Exp5-D 使用的规则较保守：

```text
accepted_class_ids: [1, 4]
score_threshold: 0.55
min_mask_area: 16
max_mask_area_fraction: 0.08
require_candidate_region: true
require_uncertainty_gate: true
apply_only_where_prediction_differs: true
mask_priority: score
```

也就是说，本实验没有让 SAM3 直接修正所有类别，而是先只允许相对更像对象候选的：

```text
building
car
```

进入 refinement。

这样做的目的不是针对 Potsdam 类别定制，而是验证一个通用原则：

```text
开放词汇 mask proposer 更适合先修正边界清晰、对象性较强的类别，
而不是直接修正大面积地表类别。
```

## 3. 结果总览

根据当前输出：

```text
results_exp5d_sam3_conservative_refinement_heldout
```

在 60 个 candidate patches 上，SAM3 共生成：

```text
total SAM3 masks: 465
accepted masks: 41
accepted ratio: 8.82%
changed pixels: 6,420
improved pixels: 810
degraded pixels: 4,928
still wrong changed pixels: 682
change precision: 0.1262
degradation rate: 0.7676
```

核心结论非常明确：

```text
当前保守规则并没有形成可靠 refinement。
虽然改动像素很少，但被改坏的像素远多于被改对的像素。
```

被改动的 6,420 个像素中：

```text
12.62% 是真正修正；
76.76% 是误伤；
10.62% 是从一种错误改成另一种错误。
```

这说明：

```text
仅靠 SAM3 score + candidate region + uncertainty gate + 面积过滤，
还不足以判断一个 SAM3 mask 是否应该自动写入最终语义图。
```

## 4. 指标变化

当前脚本输出的 processed image 口径包含 13 张图，因为 heldout 中有 1 张图没有入选 candidate patch。

processed image 口径下：

```text
baseline OA: 0.901566
refined OA: 0.901557
delta OA: -0.000009

baseline mIoU: 0.768540
refined mIoU: 0.768535
delta mIoU: -0.000005
```

按完整 heldout 14 张图、公平地将无 refinement 图保持原预测不变后，结果为：

```text
baseline OA: 0.902066
refined OA: 0.902057
delta OA: -0.000009

baseline mIoU: 0.768670
refined mIoU: 0.768666
delta mIoU: -0.000005
```

也就是说：

```text
整体指标几乎不变，但方向是轻微下降。
```

这说明当前 refinement 规则影响范围很小，但净收益为负。

## 5. 类别 IoU 变化

完整 heldout 口径下，各类 IoU 变化为：

```text
impervious surface: 0.857596 -> 0.857571  delta -0.000025
building:           0.924765 -> 0.924727  delta -0.000038
low vegetation:     0.776710 -> 0.776713  delta +0.000003
tree:               0.786032 -> 0.786033  delta +0.000001
car:                0.903402 -> 0.903438  delta +0.000035
clutter/background: 0.363517 -> 0.363512  delta -0.000005
```

可以看到：

```text
car 类有极小提升；
building 类反而略降；
整体 mIoU 净下降。
```

这与 Exp5-C 的发现一致：

```text
car mask 通常更小、更精确；
building mask 虽有较高 GT precision，但在自动写入语义图时仍会带来局部误伤。
```

## 6. mask 接受与拒绝情况

465 个 SAM3 masks 中：

```text
accepted: 41
area_too_small: 228
score_below_threshold: 147
class_not_accepted: 49
```

按类别统计：

```text
all masks:
  class 0: 30
  class 1: 13
  class 2: 19
  class 4: 403

accepted masks:
  class 1: 4
  class 4: 37
```

这说明当前规则确实过滤掉了大部分 SAM3 输出，尤其是：

```text
大量小 car masks 因面积过小或 score 不足被过滤；
class 0 / class 2 被策略性排除；
最终实际进入 refinement 的 masks 很少。
```

这验证了一个重要事实：

```text
SAM3 candidate mask 输出数量多，但能进入自动 refinement 的候选很少。
```

## 7. 图像级差异

13 张 processed images 中，只有 4 张图发生了实际改动：

```text
top_potsdam_5_13:
  changed 96
  improved 96
  degraded 0
  change precision 1.000

top_potsdam_4_14:
  changed 651
  improved 405
  degraded 222
  change precision 0.622

top_potsdam_5_14:
  changed 67
  improved 19
  degraded 48
  change precision 0.284

top_potsdam_6_15:
  changed 5606
  improved 290
  degraded 4658
  change precision 0.052
```

其中 `top_potsdam_6_15` 主导了负面结果：

```text
changed pixels: 5,606
improved pixels: 290
degraded pixels: 4,658
degradation rate: 0.831
```

这说明自动 refinement 的最大风险不是平均表现，而是少数 patch 或少数 mask 的灾难性误伤。

## 8. 对技术路线的含义

Exp5-D 的结果不支持：

```text
直接将通过筛选的 SAM3 masks 写入最终 semantic map。
```

即便当前规则已经比较保守，仍然出现：

```text
degraded pixels >> improved pixels
```

这说明 SAM3 mask 的 score、面积和不确定性门控还不能充分表达：

```text
该 mask 是否语义正确；
该 mask 是否应覆盖 coarse model 的预测；
该 mask 是否只是局部对象候选而非语义类别修正。
```

因此，SAM3 在当前体系中的定位应进一步收敛为：

```text
候选解释模块 / 交互式 refinement 提示模块 / 对象候选发现模块
```

而不是：

```text
自动语义标签覆盖模块
```

## 9. 与论文目标的关系

从论文主线看，Exp5-D 是有价值的。

它说明：

```text
构建遥感推理分割体系时，不能简单把开放词汇模型的 mask proposal 当成最终语义结果；
即使有 coarse semantic prior 和 uncertainty candidate region，也仍然需要更强的规则、先验或人工确认机制。
```

这有助于避免论文走偏成：

```text
为了 Potsdam 分数而不断调后处理规则。
```

相反，它支持一个更稳的框架结论：

```text
粗语义模型负责稳定 dense prediction；
开放词汇模型负责提供候选对象和解释；
自动融合必须非常保守；
水利或遥感空间先验应成为决定是否采纳候选 mask 的关键约束。
```

## 10. 下一步建议

下一步不建议继续单纯放宽 Exp5-D 的阈值，因为当前不是“改得太少”的问题，而是：

```text
被接受的改动中误伤比例过高。
```

更合理的后续方向有两个：

### 方向一：Exp5-E，改为 mask proposal 质量评估，而不是自动写入

继续分析 SAM3 masks：

```text
哪些 mask 可以作为人工交互候选？
哪些 mask 可以作为边界解释？
哪些 mask 只能作为不确定区域提示？
```

输出不直接改语义图，而是生成：

```text
candidate object layer
uncertainty explanation layer
review priority map
```

这更符合可解释推理分割体系。

### 方向二：引入更强的可迁移先验后再 fusion

例如：

```text
类别一致性：
  SAM3 building mask 只在 coarse model 同时对 building 有较高概率时采纳。

边界一致性：
  SAM3 mask 边界必须与 coarse model 边界或影像梯度一致。

对象形态先验：
  车辆、小建筑、线性水体等对象使用不同形态规则。

多源先验：
  对水利任务引入 DEM、河网、历史水体等空间约束。
```

这比继续在 Potsdam 上调 `score_threshold` 更符合论文目标。

## 11. 当前结论

一句话总结 Exp5-D：

```text
SAM3 候选 mask 经过保守筛选后仍不足以安全自动修正 SegFormer 语义结果；
其更合理的角色是候选对象发现与解释层，而不是直接标签覆盖层。
```

这个结论对最终体系很重要：

```text
推理分割框架需要 coarse model、open-vocabulary mask proposer、candidate region、空间先验和质量控制协同；
不能把任一单个模块的输出直接当成最终结果。
```
