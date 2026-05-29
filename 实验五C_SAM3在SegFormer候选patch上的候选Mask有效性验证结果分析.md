# 实验五C：SAM3 在 SegFormer candidate patches 上的候选 mask 有效性验证结果分析

## 1. 实验目的回顾

Exp5-C 的核心问题是：

```text
在 SegFormer 已经标出的疑难 candidate patches 上，
SAM3 生成的候选 mask 是否能够有效覆盖 SegFormer 的错误区域？
```

这个实验不是直接做最终融合，而是判断 SAM3 是否具备作为局部 refinement 候选生成器的价值。

如果 SAM3 在这些候选 patch 上能够稳定覆盖错误区域，那么后续可以继续设计 SegFormer + SAM3 的局部修正规则；如果覆盖弱或不稳定，就说明 SAM3 不能简单作为监督语义模型之后的通用修正模块。

## 2. 总体结果

本次实验在 heldout split 上选取 60 个 SegFormer candidate patches，并在这些 patch 上运行 SAM3 prompt ensemble。

聚合结果如下：

```text
patches: 60
SAM3 masks: 465
candidate pixels: 26,961,588
SegFormer error pixels: 10,602,086
clutter/background FN pixels: 3,107,873
SAM3 union pixels: 7,644,700

sam_error_coverage: 0.2366
sam_clutter_fn_coverage: 0.1988
sam_error_precision: 0.3281
sam_clutter_fn_precision: 0.0808
sam_candidate_overlap_ratio: 0.2835
```

可以得到一个直接判断：

```text
SAM3 在 candidate patches 中确实命中了部分 SegFormer 错误区域，
但覆盖率和精度都不足以支撑“直接用 SAM3 修正 SegFormer”的方案。
```

其中，SAM3 mask union 覆盖了约 23.66% 的 SegFormer 错误像素，覆盖了约 19.88% 的 clutter/background 漏检像素。这个结果说明 SAM3 不是完全无效，但它的候选 mask 仍然偏稀疏、偏不稳定。

## 3. patch 级结果特征

60 个 patch 的 patch-level 统计如下：

```text
zero-mask patches: 14 / 60
total SAM3 masks: 465
num_masks per patch: 0 - 103

mean sam_error_coverage: 0.2278
median sam_error_coverage: 0.0161

mean sam_clutter_fn_coverage: 0.1525
median sam_clutter_fn_coverage: 0.0000

patches with sam_error_coverage >= 0.5: 13 / 60
patches with sam_clutter_fn_coverage >= 0.5: 9 / 60
patches with sam_error_precision >= 0.4: 7 / 60
patches with sam_clutter_fn_precision >= 0.2: 4 / 60
```

这组数字很关键：

```text
平均覆盖率看起来尚可，但中位数很低。
```

也就是说，SAM3 的效果高度集中在少数 patch 上：

1. 少数 patch 中 SAM3 可以覆盖大量错误区域。
2. 大量 patch 中 SAM3 几乎没有提供有效候选。
3. 14 个 patch 没有生成任何有效 mask。
4. clutter/background FN 的中位覆盖率为 0，说明 SAM3 对背景漏检问题并不稳定。

因此，SAM3 的局部候选能力是“选择性有效”，而不是“普遍有效”。

## 4. 类别层面的现象

按 SAM3 prompt 类别统计 mask 后，结果呈现出明显不均衡：

```text
class 0 impervious surface:
  masks: 30
  eval area: 7,707,872
  error precision: 0.3312
  clutter precision: 0.0794
  GT class precision: 0.1285

class 1 building:
  masks: 13
  eval area: 91,208
  error precision: 0.0906
  clutter precision: 0.0351
  GT class precision: 0.8140

class 2 low vegetation:
  masks: 19
  eval area: 395,766
  error precision: 0.3398
  clutter precision: 0.0078
  GT class precision: 0.4691

class 3 tree:
  masks: 0

class 4 car:
  masks: 403
  eval area: 40,532
  error precision: 0.1868
  clutter precision: 0.0000
  GT class precision: 0.8862
```

这说明 SAM3 的候选 mask 并不是均匀地服务于所有类别：

```text
impervious surface mask 数量少，但面积巨大，是主要覆盖来源；
car mask 数量极多，但面积很小，更多是局部实例候选；
building 和 car 的 GT precision 较高，但覆盖面积有限；
tree 在当前阈值和 prompt 下没有产生有效 mask。
```

尤其需要注意 class 0：

```text
impervious surface 的 mask 覆盖面积很大，
但 GT class precision 只有 0.1285。
```

这意味着它经常不是精确识别“硬化地表”，而是生成了较大的区域 mask，虽然能扫到一部分错误像素，但类别语义本身不可靠。

## 5. 大 mask 与小 mask 的差异

SAM3 mask 面积分布非常偏：

```text
mask eval_area min/max: 0 - 532,001
mask eval_area mean: 17,710
mask eval_area median: 53
large masks > 100,000 pixels: 23
tiny masks < 1,000 pixels: 397
non-empty masks: 420 / 465
```

这说明输出主要由两类 mask 组成：

```text
少数超大 impervious-like masks；
大量极小 car-like masks。
```

这种分布对融合很不友好：

1. 大 mask 容易覆盖错误区域，但也容易误伤大量正确区域。
2. 小 mask 类别精度较高，但对整体错误修复贡献很小。
3. 如果不做面积过滤、类别过滤和置信度校准，SAM3 mask 很难直接转化为稳定的语义修正。

## 6. 对 clutter/background 问题的启示

Exp4 和 Exp5-A/B 已经发现，SegFormer 的主要短板之一是 clutter/background recall 偏低，很多 clutter/background 被预测为 impervious、building 或 low vegetation。

Exp5-C 进一步说明：

```text
SAM3 对 clutter/background FN 有一定覆盖，但不稳定。
```

整体上：

```text
sam_clutter_fn_coverage = 0.1988
sam_clutter_fn_precision = 0.0808
```

这说明：

1. SAM3 可以覆盖约 19.88% 的 clutter/background 漏检区域。
2. 但 SAM3 mask 中真正属于 clutter/background FN 的比例只有 8.08%。
3. 因为本实验没有主动 prompt background，SAM3 并不能直接告诉我们“哪里是背景”。
4. SAM3 更多是在生成前景对象或前景区域 mask，而不是解决 background 语义类别。

所以，SAM3 不适合作为 clutter/background 的直接分类器。更合理的用法是：

```text
用 SAM3 mask 解释候选区域内部的前景对象结构；
没有被可靠前景 mask 覆盖、且 SegFormer 低置信或高熵的区域，才可能被进一步判为 clutter/background 风险区。
```

## 7. 与前序实验的联系

Exp1-Exp3 表明：

```text
SAM3 zero-shot / prompt ensemble 无法独立完成 Potsdam 六类密集语义分割。
```

Exp4 表明：

```text
SegFormer 监督训练可以提供稳定 dense semantic baseline。
```

Exp5-A/B 表明：

```text
SegFormer 的不确定性、margin 和边界区域可以较好定位错误候选区。
```

Exp5-C 进一步表明：

```text
SAM3 在这些候选区中具有局部候选 mask 价值，
但这种价值是稀疏的、类别不均衡的，不能直接转化为全局语义修正。
```

因此，目前最稳妥的结论不是“SegFormer + SAM3 融合已经成功”，而是：

```text
SegFormer candidate region 可以有效缩小需要检查的区域；
SAM3 可以在其中提供部分对象级候选；
但 SAM3 candidate masks 必须经过严格筛选，才能进入 refinement。
```

## 8. 当前结论

Exp5-C 支持以下结论：

```text
SAM3 适合作为候选区域内的 object/mask proposal generator，
但不适合作为直接覆盖 SegFormer 错误区域的可靠 dense correction module。
```

更具体地说：

1. SAM3 可以命中部分 SegFormer 错误区域，说明它不是完全无用。
2. SAM3 对错误区域的整体覆盖率约 23.66%，不够高。
3. SAM3 对 clutter/background FN 的覆盖率约 19.88%，更不稳定。
4. SAM3 输出强烈依赖类别和 prompt，class 0 大 mask 与 class 4 小 mask 主导结果。
5. 直接把 SAM3 mask 覆盖区域改成 SAM3 prompt 类别，风险很高。
6. 后续融合必须使用限制条件，例如面积过滤、类别一致性、SegFormer 低置信门控、边界门控、mask precision 估计等。

## 9. 下一步建议

下一步不建议直接做“SAM3 覆盖哪里就改哪里”的硬融合。

更合理的是做 Exp5-D：

```text
SAM3 candidate mask 筛选与保守 refinement 规则验证
```

建议先从保守规则开始：

1. 只在 SegFormer candidate region 内考虑 SAM3。
2. 只接受面积不过大的 mask，避免 class 0 巨型 mask 误伤。
3. 对 building、car 等 GT precision 较高的类别优先尝试 refinement。
4. 对 impervious surface 大 mask 暂时只作为不确定区域解释，不直接改标签。
5. 对 clutter/background 不直接由 SAM3 预测，而是结合 SegFormer 低置信、高熵、无可靠前景 mask 覆盖等条件推断。

一句话总结：

```text
Exp5-C 证明了 SAM3 在 SegFormer 候选区域中有局部 proposal 价值，
但也证明了它不能被直接当作可靠语义修正器；后续必须转向“候选筛选 + 保守融合”。
```
