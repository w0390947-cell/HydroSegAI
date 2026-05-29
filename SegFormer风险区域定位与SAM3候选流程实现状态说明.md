# SegFormer 风险区域定位与 SAM3 候选流程实现状态说明

## 1. 本文档要说明的问题

当前我们讨论的系统流程是：

```text
SegFormer 输出语义图和类别概率
        ↓
计算 low confidence / high entropy
        ↓
提取类别边界区域
        ↓
结合弱类别、小目标、空间先验
        ↓
生成 candidate regions
        ↓
把这些区域交给 SAM3 或候选筛选模块
```

这个流程的目的不是让 SAM3 重新处理整张图，也不是让 SAM3 直接替代 SegFormer，而是：

```text
先用 SegFormer 得到稳定的 dense semantic prediction；
再找出 SegFormer 可能出错、值得复核或值得进一步生成 proposal 的局部区域；
最后让 SAM3 只在这些候选区域中生成 candidate masks。
```

从当前代码和实验结果看，这个流程已经实现了主要骨架，但还没有实现成最终的自然语言推理分割系统。

## 2. 总体实现状态

| 流程环节 | 当前状态 | 对应实验 / 脚本 | 说明 |
|---|---|---|---|
| SegFormer 输出完整语义图 | 已实现 | Exp4，`train_segformer_potsdam.py`，`eval_segformer_potsdam.py` | 已训练 Potsdam RGB SegFormer，并完成 val / heldout full-tile evaluation |
| 输出类别概率 / 不确定性图 | 已实现 | Exp5-A，`eval_segformer_potsdam.py --save-uncertainty` | 已保存 confidence、entropy、margin、error 图 |
| low confidence 分析 | 已实现分析，未进入 Exp5-B candidate 合并 | Exp5-A | 已统计 low confidence 与错误关系，但 Exp5-B 第一版 candidate region 没有显式使用 low confidence |
| high entropy 区域提取 | 已实现 | Exp5-A / Exp5-B，`exp5b_segformer_candidate_regions.py` | 已使用每张图 entropy top20% 作为候选区域组成部分 |
| low margin 区域提取 | 已实现 | Exp5-A / Exp5-B，`exp5b_segformer_candidate_regions.py` | 已使用每张图 margin lowest20% 作为候选区域组成部分 |
| 类别边界区域提取 | 已实现第一版 | Exp5-B，`boundary_band_from_prediction` | 已对指定预测类别边界做 morphology gradient + dilation |
| 弱类别处理 | 部分实现 | Exp5-A / Exp5-B | 已针对 `clutter/background` 错误做分析，并在 boundary classes 中纳入 clutter，但还没有通用弱类别模块 |
| 小目标机制 | 尚未系统实现 | Exp5-C/E 有间接分析 | SAM3 car-like 小 mask 质量较高，但还没有专门的小目标候选生成和排序机制 |
| 空间先验冲突检测 | 尚未实现 | 后续 Exp6 / 水利迁移方向 | 尚未接入 DSM / DEM / NDVI / NDWI / 河网 / 水体先验 |
| candidate regions 生成 | 已实现 | Exp5-B，`exp5b_segformer_candidate_regions.py` | 已生成 candidate、high entropy、low margin、boundary band 及 overlay |
| candidate patches 生成并交给 SAM3 | 已实现 | Exp5-C，`exp5c_sam3_candidate_patch_validation.py` | 已在 candidate patches 上运行 SAM3 并评估 proposal 有效性 |
| SAM3 自动 refinement | 已实现并验证不可靠 | Exp5-D，`exp5d_sam3_conservative_refinement.py` | 保守写入产生净负收益，不建议作为主路线 |
| SAM3 proposal 质量审计 | 已实现 | Exp5-E，`exp5e_sam3_proposal_quality_audit.py` | 已区分 object proposal、error explanation、large surface risky 等层级 |
| reasoning package 输出 | 已实现雏形 | Exp5-F，`exp5f_reasoning_segmentation_package.py` | 已输出 final prediction、review priority、candidate components、manifest |
| 自然语言 query 筛选候选 mask | 尚未实现 | 后续 Exp6 | 还没有实现 query 解析、候选 mask 排序和目标选择 |

## 3. 已实现部分一：SegFormer 完整语义预测

这一部分已经由 Exp4 完成。

当前系统已经可以使用训练好的 SegFormer-B0 对 Potsdam heldout full-tile 进行完整语义预测。它输出的是每个像素的类别：

```text
impervious surface
building
low vegetation
tree
car
clutter/background
```

对应脚本是：

```text
train_segformer_potsdam.py
eval_segformer_potsdam.py
```

heldout full-tile 结果为：

```text
OA      = 0.9021
mIoU    = 0.7687
Mean F1 = 0.8535
FWIoU   = 0.8251
```

这说明 SegFormer 已经可以作为系统中的稳定场景语义 prior。也就是说，它负责先给出一张完整的语义底图。

但它不是完美的。当前主要短板是：

```text
clutter/background IoU = 0.3635
```

这个类的主要问题是欠召回，即真实的 clutter/background 容易被预测成 impervious surface、low vegetation 或 building。

## 4. 已实现部分二：不确定性图输出

这一部分由 Exp5-A 完成。

运行命令是：

```bash
python eval_segformer_potsdam.py \
  --config segformer_potsdam_rgb.yaml \
  --split heldout \
  --save-uncertainty \
  --output-dir results_segformer_potsdam_rgb/eval_heldout_uncertainty
```

输出目录是：

```text
results_segformer_potsdam_rgb/eval_heldout_uncertainty/
```

该目录中除了常规预测结果外，还额外保存：

```text
uncertainty/*_confidence.png
uncertainty/*_entropy.png
uncertainty/*_margin.png
uncertainty/*_error.png
```

这些图的含义是：

| 输出 | 含义 | 判断方式 |
|---|---|---|
| confidence | top-1 softmax probability | 越低越不确定 |
| entropy | 归一化 softmax entropy | 越高越不确定 |
| margin | top-1 probability - top-2 probability | 越低越不确定 |
| error | prediction 是否等于 ground truth | 仅用于实验分析，不用于真实推理 |

Exp5-A 的结论是：

```text
错误像素整体上确实更低 confidence、更高 entropy、更低 margin。
```

正确像素与错误像素的平均值对比为：

```text
correct:
  confidence mean = 0.9592
  entropy mean    = 0.0590
  margin mean     = 0.9275

error:
  confidence mean = 0.7708
  entropy mean    = 0.2951
  margin mean     = 0.5863
```

因此，不确定性图可以作为错误风险定位的有效信号。

## 5. 已实现部分三：high entropy / low margin 候选区域

这一部分由 Exp5-B 实现。

对应脚本是：

```text
exp5b_segformer_candidate_regions.py
```

运行时使用：

```bash
python exp5b_segformer_candidate_regions.py \
  --eval-dir results_segformer_potsdam_rgb/eval_heldout_uncertainty \
  --output-dir results_exp5b_segformer_candidate_regions_heldout \
  --entropy-fraction 0.20 \
  --margin-fraction 0.20 \
  --boundary-radius 8 \
  --boundary-classes 0,1,2,5
```

其中已经实现了两个不确定性候选区域：

```text
high_entropy_top20:
每张图 entropy 最高的 20% 有效像素。

low_margin_top20:
每张图 margin 最低的 20% 有效像素。
```

注意：当前 Exp5-B 第一版 candidate region 使用了 `high entropy` 和 `low margin`，但没有把 `low confidence` 作为合并条件之一。`confidence` 已经在 Exp5-A 中输出并分析过，但还没有进入候选区域生成公式。

也就是说：

```text
low confidence:
  已计算、已保存、已分析；
  但尚未作为 Exp5-B candidate region 的显式组成部分。

high entropy:
  已计算、已保存、已用于 candidate region。

low margin:
  已计算、已保存、已用于 candidate region。
```

## 6. 已实现部分四：类别边界区域提取

类别边界区域也已经在 Exp5-B 中实现了第一版。

代码函数是：

```text
boundary_band_from_prediction(...)
```

它的逻辑是：

```text
1. 从 SegFormer prediction_id 图中取出指定类别；
2. 对每个指定类别的二值 mask 做 morphology gradient；
3. 得到该类别的预测边界；
4. 用 dilation 将边界向外扩张；
5. 只保留有效像素范围内的边界带。
```

当前使用的参数是：

```text
boundary_radius = 8
boundary_classes = 0,1,2,5
```

对应类别是：

```text
0: impervious surface
1: building
2: low vegetation
5: clutter/background
```

也就是说，当前系统已经能自动提取：

```text
道路/硬化地表边界
建筑边界
低矮植被边界
clutter/background 相关边界
```

这些边界区域会参与最终 candidate region 合并。

Exp5-B 中，boundary band 单独表现为：

```text
区域占比: 21.12%
error 覆盖率: 52.93%
clutter FN 覆盖率: 47.44%
clutter -> building 覆盖率: 75.75%
```

这说明边界区域对某些类别混淆很有价值，尤其是 clutter 被错分为 building 的情况。

## 7. 已实现部分五：candidate regions 合并

Exp5-B 当前的候选区域定义是：

```text
candidate_region =
    high_entropy_top20
    OR low_margin_top20
    OR boundary_band(predicted classes in {impervious, building, low vegetation, clutter})
```

这个逻辑已经完整实现，并输出了以下文件：

```text
results_exp5b_segformer_candidate_regions_heldout/
  masks/*_candidate.png
  masks/*_high_entropy.png
  masks/*_low_margin.png
  masks/*_boundary_band.png
  overlays/*_candidate_overlay.png
  metrics/candidate_region_summary.json
  metrics/candidate_region_per_image.csv
```

数据集级结果为：

```text
candidate ratio = 31.11%
error coverage = 80.47%
clutter FN coverage = 65.86%
```

这说明：

```text
系统只用约 31.11% 的有效像素区域，就覆盖了 80.47% 的 SegFormer 错误像素。
```

因此，candidate region 已经可以作为 SAM3 局部 proposal 的第一版搜索空间。

但它也有明显限制：

```text
clutter -> impervious surface 覆盖率只有 54.48%；
top_potsdam_4_14_RGB 这类高置信系统性错分仍难捕获；
candidate 区域仍偏大，不能直接逐像素运行 SAM3。
```

## 8. 已实现部分六：candidate patches 交给 SAM3

Exp5-C 已经把 candidate region 进一步转成 candidate patches，并在这些 patch 上运行 SAM3。

对应脚本是：

```text
exp5c_sam3_candidate_patch_validation.py
```

这个实验验证的问题是：

```text
在 SegFormer candidate patches 上运行 SAM3，
SAM3 生成的 masks 是否能命中 SegFormer 错误区域？
```

结果为：

```text
candidate patches: 60
SAM3 masks: 465
sam_error_coverage: 0.2366
sam_clutter_fn_coverage: 0.1988
sam_error_precision: 0.3281
sam_clutter_fn_precision: 0.0808
sam_candidate_overlap_ratio: 0.2835
```

这说明：

```text
SAM3 在 candidate patches 上确实能命中一部分错误区域；
但覆盖率和精度都不足以支撑“直接自动修正”。
```

因此，SAM3 当前更合理的定位是：

```text
候选 mask proposal 模块；
错误解释模块；
交互式复核建议模块。
```

而不是：

```text
自动语义标签覆盖模块。
```

## 9. 已实现部分七：自动 refinement 验证及否定

Exp5-D 已经尝试过保守自动 refinement。

对应脚本是：

```text
exp5d_sam3_conservative_refinement.py
```

它尝试在严格条件下接受部分 SAM3 masks，例如：

```text
只接受 building / car；
要求 SAM3 score 达到阈值；
要求位于 candidate region；
要求满足 uncertainty gate；
限制 mask 面积；
只改写与 SegFormer 不同的位置。
```

结果为：

```text
accepted masks: 41 / 465
changed pixels: 6,420
improved pixels: 810
degraded pixels: 4,928
change precision: 0.1262
degradation rate: 0.7676
delta mIoU: -0.000005
```

这说明：

```text
即使非常保守，直接把 SAM3 mask 写回 SegFormer 语义图仍然不可靠。
```

所以当前系统已经验证并排除了一个方向：

```text
SegFormer prediction + SAM3 mask 自动覆盖 = 最终结果
```

这个方向不应作为后续主线。

## 10. 已实现部分八：SAM3 proposal 质量审计

Exp5-E 已经对 SAM3 proposals 做了质量分层审计。

对应脚本是：

```text
exp5e_sam3_proposal_quality_audit.py
```

总体上，SAM3 masks 的面积加权语义精度并不高：

```text
weighted GT precision = 0.1562
weighted error precision = 0.3282
weighted clutter FN precision = 0.0751
```

但质量分层后发现：

```text
object_proposal_high_precision:
  masks = 230
  eval_area = 75,607
  GT precision = 0.9928

error_region_explanation:
  masks = 44
  eval_area = 3,034,341
  error precision = 0.4126

large_surface_risky:
  masks = 14
  eval_area = 4,524,730
  GT precision = 0.1508
```

这说明 SAM3 的价值不是整体语义覆盖，而是分层使用：

```text
小面积高精度 masks:
  适合作为 object proposal / review proposal。

大面积低精度 masks:
  更适合作为 risk explanation，不适合自动改标签。

中等质量 masks:
  需要额外先验、语言条件或视觉语言匹配来判断。
```

这一点非常重要，因为它直接服务于后续自然语言推理分割：

```text
自然语言 query 最终需要的是候选目标 masks；
不是让 SAM3 改写整张语义图。
```

## 11. 已实现部分九：reasoning package 输出

Exp5-F 已经把前面模块组装成了一个阶段性系统输出包。

对应脚本是：

```text
exp5f_reasoning_segmentation_package.py
```

输出目录是：

```text
results_exp5f_reasoning_segmentation_package_heldout/
```

主要输出包括：

```text
final_predictions/
review_priority/
candidate_components/
overlays/
manifests/
metrics/
```

它的系统逻辑是：

```text
final prediction = SegFormer coarse prediction
review priority = candidate / entropy / low margin / boundary 综合风险图
SAM3 proposal audit = 写入 manifest 的对象候选与风险解释建议
```

总体结果为：

```text
candidate ratio = 0.3111
candidate error coverage = 0.8047

high priority ratio = 0.1115
high priority error coverage = 0.4969
```

这说明：

```text
系统已经能够输出稳定语义图 + 风险复核区域 + candidate components + SAM3 proposal 建议。
```

这已经是“遥感推理分割系统雏形”，但还不是最终的“自然语言 query 驱动目标分割系统”。

## 12. 部分实现但还不完整的部分

### 12.1 Low confidence 尚未进入 candidate region 公式

当前系统已经保存并分析了 confidence 图。

但是 Exp5-B 当前 candidate region 公式是：

```text
high_entropy_top20
OR low_margin_top20
OR boundary_band
```

还没有加入：

```text
low_confidence_topK
```

原因是 Exp5-A 里 high entropy、low confidence、low margin 三者覆盖效果非常接近，而第一版 candidate region 选择了 high entropy + low margin + boundary。后续如果要更完整，可以做一个消融：

```text
entropy only
margin only
confidence only
entropy + margin
entropy + margin + confidence
entropy + margin + boundary
entropy + margin + confidence + boundary
```

### 12.2 弱类别机制还不是通用模块

当前系统已经明确知道 `clutter/background` 是弱类别，并做了很多统计：

```text
clutter/background IoU 低；
clutter false negative 多；
主要错分方向是 clutter -> impervious / low vegetation / building。
```

Exp5-B 也把 clutter 纳入了 boundary classes。

但是，目前还没有一个通用的弱类别模块，例如：

```text
读取每个类别的 IoU / recall；
自动识别弱类别；
自动为弱类别生成额外 candidate regions；
自动为弱类别设计候选 prompts 或规则。
```

因此，弱类别处理目前是：

```text
已分析；
已部分用于 candidate boundary；
尚未系统化、通用化。
```

### 12.3 类别边界区域已实现，但还没有充分消融

边界区域已经实现了。

但还没有完整做下面的实验：

```text
只用 boundary band；
只用 uncertainty；
uncertainty + boundary；
不同 boundary radius；
不同 boundary class set；
边界区域是否更适合 SAM3 proposal。
```

Exp5-B 已经给出了第一版统计，但如果论文中要强调 boundary module，最好进一步做一个系统消融。

### 12.4 小目标机制还没有专门实现

目前系统中与小目标相关的证据主要来自 Exp5-E：

```text
大量 car-like 小面积 masks 的 GT precision 很高；
object_proposal_high_precision 层具有很高精度。
```

但系统还没有专门实现：

```text
小连通域检测；
小目标 candidate patch 生成；
车辆 / 漂浮物 / 小型设施的候选增强；
基于面积、长宽比、形状的 objectness score；
小目标专用 prompt 或 detector。
```

所以小目标目前是：

```text
有实验现象；
有 proposal 审计结果；
但还没有形成独立模块。
```

## 13. 尚未实现的部分

### 13.1 空间先验冲突检测尚未实现

目前系统没有真正接入：

```text
DSM
DEM
nDSM
NDVI
NDWI
河网
历史水体范围
水利工程边界
水位 / 降雨信息
```

因此，也还没有实现类似下面的判断：

```text
预测为 road，但与 water prior 高度重叠；
预测为 building，但 DSM 高度很低；
预测为 water，但位于高坡区域且远离河道；
候选 car mask 与 water mask 有明显重叠。
```

这部分是后续从 Potsdam 走向水利自然语言推理分割时必须补的模块。

### 13.2 自然语言 query 解析尚未实现

当前系统还不能接收用户输入：

```text
在水里的那辆红色的四座小汽车
```

并自动解析为：

```text
target = car
attributes = red, small/passenger car, four-seat
relation = inside or overlapping with water
```

也就是说，下面这一层还没有实现：

```text
自然语言 query
    ↓
target / attribute / relation
```

后续可以用两条路线实现：

```text
规则模板解析；
LLM 解析成结构化 JSON。
```

### 13.3 候选 mask 排序尚未实现

当前系统已经能生成 SAM3 candidate masks，也能做 proposal audit。

但还不能根据自然语言 query 对候选进行排序，例如：

```text
候选 1 是否是 car？
候选 1 是否是红色？
候选 1 是否在水里？
候选 1 是否比候选 2 更符合 query？
```

这一步需要引入：

```text
颜色/形状/面积等规则打分；
SegFormer scene prior；
视觉语言模型，例如 CLIP / SigLIP；
或多模态大模型。
```

### 13.4 最终目标 mask 选择尚未实现

当前系统的最终输出仍然是：

```text
SegFormer final semantic map
+ review priority
+ candidate components
+ proposal recommendations
```

而不是：

```text
用户 query 对应的唯一目标 mask
```

因此，最终自然语言推理分割的最后一步还没有完成：

```text
candidate masks
    ↓
candidate ranking
    ↓
best candidate selection
    ↓
final referred target mask
```

## 14. 当前系统已经达到的阶段

当前系统已经完成的是：

```text
SegFormer dense prediction
    ↓
uncertainty / margin / boundary risk localization
    ↓
candidate region generation
    ↓
SAM3 local proposal generation
    ↓
proposal audit
    ↓
reasoning package
```

这可以称为：

```text
遥感推理分割系统雏形
```

它已经证明：

```text
1. SegFormer 可以提供稳定语义底图；
2. 不确定性和边界可以定位大量错误风险区域；
3. candidate region 可以把搜索空间从整图压缩到约 31.11%；
4. SAM3 在 candidate patches 上能生成一批有意义的候选 masks；
5. SAM3 不适合直接自动改写 SegFormer 语义图；
6. SAM3 更适合作为 object proposal / risk explanation / review proposal；
7. reasoning package 比硬融合更稳健。
```

## 15. 当前系统还没有达到的阶段

当前系统还没有完成的是：

```text
自然语言 query
    ↓
语言解析
    ↓
候选 mask 语义匹配
    ↓
候选排序
    ↓
输出 query 对应的目标 mask
```

也就是说，它还不是完整的：

```text
Referring Expression Segmentation / Grounded Segmentation system
```

它目前更准确的状态是：

```text
SegFormer + SAM3 proposal + risk reasoning package
```

下一阶段 Exp6 应该补上的正是：

```text
query parser
candidate mask ranker
visual-language matching
spatial relation checking
final target mask selector
```

## 16. 建议下一步实现路线

建议不要继续优先做自动 refinement，而是进入 Exp6。

第一步，定义 query schema：

```json
{
  "target": "car",
  "attributes": {
    "color": "red",
    "type": "small passenger car"
  },
  "relations": [
    {
      "type": "inside_or_overlapping",
      "object": "water"
    }
  ]
}
```

第二步，给每个 SAM3 candidate mask 计算可解释特征：

```text
mask area
bbox size
aspect ratio
dominant color
mean RGB
overlap with SegFormer class regions
overlap with candidate region / high priority region
SAM3 score
```

第三步，先做规则版 candidate ranking：

```text
final_score(candidate)
= α * target_class_score
+ β * color_match_score
+ γ * spatial_relation_score
+ δ * mask_quality_score
+ ε * scene_prior_score
```

第四步，再接入视觉语言模型或多模态大模型：

```text
candidate crop + mask + query
    ↓
VLM / MLLM
    ↓
candidate-query matching score
```

第五步，输出最终目标 mask 和解释：

```text
selected_mask
candidate_ranking
match_reason
uncertainty
review_suggestion
```

## 17. 一句话总结

当前系统已经实现了：

```text
SegFormer 错误风险定位 + candidate region + SAM3 局部 proposal + proposal audit + reasoning package。
```

当前系统尚未实现：

```text
自然语言 query 驱动的候选 mask 排序与最终目标选择。
```

因此，下一步最关键的工作不是继续调 SAM3 自动改写，而是实现 Exp6：

```text
自然语言 query -> 结构化条件 -> candidate mask ranking -> referred target mask。
```
