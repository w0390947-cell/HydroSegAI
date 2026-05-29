# 实验六：自然语言 Query 驱动候选 Mask 筛选实验设计说明

## 1. Exp6 的核心定位

Exp6 的目标不是继续提升 Potsdam 六类语义分割 mIoU，也不是继续证明 SAM3 能不能自动修正 SegFormer。

Exp6 要正式进入论文真正想做的任务：

```text
自然语言推理分割：
用户输入一句自然语言描述，系统从遥感图像中分割出最符合描述的目标。
```

例如：

```text
在水里的那辆红色的四座小汽车
靠近道路边缘的白色车辆
建筑旁边的小型车辆
植被区域附近的车辆
硬化地表上的独立车辆
```

在当前 Potsdam 阶段，我们暂时没有水利数据集，也没有真实水体类别。因此 Exp6 不应强行声称已经完成水利场景验证，而应先在 Potsdam 上验证自然语言推理分割框架中的关键机制：

```text
自然语言 query
    ↓
结构化条件解析
    ↓
候选 mask 集合
    ↓
候选 mask 特征计算
    ↓
候选排序
    ↓
输出最匹配 query 的目标 mask
```

## 2. Exp6 与前面实验的关系

Exp1-Exp3 证明：

```text
SAM3 不能直接 zero-shot 完成 Potsdam 六类密集语义分割；
prompt ensemble 有帮助，但仍不足；
IRRG / RGBIR 直接替代 RGB 输入 SAM3 没有收益。
```

Exp4 证明：

```text
SegFormer 可以提供稳定 dense semantic prior。
```

Exp5 证明：

```text
uncertainty / boundary 可以定位风险区域；
SAM3 在 candidate patches 上可以生成候选 masks；
SAM3 不适合自动覆盖 SegFormer；
SAM3 更适合作为 object proposal / review proposal / error explanation。
```

因此 Exp6 的合理起点是：

```text
不再让 SAM3 自动改语义图；
而是把 SAM3 masks 当成候选目标集合；
再根据自然语言 query 选择最匹配的 candidate mask。
```

## 3. Exp6 的总体流程

建议 Exp6 的系统流程为：

```text
输入：
  RGB image
  natural language query
  SegFormer prediction / probability / uncertainty
  Exp5 candidate regions
  SAM3 candidate masks

处理：
  1. query 解析为结构化条件；
  2. 为每个 SAM3 candidate mask 计算视觉、语义、空间特征；
  3. 根据 query 对 candidate masks 进行打分；
  4. 选择分数最高的 candidate；
  5. 输出目标 mask、候选排序和解释。

输出：
  selected target mask
  candidate ranking table
  query parse result
  reasoning explanation
  visualization overlay
```

## 4. Exp6-A：构建自然语言 Query 数据集

### 4.1 实验目的

Exp6-A 的目的不是训练模型，而是构造一个小规模、可控、可评价的 query benchmark。

它回答：

```text
我们能否在 Potsdam heldout 图像上构造一批自然语言目标描述，
并为每条 query 指定可评价的目标 mask 或目标区域？
```

### 4.2 第一阶段建议只做 car 类

建议 Exp6-A 第一版只围绕 `car` 做。

原因：

```text
1. Potsdam 中 car 是明确对象类，不像 impervious surface 那样是大面积区域；
2. Exp4 中 SegFormer car 预测较好，可作为场景 prior；
3. Exp5-E 显示 SAM3 的小面积 object proposals 精度较高，其中 car-like masks 有价值；
4. car 支持自然语言属性描述，例如颜色、大小、位置关系；
5. car 更接近用户说的“那辆小汽车”这种目标级 referring expression。
```

不建议一开始做：

```text
impervious surface
building
low vegetation
tree
clutter/background
```

因为这些类别更偏语义区域，不是天然的单目标 referring object。后续可以扩展到 building instance、道路区域或水利目标。

### 4.3 Query 类型设计

第一版 query 可以分为四类。

第一类：类别 query。

```text
这辆车
图中的小汽车
候选区域中的车辆
```

这一类最简单，只验证 candidate mask 是否能找到 car。

第二类：颜色属性 query。

```text
白色的小汽车
深色的小汽车
红色的小汽车
灰色的小汽车
```

Potsdam 俯视图中车辆颜色不一定总是清晰，因此颜色 query 需要谨慎，只选择肉眼可区分的样本。

第三类：空间关系 query。

```text
道路旁边的小汽车
建筑旁边的小汽车
植被附近的小汽车
停在硬化地表上的小汽车
靠近树木的小汽车
```

这些 query 可以利用 SegFormer scene prior 来判断 candidate mask 与 building、tree、low vegetation、impervious surface 的空间关系。

第四类：组合 query。

```text
建筑旁边的白色小汽车
硬化地表上的深色小汽车
植被附近的独立小汽车
道路边缘的小型车辆
```

这一类更接近真正的自然语言推理分割。

### 4.4 Query 标注格式

建议建立一个 JSONL 文件：

```text
exp6_queries_potsdam_car_dev.jsonl
```

每行一个 query：

```json
{
  "query_id": "q_000001",
  "image_id": "top_potsdam_2_13_RGB",
  "query_text": "建筑旁边的白色小汽车",
  "target_class": "car",
  "attributes": {
    "color": "white"
  },
  "relations": [
    {
      "type": "near",
      "object_class": "building"
    }
  ],
  "target_mask_source": "gt_connected_component",
  "target_component_id": 12,
  "notes": "目标车辆清晰可见，靠近建筑边缘"
}
```

第一版 target mask 可以来自 Potsdam GT 中的 car 连通域。

也就是说：

```text
从 heldout GT 中提取 car connected components；
人工选择其中可描述的车辆；
为每个车辆写一条或多条 query；
把该 connected component 作为 target mask。
```

这样可以让 Exp6 有明确评价标准。

注意：GT 只用于构造 benchmark 和评价，不用于真实推理打分。

## 5. Exp6-B：Query 解析模块

### 5.1 实验目的

Exp6-B 的目标是把自然语言 query 转成结构化条件。

例如：

```text
建筑旁边的白色小汽车
```

解析为：

```json
{
  "target": "car",
  "attributes": {
    "color": "white"
  },
  "relations": [
    {
      "type": "near",
      "object": "building"
    }
  ]
}
```

### 5.2 第一版建议使用规则解析

第一版不建议一上来就接复杂大模型。

建议先做规则版 parser：

```text
car keywords:
  car, vehicle, small vehicle, 小汽车, 车辆

color keywords:
  white, black, red, gray, blue, 白色, 黑色, 红色, 灰色

relation keywords:
  near, beside, next to, adjacent to, close to
  旁边, 附近, 靠近, 边缘

scene object keywords:
  building, road, impervious surface, tree, low vegetation
  建筑, 道路, 硬化地表, 树木, 植被
```

这样做的好处是：

```text
1. 可控；
2. 容易调试；
3. 不依赖外部 API；
4. 便于论文解释；
5. 后续可以替换为 LLM parser。
```

### 5.3 第二版可以接 LLM

当规则版跑通后，可以加入 LLM parser。

LLM 的输出必须是结构化 JSON，而不是自由文本：

```json
{
  "target": "car",
  "attributes": {
    "color": "red",
    "size": "small"
  },
  "relations": [
    {
      "type": "inside_or_overlapping",
      "object": "water"
    }
  ],
  "constraints": []
}
```

论文中可以把这一步表述为：

```text
LLM is used only as a language-side query parser,
not as a direct segmentation model.
```

这样可以避免“把分割问题全丢给大模型”的逻辑风险。

## 6. Exp6-C：Candidate Mask 特征计算与规则排序

### 6.1 实验目的

Exp6-C 的目标是：

```text
给定 query 和一组 SAM3 candidate masks，
能否用可解释规则选出最匹配的目标 mask？
```

第一版可以不接视觉语言模型，先做规则打分。

### 6.2 Candidate mask 来源

候选 mask 可以来自 Exp5-C / Exp5-E 的 SAM3 proposals。

建议优先使用 Exp5-E 审计后的高质量候选：

```text
object_proposal_high_precision
candidate_object_layer
car-related proposals
```

不要把所有大面积 mask 都纳入第一版 ranking，否则噪声太大。

候选集合可以定义为：

```text
candidate masks satisfying:
  mask area within car-like range
  bbox aspect ratio within reasonable range
  overlaps or is close to SegFormer car prediction
  not classified as large_surface_risky
```

### 6.3 每个 candidate mask 需要计算的特征

建议为每个 mask 计算：

```text
mask_area
bbox_width
bbox_height
bbox_aspect_ratio
centroid_x
centroid_y
SAM3_score
dominant_color
mean_RGB
color_histogram
overlap_with_SegFormer_car
overlap_with_impervious_surface
distance_to_building
distance_to_tree
distance_to_low_vegetation
distance_to_candidate_region
inside_high_priority_region_ratio
```

如果后续有水利数据，还可以加入：

```text
overlap_with_water
distance_to_river
DEM_mean
NDWI_mean
```

### 6.4 规则打分方式

第一版规则打分可以写成：

```text
final_score(candidate)
= w_class    * class_match_score
+ w_color    * color_match_score
+ w_relation * relation_match_score
+ w_shape    * shape_score
+ w_mask     * mask_quality_score
+ w_prior    * scene_prior_score
```

其中：

```text
class_match_score:
  candidate 与 SegFormer car 区域的 overlap；
  或 candidate 是否满足 car-like 面积 / 长宽比。

color_match_score:
  candidate 内部 RGB 是否符合 query color。

relation_match_score:
  candidate 到 building / tree / low vegetation / impervious surface 的距离或接触关系。

shape_score:
  面积、长宽比是否符合小汽车。

mask_quality_score:
  SAM3 score、mask 面积是否合理。

scene_prior_score:
  candidate 是否位于合理场景区域，例如 impervious surface。
```

### 6.5 第一版评价指标

Exp6-C 可以用这些指标评价：

```text
Top-1 accuracy:
  排名第一的 candidate 是否命中目标 GT component。

Top-3 recall:
  目标是否出现在前三个候选中。

Selected mask IoU:
  选中 candidate 与 target component 的 IoU。

Pointing accuracy:
  选中 candidate 的中心点是否落在目标 GT mask 内。

Candidate coverage:
  目标 GT component 是否被任一 SAM3 candidate 覆盖。
```

其中最重要的是：

```text
Top-1 accuracy
Top-3 recall
Selected mask IoU
```

## 7. Exp6-D：视觉语言模型候选排序

### 7.1 实验目的

Exp6-D 在规则排序基础上，引入视觉语言模型或多模态大模型进行 candidate-query matching。

它回答：

```text
视觉语言模型能否比规则排序更好地判断哪个 candidate mask 符合自然语言 query？
```

### 7.2 推荐输入形式

对每个 candidate mask，生成一个 crop：

```text
原图局部 crop；
candidate mask overlay；
candidate bbox；
可选：周边上下文区域。
```

然后输入：

```text
query text + candidate crop
```

模型输出：

```text
candidate-query matching score
```

### 7.3 可选模型路线

路线 A：CLIP / SigLIP。

```text
优点：
  本地可跑；
  适合批量候选排序；
  实验可复现。

缺点：
  对复杂空间关系理解较弱；
  对俯视遥感小目标可能不稳定。
```

路线 B：多模态大模型。

```text
优点：
  更擅长自然语言关系判断；
  可以输出解释。

缺点：
  成本高；
  结果稳定性和复现性需要控制；
  如果使用外部 API，实验环境和论文表述要谨慎。
```

路线 C：规则分数 + VLM 分数融合。

```text
final_score
= α * rule_score
+ β * vlm_score
```

这是最推荐的路线，因为它既保留可解释规则，又利用视觉语言模型处理复杂描述。

## 8. Exp6-E：自然语言推理分割输出包

Exp6-E 可以把 Exp6-A 到 D 组装成完整输出包。

输出目录建议为：

```text
results_exp6_referring_segmentation_potsdam_car/
```

目录结构：

```text
queries/
  exp6_queries_potsdam_car_dev.jsonl

parsed_queries/
  *_query_parse.json

candidate_features/
  *_candidate_features.csv

rankings/
  *_candidate_ranking.csv
  *_candidate_ranking.json

selected_masks/
  *_selected_mask.png
  *_selected_overlay.png

visualizations/
  *_query_result_panel.png

metrics/
  exp6_summary.json
  exp6_per_query.csv

manifests/
  *_reasoning_manifest.json
```

每条 query 的 manifest 可以包含：

```json
{
  "query_id": "q_000001",
  "query_text": "建筑旁边的白色小汽车",
  "parsed_query": {
    "target": "car",
    "attributes": {
      "color": "white"
    },
    "relations": [
      {
        "type": "near",
        "object": "building"
      }
    ]
  },
  "selected_candidate_id": "mask_000123",
  "scores": {
    "final_score": 0.82,
    "class_match_score": 0.91,
    "color_match_score": 0.77,
    "relation_match_score": 0.84,
    "mask_quality_score": 0.76
  },
  "explanation": [
    "candidate 与 SegFormer car 区域高度重叠",
    "candidate 区域颜色接近 white",
    "candidate 距离 building 边界较近",
    "candidate 面积和长宽比符合小汽车"
  ]
}
```

## 9. Exp6 的推荐分阶段安排

### Exp6-A：Query benchmark 构建

目标：

```text
从 Potsdam heldout 中构造 car referring expression benchmark。
```

产物：

```text
exp6_queries_potsdam_car_dev.jsonl
GT car connected components
query 可视化检查图
```

### Exp6-B：规则 query parser

目标：

```text
把自然语言 query 解析为 target / attribute / relation。
```

产物：

```text
parsed query JSON
parser coverage report
```

### Exp6-C：规则版 candidate ranking

目标：

```text
不用大模型，先用可解释规则从 SAM3 masks 中选择目标。
```

产物：

```text
candidate features
ranking table
selected masks
Top-1 / Top-3 / IoU metrics
```

### Exp6-D：VLM / MLLM 候选排序

目标：

```text
引入视觉语言模型，提升复杂 query 的候选匹配能力。
```

产物：

```text
VLM scores
rule-only vs VLM-only vs rule+VLM 对比
```

### Exp6-E：自然语言推理分割输出包

目标：

```text
形成完整的 query -> selected mask -> explanation 输出体系。
```

产物：

```text
selected target masks
query reasoning manifests
visual result panels
summary metrics
```

## 10. 第一版不要做得太复杂

第一版 Exp6 应该克制。

建议先做：

```text
Potsdam heldout
car 类
20-50 条人工构造 query
规则 parser
规则 candidate ranking
使用 Exp5 的 SAM3 proposals
```

暂时不要一开始就做：

```text
所有类别；
复杂长文本；
多轮对话；
真实水利场景；
自动生成大量 query；
直接接外部大模型做全流程；
追求极高 mIoU。
```

因为 Exp6 的第一目标是验证：

```text
自然语言条件能否驱动候选 mask 筛选。
```

不是一次性完成最终产品。

## 11. Exp6 成功的判断标准

Exp6 如果满足以下条件，就说明路线成立：

```text
1. 系统能解析 query 中的 target / attribute / relation；
2. SAM3 candidate masks 中确实包含一部分目标对象；
3. 规则或 VLM ranking 能把目标 candidate 排到前列；
4. 输出的 selected mask 与目标 GT component 有明显重叠；
5. manifest 能解释为什么选择这个 candidate；
6. 失败案例可以被归因，例如候选缺失、颜色不可见、关系判断错误。
```

具体指标可以设为：

```text
Top-1 accuracy
Top-3 recall
mean selected IoU
candidate coverage
parser success rate
failure reason distribution
```

## 12. Exp6 可能失败的地方

Exp6 可能遇到几个问题。

第一，SAM3 candidate 中没有目标。

这种情况下 ranking 再好也没用。需要统计：

```text
candidate coverage
```

第二，目标太小，遥感俯视图无法判断颜色或细节。

例如“四座”这种属性，在 Potsdam 俯视图中几乎不可见。第一版 query 不应过多依赖这种不可观测属性。

第三，Potsdam 没有 water 类。

所以“在水里”这种 query 不能在 Potsdam 上真实验证。它只能作为未来水利场景目标。Potsdam 阶段可以用：

```text
near building
near road / impervious surface
near tree
near vegetation
```

第四，SegFormer prior 与自然语言概念不完全一致。

例如 query 说“道路旁边”，但 Potsdam 类别中道路通常属于 `impervious surface`，不是单独 road 类。这需要在 parser 中建立概念映射：

```text
road / pavement / hard surface -> impervious surface
```

第五，VLM 可能不适应遥感俯视小目标。

所以 VLM 不能一开始作为唯一判断，应与规则特征结合。

## 13. Exp6 与最终水利目标的关系

Exp6 在 Potsdam 上做 car referring segmentation，并不是最终水利应用。

它的作用是验证通用机制：

```text
natural language query
    ↓
candidate proposal
    ↓
scene prior / spatial relation
    ↓
candidate ranking
    ↓
selected mask
```

迁移到水利场景时，可以替换：

```text
Potsdam car query
```

为：

```text
水中的漂浮物
河道中的船只
被淹没的道路
靠近堤防的施工设施
岸线附近的闸门或建筑
```

同时替换 coarse prior：

```text
Potsdam SegFormer prior
```

为：

```text
water / land / vegetation / building / road / flood prior
```

再加入：

```text
DEM
NDWI
河网
历史水体
水利工程边界
```

这样 Exp6 就能成为最终水利自然语言推理分割体系的技术验证桥梁。

## 14. 建议优先实现的代码文件

建议按下面顺序写代码：

```text
exp6a_build_referring_query_dataset.py
exp6b_parse_referring_queries.py
exp6c_rank_candidate_masks_rules.py
exp6d_rank_candidate_masks_vlm.py
exp6e_referring_segmentation_package.py
```

对应配置：

```text
exp6a_build_referring_query_dataset.yaml
exp6b_parse_referring_queries.yaml
exp6c_rank_candidate_masks_rules.yaml
exp6d_rank_candidate_masks_vlm.yaml
exp6e_referring_segmentation_package.yaml
```

第一版可以先只实现：

```text
Exp6-A + Exp6-B + Exp6-C
```

也就是：

```text
构建 query benchmark；
规则解析；
规则排序候选 masks。
```

等这条链路跑通，再考虑 Exp6-D 的 VLM。

## 15. 一句话总结

Exp6 应该从“SegFormer + SAM3 是否能自动提升 mIoU”转向：

```text
自然语言 query 能否驱动 SAM3 candidate masks 的选择。
```

第一版最合理的实验是：

```text
在 Potsdam heldout 上，以 car 为目标对象，
构建少量 referring expression queries，
利用 Exp5 生成的 SAM3 proposals，
通过规则解析和可解释打分选出目标 mask。
```

如果这一步成立，论文主线就会从普通遥感语义分割自然过渡到：

```text
面向遥感与水利场景的自然语言推理分割框架。
```
