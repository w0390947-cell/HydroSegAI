# Potsdam 多源数据利用方案

## 1. 当前数据使用情况

当前 Exp1 / Exp2-A / Exp2-B / Exp2-C 只使用了 Potsdam 数据集中的两类数据：

```text
Potsdam/2_Ortho_RGB/2_Ortho_RGB
Potsdam/5_Labels_all_noBoundary
```

其中：

| 数据 | 当前用途 |
|---|---|
| `2_Ortho_RGB/2_Ortho_RGB` | 作为 SAM3 输入影像 |
| `5_Labels_all_noBoundary` | 作为 ground truth 标签进行评估 |

这些实验目前属于：

```text
RGB-only zero-shot / prompt ablation baseline
```

它们适合用于验证通用开放词汇模型在遥感 RGB 输入下的直接迁移能力，但还没有充分利用 Potsdam 的多源遥感信息。

## 2. Potsdam 其他数据的潜在作用

从论文目标看，Potsdam 的其他数据不应该被简单地“全部塞进 SAM3”，而应该按推理分割体系中的不同模块角色进行利用。

| Potsdam 数据 | 适合承担的角色 |
|---|---|
| `2_Ortho_RGB` | 基础视觉输入，当前 Exp1/Exp2 已用 |
| `3_Ortho_IRRG` | 植被增强视觉输入，适合分析红外信息对 `tree` / `low vegetation` 的帮助 |
| `4_Ortho_RGBIR` | 多光谱输入，适合计算 NDVI 或训练多模态粗分割模型 |
| `1_DSM` | 高度先验，适合区分 `building` / `tree` / `road` / `low vegetation` |
| `5_Labels_all_noBoundary` | 主评估标签，忽略边界区域 |
| `5_Labels_all` | 可用于边界分析、训练粗分割模型、边界误差对比 |

这些数据可以分别服务于：

```text
多模态输入消融
光谱先验构造
高度先验构造
粗语义分割模型训练
边界误差分析
```

## 3. 第一阶段：多模态输入消融

第一阶段建议先不训练新模型，而是分析不同遥感影像输入对 SAM3 候选 mask 的影响。

可设计实验：

```text
Exp3-A: SAM3 + RGB
Exp3-B: SAM3 + IRRG
Exp3-C: SAM3 + RGBIR-derived RGB composite
Exp3-D: RGB + IRRG + RGBIR composite 多视图结果融合
```

该阶段关注的问题是：

```text
红外信息是否有助于 tree / low vegetation？
假彩色输入是否会破坏 SAM3 的通用视觉理解？
SAM3 对自然 RGB 和遥感假彩色输入的适应性是否不同？
多视图分别推理后融合，是否比单一视图更稳定？
```

这类实验的目标不一定是显著提高 mIoU，而是分析开放词汇分割模型面对遥感多模态数据时的适配问题。

该阶段可以支撑论文中的观点：

```text
通用视觉模型通常基于自然图像分布训练，
遥感假彩色、多光谱和俯视影像输入可能导致其候选 mask 生成行为发生变化。
```

### 3.1 Exp3-D：多视图结果融合

Exp3-D 不是把 RGB、IRRG 和 RGBIR 原样拼接成多通道输入给 SAM3。

原因是 SAM3 通常接收 3 通道图像输入，而 Potsdam 多源数据直接拼接后会超过 3 通道，直接输入需要修改模型视觉编码器，不符合当前“不改模型本体”的实验原则。

因此，Exp3-D 更合理的实现方式是：

```text
RGB 输入 SAM3 → masks_rgb
IRRG 输入 SAM3 → masks_irrg
RGBIR composite 输入 SAM3 → masks_rgbir
        ↓
多视图 mask / score 融合
        ↓
最终语义分割结果
```

也就是说，同一块 Potsdam 影像用不同遥感视图分别调用 SAM3，再在候选 mask 或最终类别 score 层面融合。

可采用的融合方式包括：

```text
平均融合：
final_score = (score_rgb + score_irrg + score_rgbir) / 3

加权融合：
final_score = α * score_rgb + β * score_irrg + γ * score_rgbir

类别相关加权：
植被类提高 IRRG / RGBIR composite 权重；
建筑、道路等类别保留较高 RGB 权重。
```

Exp3-D 想回答的问题是：

```text
通用开放词汇模型能否通过多视图分别推理的方式，间接利用遥感多源信息？
```

它在论文中的意义是：

```text
不修改 SAM3 模型结构的前提下，探索多源遥感输入参与推理分割的可行路径。
```

## 4. 第二阶段：将 DSM / RGBIR 转化为推理先验

DSM 和 RGBIR 不一定适合直接输入 SAM3，但非常适合作为推理分割体系中的先验约束。

### 4.1 DSM 高度先验

DSM 可用于构造高度相关规则。

例如：

| 类别 | 可能的高度先验 |
|---|---|
| `building` | 通常高度较高，区域较规则 |
| `tree` | 通常高度较高，但纹理和植被特征明显 |
| `impervious surface` | 通常高度较低 |
| `low vegetation` | 通常高度较低 |
| `car` | 小面积目标，可能有局部高度凸起 |

可用于规则：

```text
building / tree 候选 mask 如果 DSM 高度明显较高，可信度上升；
impervious surface / low vegetation 候选 mask 如果高度过高，可信度下降；
```

### 4.2 RGBIR / NDVI 光谱先验

RGBIR 可以用于计算植被指数。

典型形式为：

```text
NDVI = (NIR - R) / (NIR + R)
```

可用于规则：

```text
tree / low vegetation 候选 mask 如果 NDVI 较高，可信度上升；
tree / low vegetation 候选 mask 如果 NDVI 很低，可信度下降；
impervious surface / building 候选 mask 如果 NDVI 很高，可信度下降；
```

### 4.3 与 SAM3 mask score 融合

该阶段可以形成如下推理分割评分：

```text
final_score
= SAM3_mask_score
+ height_prior_score
+ vegetation_prior_score
+ area_or_shape_rule_score
```

也可以采用规则型筛选：

```text
SAM3 生成候选 mask；
DSM 判断高度合理性；
NDVI 判断植被可能性；
面积和形状规则过滤异常 mask；
最终进行类别融合。
```

这一阶段对应论文中的：

```text
候选 mask 生成 + 遥感物理先验约束
```

## 5. 第三阶段：训练粗语义分割模型并与 SAM3 融合

第三阶段对应论文中的实验五。

可以使用：

```text
RGB
RGBIR
DSM
nDSM
```

训练一个粗语义分割模型，例如：

```text
SegFormer
DeepLabV3+
U-Net
HRNet
Mask2Former
```

该模型输出 Potsdam 六类 coarse logits：

```text
impervious surface
building
low vegetation
tree
car
clutter/background
```

然后与 SAM3 结果融合：

```text
SegFormer 提供稳定类别先验；
SAM3 提供候选 mask 和边界细化；
DSM / NDVI 提供遥感物理先验；
融合模块输出最终语义图。
```

该阶段最符合最终论文中的推理分割体系，但实现成本也最高，因此建议放在 Exp2 和 Exp3/Exp4 之后。

## 6. 建议实验路线

建议后续实验路线如下：

```text
Exp1/Exp2:
RGB-only SAM3 prompt 推理基线与消融

Exp3:
IRRG / RGBIR 多模态输入消融

Exp4:
DSM / NDVI 遥感先验约束融合

Exp5:
粗语义分割模型训练与 SAM3 融合
```

这样，Potsdam 的多源数据不是零散使用，而是逐步服务于论文主线：

```text
从单一 RGB + prompt
走向多源遥感输入
再走向空间 / 光谱 / 高度先验
最终形成推理分割体系。
```

## 7. 与论文主题的关系

该多源数据利用路线服务于本文的核心目标：

```text
构建面向遥感与水利场景的通用推理分割模型技术体系。
```

Potsdam 数据集在论文中的作用不只是提供评估标签，而是提供一个标准遥感场景，用于逐步验证：

```text
通用开放词汇分割模型直接 zero-shot 的局限；
多模态遥感输入对候选 mask 的影响；
高度和光谱先验对推理分割的帮助；
粗语义分割模型与开放词汇候选 mask 的协同潜力。
```

最终，这些实验为水利场景中的推理分割应用提供方法基础。
