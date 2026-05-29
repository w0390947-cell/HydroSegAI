# 实验三：SAM3 + Potsdam 多源遥感输入模态实验说明

## 1. 实验定位

Exp3 系列实验用于分析多源遥感输入模态对 SAM3 开放词汇推理分割结果的影响。

根据 Exp1 与 Exp2 的已有结果，实验三不应继续把变量放在 prompt 或简单 mask 后处理上，而应回答：

```text
在固定当前最有效的 prompt 策略后，
RGB、IRRG、RGBIR 派生三通道输入以及多视图融合，
是否能够改善 SAM3 在 Potsdam 遥感场景中的候选 mask 生成和最终语义分割结果？
```

已有实验结论如下：

| 实验 | 关键变量 | OA | mIoU | Mean F1 | 结论 |
|---|---|---:|---:|---:|---|
| Exp1 | 六类官方 prompt | 0.0892 | 0.0578 | 0.1020 | 官方遥感类别名直接迁移很弱 |
| Exp2-A | 去掉主动背景 prompt | 0.0892 | 0.0578 | 0.1020 | 不主动 prompt 背景本身不产生变化 |
| Exp2-B | 前景视觉 prompt ensemble | 0.1659 | 0.1052 | 0.1806 | 明显提升，是当前最有效策略 |
| Exp2-C | Exp2-B + 固定面积过滤 | 0.1518 | 0.1013 | 0.1753 | 相对 Exp2-B 略降，固定面积过滤不是增益项 |

因此，Exp3 的推荐设计是：

```text
固定 Exp2-B 的视觉 prompt ensemble 策略；
不启用 Exp2-C 的 class-aware area filtering；
采用 24 张 development split 分析模态敏感性；
采用 14 张 held-out split 做最终评估；
只改变输入模态或多视图融合方式。
```

这样做的原因是：Exp2-C 已经证明固定面积过滤会误删部分有用的大面积前景 mask，尤其是不透水表面。因此如果 Exp3 继续继承 Exp2-C，就会把一个已知负收益的后处理变量混入“输入模态”实验，导致结果解释不干净。

## 2. Exp3 核心假设

Exp1/Exp2 暴露出的主要问题不是 precision 不够，而是前景召回严重不足，大量真实前景被回退为 `clutter/background`。

特别是：

```text
Exp2-B 之后，预测为 clutter/background 的像素仍占 69.38%，
而真实 clutter/background 只占 4.65%。
```

类别层面仍存在明显短板：

| 类别 | Exp2-B 现象 | Exp3 关注点 |
|---|---|---|
| impervious surface | 有提升，但仍大量回退背景 | IRRG/RGBIR 是否改变硬化地表与建筑/植被的混淆 |
| building | precision 高但 recall 很低 | 近红外或多视图是否能增加建筑候选覆盖 |
| low vegetation | prompt ensemble 提升最明显 | NIR/NDVI 信息是否进一步提升草地、低矮植被召回 |
| tree | 几乎没有有效召回 | IRRG/RGBIR 是否能让树冠更容易被 SAM3 发现 |
| car | 小目标 precision 高，recall 中等 | 多视图融合是否增加车辆漏检或误检 |
| clutter/background | 严重过预测 | 多模态输入是否减少前景像素回退背景 |

因此，Exp3 不只是看总体 mIoU 是否提升，还要重点看：

```text
前景预测占比是否增加；
真实前景误分为 clutter/background 的比例是否下降；
vegetation/tree 是否从近红外信息中获益；
多视图融合是否提升 recall，还是只引入更多噪声。
```

## 3. 类别选择性模态分析目标

Exp3-B 和 Exp3-C 不应只被理解为“IRRG / RGBIR 是否整体优于 RGB”的单一比较。更重要的问题是：

```text
不同输入模态是否对不同 Potsdam 类别具有类别选择性优势？
```

也就是说，某个模态即使整体 mIoU 没有超过 RGB，也可能在特定类别上更有价值。例如近红外相关输入可能更利于植被类候选 mask 生成，而 RGB 可能仍更适合建筑、车辆等更接近自然图像预训练分布的类别。

推荐按以下思路解释 Exp3-B / Exp3-C：

| 类别 | 需要观察的模态偏好 | 分析理由 |
|---|---|---|
| `impervious surface` | RGB、IRRG、RGBIR 哪个更能覆盖道路/硬化地表 | 硬化地表面积大，容易与建筑、背景和裸地混淆 |
| `building` | RGB 是否仍优于近红外输入 | roof/building 更接近通用视觉概念，RGB 可能更符合 SAM3 预训练分布 |
| `low vegetation` | IRRG / RGBIR 是否优于 RGB | 低矮植被具有明显近红外响应 |
| `tree` | IRRG / RGBIR 是否提高召回 | 树冠在 RGB 下几乎没有召回，近红外可能增强可分性 |
| `car` | RGB 是否优于 IRRG / RGBIR | 车辆是小目标，颜色和外观形状更依赖 RGB |
| `clutter/background` | 是否继续只作为 fallback | 不主动 prompt 背景，重点观察其他模态是否减少前景回退背景 |

因此，Exp3 的分析表格除了报告总体指标，还应报告每个类别在不同模态下的：

```text
IoU
F1
Precision
Recall
预测像素占比
真实类别误分为 clutter/background 的比例
```

如果 Exp3-B / Exp3-C 显示出清晰的类别级互补性，则后续可以设计：

```text
按类别选择输入模态；
或按类别设置多视图融合权重；
或构建 class-aware modality fusion / class-wise routing。
```

但需要注意论文严谨性：不能直接用测试集结果挑选每个类别的最优模态，然后再在同一测试集上宣称该选择策略有效。若要把类别级模态选择作为正式方法，应优先使用遥感物理先验预先指定类别-模态映射，或在独立验证集上确定映射，再在测试集上评估。

## 4. 数据划分与评估设置

Exp3 仍使用 Potsdam 数据集和 noBoundary 标签，评估类别与 Exp1/Exp2 保持一致。与 Exp1/Exp2 的 38 张 full-reference 消融分析不同，Exp3 为避免根据完整测试结果选择模态规则，采用两阶段划分：

```text
development split：24 张官方公开给参赛者的标签瓦片，用于模态敏感性分析和规则确定；
held-out split：剩余 14 张完整参考标签瓦片，用于最终评估。
```

| 项目 | 内容 |
|---|---|
| development 标签 | `Potsdam/5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary` |
| held-out 标签 | `Potsdam/5_Labels_all_noBoundary` |
| 标签语义 | Potsdam 标准 6 类 |
| ignore 区域 | noBoundary 标签中的黑色 `(0,0,0)`，映射为 `255` |
| development split | 官方公开 24 张 participant-labeled tiles |
| held-out split | 其余 14 张 fully referenced tiles |
| patch size | `1008` |
| stride | `672` |
| score threshold | `0.5` |
| patch 方式 | 边缘对齐滑窗 |
| patch 融合 | SAM3 置信度加权融合 |

development split 包含：

```text
top_potsdam_2_10, top_potsdam_2_11, top_potsdam_2_12
top_potsdam_3_10, top_potsdam_3_11, top_potsdam_3_12
top_potsdam_4_10, top_potsdam_4_11, top_potsdam_4_12
top_potsdam_5_10, top_potsdam_5_11, top_potsdam_5_12
top_potsdam_6_7, top_potsdam_6_8, top_potsdam_6_9
top_potsdam_6_10, top_potsdam_6_11, top_potsdam_6_12
top_potsdam_7_7, top_potsdam_7_8, top_potsdam_7_9
top_potsdam_7_10, top_potsdam_7_11, top_potsdam_7_12
```

held-out split 包含：

```text
top_potsdam_2_13, top_potsdam_2_14
top_potsdam_3_13, top_potsdam_3_14
top_potsdam_4_13, top_potsdam_4_14, top_potsdam_4_15
top_potsdam_5_13, top_potsdam_5_14, top_potsdam_5_15
top_potsdam_6_13, top_potsdam_6_14, top_potsdam_6_15
top_potsdam_7_13
```

Exp3 配置通过 `evaluation.split` 控制划分：

```yaml
paths:
  label_subdir: "5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary"

evaluation:
  split: "dev"      # 24 张 development split
  test_images: []   # 留空时由 split 自动决定样本
```

held-out 配置使用完整参考 noBoundary 标签目录：

```yaml
paths:
  label_subdir: "5_Labels_all_noBoundary"

evaluation:
  split: "heldout"  # 14 张 held-out split
  test_images: []
```

可选值：

| split | 含义 |
|---|---|
| `dev` | 24 张 official participant-labeled tiles，用于模态分析，使用 participant noBoundary 标签目录 |
| `heldout` | 14 张 held-out fully referenced tiles，用于最终评估，使用 complete reference noBoundary 标签目录 |
| `all` | 自动发现所有可配对样本，仅用于补充 full-reference 分析 |

类别定义：

| 类别 ID | 类别名 | Potsdam 颜色 |
|---:|---|---|
| 0 | `impervious surface` | 白色 `(255,255,255)` |
| 1 | `building` | 蓝色 `(0,0,255)` |
| 2 | `low vegetation` | 青色 `(0,255,255)` |
| 3 | `tree` | 绿色 `(0,255,0)` |
| 4 | `car` | 黄色 `(255,255,0)` |
| 5 | `clutter/background` | 红色 `(255,0,0)` |
| 255 | `ignore` | 黑色 `(0,0,0)` |

如需调试小样本，可显式填写 `evaluation.test_images`；此时显式列表优先于 `evaluation.split`。

## 5. 固定推理策略

Exp3-A/B/C/D 固定使用 Exp2-B 的 prompt 策略：

```text
不主动 prompt clutter/background
五个前景类别使用 visual prompt ensemble
不启用 class-aware mask area filtering
未被有效前景 mask 覆盖的像素回退为 clutter/background
```

主动 prompt 类别：

```text
0 impervious surface
1 building
2 low vegetation
3 tree
4 car
```

不主动 prompt 类别：

```text
5 clutter/background
```

但 `clutter/background` 仍然保留为：

```text
评估类别
fallback 类别
```

当前每类前景使用 5 个 prompt variants：

| 类别 | Prompt variants |
|---|---|
| `impervious surface` | `impervious surface`; `road and pavement`; `asphalt road`; `paved area`; `parking lot` |
| `building` | `building`; `buildings`; `rooftop`; `building roof`; `house roof` |
| `low vegetation` | `low vegetation`; `grass`; `lawn`; `low plants`; `ground vegetation` |
| `tree` | `tree`; `trees`; `tree crown`; `tall tree`; `urban trees` |
| `car` | `car`; `cars`; `vehicle`; `small vehicle`; `parked car` |

每个 patch 的 prompt 调用数为：

```text
5 类 x 每类 5 个 prompt = 25 次
```

Exp3 配置中保留 `mask_filter` 字段，是为了复用 Exp2-C 的 evaluator 代码和保留后续扩展入口；但默认必须关闭：

```yaml
mask_filter:
  enabled: false
```

论文中应将 Exp3 表述为：

```text
fixed Exp2-B visual prompt ensemble strategy
```

不应表述为继承 Exp2-C，因为 Exp2-C 的固定面积过滤不是当前最优策略。

## 6. Exp3-A：RGB Modality

### 6.1 实验目的

Exp3-A 使用 Potsdam RGB 正射影像作为输入，是 Exp3 系列中的单模态 RGB 参照实验。

它回答的问题是：

```text
在固定 Exp2-B prompt ensemble 后，RGB 输入下的结果表现如何？
```

Exp3-A 也用于确认：关闭面积过滤后，RGB 模态结果应与 Exp2-B 在同口径下基本一致。若结果不同，应优先检查配置、代码版本和输出目录。

### 6.2 实验变量

| 项目 | 内容 |
|---|---|
| 输入影像 | `Potsdam/2_Ortho_RGB/2_Ortho_RGB` |
| 文件模式 | `{image_id}_RGB.tif` |
| 输入模态 | RGB |
| 固定策略 | Exp2-B visual prompt ensemble |
| mask filter | disabled |

### 6.3 对应文件

| 类型 | 路径 |
|---|---|
| 脚本 | `exp3a_sam3_potsdam_rgb_modality.py` |
| dev 配置 | `exp3a_sam3_potsdam_rgb_modality.yaml` |
| held-out 配置 | `exp3a_sam3_potsdam_rgb_modality_heldout.yaml` |
| dev 结果目录 | `results_exp3a_sam3_potsdam_rgb_modality_dev/` |
| held-out 结果目录 | `results_exp3a_sam3_potsdam_rgb_modality_heldout/` |

### 6.4 运行命令

正式运行：

```bash
.venv_hf/bin/python exp3a_sam3_potsdam_rgb_modality.py
```

held-out 最终评估：

```bash
.venv_hf/bin/python exp3a_sam3_potsdam_rgb_modality.py \
  --config exp3a_sam3_potsdam_rgb_modality_heldout.yaml
```

如需显式指定配置或输出目录：

```bash
.venv_hf/bin/python exp3a_sam3_potsdam_rgb_modality.py \
  --config exp3a_sam3_potsdam_rgb_modality.yaml \
  --output-dir /home/anjou/PythonENV/Test_11/results_exp3a_sam3_potsdam_rgb_modality_dev
```

## 7. Exp3-B：IRRG Modality

### 7.1 实验目的

Exp3-B 使用 Potsdam IRRG 影像作为输入，用于分析近红外信息进入三通道假彩色组合后，对不同 Potsdam 类别的候选 mask 生成是否存在选择性优势。

它回答的问题是：

```text
在固定 Exp2-B prompt ensemble 后，
IRRG 输入是否在某些类别上比 RGB 更适合 SAM3 的开放词汇候选 mask 生成？
```

因此，Exp3-B 不应只看整体 mIoU 是否超过 Exp3-A。即使整体指标没有提升，只要 `low vegetation` 或 `tree` 的 IoU / recall 明显改善，IRRG 仍然说明近红外假彩色输入具有类别级价值。

重点观察：

```text
low vegetation 和 tree 的 IoU / recall 是否提升；
impervious surface 与 vegetation 的混淆是否变化；
clutter/background 过预测是否缓解。
```

### 7.2 实验变量

| 项目 | 内容 |
|---|---|
| 输入影像 | `Potsdam/3_Ortho_IRRG/3_Ortho_IRRG` |
| 文件模式 | `{image_id}_IRRG.tif` |
| 输入模态 | IRRG |
| 固定策略 | Exp2-B visual prompt ensemble |
| mask filter | disabled |

### 7.3 对应文件

| 类型 | 路径 |
|---|---|
| 脚本 | `exp3b_sam3_potsdam_irrg_modality.py` |
| dev 配置 | `exp3b_sam3_potsdam_irrg_modality.yaml` |
| held-out 配置 | `exp3b_sam3_potsdam_irrg_modality_heldout.yaml` |
| dev 结果目录 | `results_exp3b_sam3_potsdam_irrg_modality_dev/` |
| held-out 结果目录 | `results_exp3b_sam3_potsdam_irrg_modality_heldout/` |

### 7.4 运行命令

正式运行：

```bash
.venv_hf/bin/python exp3b_sam3_potsdam_irrg_modality.py
```

held-out 最终评估：

```bash
.venv_hf/bin/python exp3b_sam3_potsdam_irrg_modality.py \
  --config exp3b_sam3_potsdam_irrg_modality_heldout.yaml
```

如需显式指定配置或输出目录：

```bash
.venv_hf/bin/python exp3b_sam3_potsdam_irrg_modality.py \
  --config exp3b_sam3_potsdam_irrg_modality.yaml \
  --output-dir /home/anjou/PythonENV/Test_11/results_exp3b_sam3_potsdam_irrg_modality_dev
```

## 8. Exp3-C：RGBIR Composite

### 8.1 实验目的

Exp3-C 使用 Potsdam RGBIR 四波段影像，并在脚本中转换为 SAM3 可接受的三通道 composite。

当前配置采用：

```yaml
input_modality:
  name: "rgbir_composite"
  rgbir_composite_mode: "nir-r-g"
```

也就是将 RGBIR 中的：

```text
NIR, R, G
```

组合成三通道输入。

该实验回答的问题是：

```text
从 RGBIR 中构造包含近红外信息的三通道输入，
是否能在特定类别上改善 SAM3 在 Potsdam 上的候选 mask 和语义结果？
```

Exp3-C 的重点不是证明 `nir-r-g` composite 在整体指标上一定优于 RGB，而是判断 RGBIR 派生输入是否能为某些类别提供更好的视觉表达。特别需要关注：

```text
low vegetation / tree 是否获得更高召回；
impervious surface 是否因 NIR 通道引入而更容易与植被区分；
building 和 car 是否因偏离自然 RGB 分布而下降；
clutter/background 过预测是否减少。
```

### 8.2 支持的 composite mode

脚本当前支持以下 RGBIR composite：

| composite mode | 通道组合 |
|---|---|
| `rgb` | R, G, B |
| `nir-r-g` | NIR, R, G |
| `nir-g-b` | NIR, G, B |
| `r-g-ndvi` | R, G, NDVI |
| `ndvi-r-g` | NDVI, R, G |

当前 Exp3-C 固定使用：

```text
nir-r-g
```

不建议在同一个 Exp3-C 主实验中同时尝试多个 composite mode，否则会把 Exp3-C 从“输入模态实验”变成“通道组合调参”。如果需要比较多个 RGBIR composite，应作为 Exp3-C 的附加子实验单独报告。

### 8.3 实验变量

| 项目 | 内容 |
|---|---|
| 输入影像 | `Potsdam/4_Ortho_RGBIR/4_Ortho_RGBIR` |
| 文件模式 | `{image_id}_RGBIR.tif` |
| 输入模态 | RGBIR-derived 3-channel composite |
| composite mode | `nir-r-g` |
| 固定策略 | Exp2-B visual prompt ensemble |
| mask filter | disabled |

### 8.4 对应文件

| 类型 | 路径 |
|---|---|
| 脚本 | `exp3c_sam3_potsdam_rgbir_composite.py` |
| dev 配置 | `exp3c_sam3_potsdam_rgbir_composite.yaml` |
| held-out 配置 | `exp3c_sam3_potsdam_rgbir_composite_heldout.yaml` |
| dev 结果目录 | `results_exp3c_sam3_potsdam_rgbir_composite_dev/` |
| held-out 结果目录 | `results_exp3c_sam3_potsdam_rgbir_composite_heldout/` |

### 8.5 运行命令

正式运行：

```bash
.venv_hf/bin/python exp3c_sam3_potsdam_rgbir_composite.py
```

held-out 最终评估：

```bash
.venv_hf/bin/python exp3c_sam3_potsdam_rgbir_composite.py \
  --config exp3c_sam3_potsdam_rgbir_composite_heldout.yaml
```

如需显式指定配置或输出目录：

```bash
.venv_hf/bin/python exp3c_sam3_potsdam_rgbir_composite.py \
  --config exp3c_sam3_potsdam_rgbir_composite.yaml \
  --output-dir /home/anjou/PythonENV/Test_11/results_exp3c_sam3_potsdam_rgbir_composite_dev
```

## 9. Exp3-D：Multi-view Fusion

### 9.1 实验目的

Exp3-D 不只选择一种输入模态，而是分别在多个视图上运行 SAM3，再融合不同视图的 patch 置信度结果。

它回答的问题是：

```text
RGB、IRRG 和 RGBIR composite 是否能提供互补候选 mask，
多视图 score fusion 是否比任一单模态更稳定？
```

Exp3-D 应放在 Exp3-A/B/C 之后解释。只有当单模态结果显示不同模态存在类别互补时，多视图融合的结果才有充分动机。

建议将 Exp3-D 分成两个层次理解：

| 层次 | 名称 | 作用 |
|---|---|---|
| Exp3-D1 | naive multi-view equal-weight fusion | RGB、IRRG、RGBIR 三视图等权融合，检验简单多视图投票是否稳定 |
| Exp3-D2 | class-aware modality fusion / class-wise routing | 根据类别选择更合适的模态或类别级权重，检验类别选择性模态融合是否有效 |

当前脚本和配置对应的是 Exp3-D1：

```text
RGB + IRRG + RGBIR composite 等权 score fusion
```

Exp3-D2 暂时建议作为后续设计或 Exp4/扩展实验，不建议在没有独立验证依据的情况下直接用测试集最优类别结果来构造融合规则。

### 9.2 当前视图配置

当前 Exp3-D 使用三个视图：

| 视图名 | 类型 | 输入路径 | 文件模式 | 权重 |
|---|---|---|---|---:|
| `rgb` | RGB | `2_Ortho_RGB/2_Ortho_RGB` | `{image_id}_RGB.tif` | 1.0 |
| `irrg` | IRRG | `3_Ortho_IRRG/3_Ortho_IRRG` | `{image_id}_IRRG.tif` | 1.0 |
| `rgbir_nir_r_g` | RGBIR composite | `4_Ortho_RGBIR/4_Ortho_RGBIR` | `{image_id}_RGBIR.tif` | 1.0 |

RGBIR 视图的 composite mode 为：

```text
nir-r-g
```

### 9.3 融合逻辑

Exp3-D 的处理流程为：

```text
读取标签
读取每个视图对应的影像
检查每个视图与标签尺寸一致
分别对每个视图进行 patch 切片
每个 patch 使用同一套 Exp2-B prompt ensemble
不启用类别面积过滤
先在 patch 内进行多类别 confidence fusion
再将该视图 patch 的 confidence_map 乘以 view weight
将所有视图的 patch 结果放入统一 merge_patches()
按类别累计重叠 patch / 多视图置信度分数
最终选择累计分数最高的类别
```

当前配置中三个视图权重均为 `1.0`，因此多视图融合等价于：

```text
RGB、IRRG、RGBIR composite 三个视图的有效 mask 置信度共同投票。
```

如果某个样本缺失任一视图影像，Exp3-D 会将该样本记为 skipped，不纳入总体指标。

### 9.4 计算量

在单视图每张约 81 个 patch、每 patch 25 个 prompt 调用的设置下：

```text
Exp3-A/B/C 单模态 dev split：
总 patch 数约 = 24 x 81 = 1944
总 prompt 调用数约 = 1944 x 25 = 48600

Exp3-A/B/C 单模态 held-out split：
总 patch 数约 = 14 x 81 = 1134
总 prompt 调用数约 = 1134 x 25 = 28350

Exp3-D 三视图 dev split：
总 patch 数约 = 24 x 81 x 3 = 5832
总 prompt 调用数约 = 5832 x 25 = 145800

Exp3-D 三视图 held-out split：
总 patch 数约 = 14 x 81 x 3 = 3402
总 prompt 调用数约 = 3402 x 25 = 85050
```

Exp3-D 的计算量约为单模态实验的 3 倍。

### 9.5 对应文件

| 类型 | 路径 |
|---|---|
| 脚本 | `exp3d_sam3_potsdam_multiview_fusion.py` |
| dev 配置 | `exp3d_sam3_potsdam_multiview_fusion.yaml` |
| held-out 配置 | `exp3d_sam3_potsdam_multiview_fusion_heldout.yaml` |
| dev 结果目录 | `results_exp3d_sam3_potsdam_multiview_fusion_dev/` |
| held-out 结果目录 | `results_exp3d_sam3_potsdam_multiview_fusion_heldout/` |

### 9.6 运行命令

正式运行：

```bash
.venv_hf/bin/python exp3d_sam3_potsdam_multiview_fusion.py
```

held-out 最终评估：

```bash
.venv_hf/bin/python exp3d_sam3_potsdam_multiview_fusion.py \
  --config exp3d_sam3_potsdam_multiview_fusion_heldout.yaml
```

如需显式指定配置或输出目录：

```bash
.venv_hf/bin/python exp3d_sam3_potsdam_multiview_fusion.py \
  --config exp3d_sam3_potsdam_multiview_fusion.yaml \
  --output-dir /home/anjou/PythonENV/Test_11/results_exp3d_sam3_potsdam_multiview_fusion_dev
```

## 10. Exp3 系列对比关系

| 实验 | 输入模态 | 多视图融合 | 固定策略 | 主要问题 |
|---|---|---|---|---|
| Exp3-A | RGB | 否 | Exp2-B | RGB 输入下的固定策略复现实验和模态基准 |
| Exp3-B | IRRG | 否 | Exp2-B | 近红外假彩色输入是否对植被、树木等类别有选择性优势 |
| Exp3-C | RGBIR composite (`nir-r-g`) | 否 | Exp2-B | RGBIR 派生三通道输入是否对特定类别有选择性优势 |
| Exp3-D1 | RGB + IRRG + RGBIR composite | 是，等权 | Exp2-B | 简单多源视图是否提供互补候选 mask |
| Exp3-D2 | 类别级模态选择或类别级权重 | 是，类别相关 | Exp2-B | 后续设计：按类别利用最合适的输入模态 |

Exp3 分析时应重点比较：

```text
不同输入模态下各类别 IoU 是否发生系统性变化；
vegetation / tree 类是否从 NIR 信息中受益；
building / impervious surface 是否在 IRRG 或 RGBIR composite 下出现混淆变化；
car 类是否因模态变化而更容易被漏检或误检；
多视图融合是否提高 dataset_mIoU、FWIoU 或主要类别 IoU；
多视图融合是否引入更多误检，导致 precision 下降；
预测为 clutter/background 的像素占比是否下降。
```

Exp3-B / Exp3-C 的结果分析应避免只写：

```text
IRRG/RGBIR 是否超过 RGB。
```

更推荐写成：

```text
IRRG/RGBIR 在哪些类别上优于 RGB；
哪些类别仍然更适合 RGB；
这些类别级差异是否足以支持后续 class-aware modality fusion。
```

## 11. 输出文件结构

每个 Exp3 实验会保存：

```text
results_exp3*/predictions/
results_exp3*/visualizations/
results_exp3*/metrics/
results_exp3*/logs/
```

主要结果文件包括：

```text
metrics/overall_metrics.json
metrics/per_image_metrics.csv
metrics/experiment_metadata.json
metrics/config_snapshot.yaml
metrics/run_context.json
```

其中：

| 文件 | 内容 |
|---|---|
| `overall_metrics.json` | 数据集级指标、逐图平均指标、模型与配置元数据、run context、Exp3 metadata |
| `per_image_metrics.csv` | 每张图的 OA、mIoU、F1、precision、recall、FWIoU |
| `experiment_metadata.json` | 输入模态、prompt 策略、mask filter 是否启用、多视图配置等 |
| `config_snapshot.yaml` | 本次运行使用的 YAML 配置快照 |
| `run_context.json` | 期望样本、成功样本、跳过样本、失败样本和指标统计范围 |

Exp3 metadata 至少包含：

```text
experiment_name
variable_under_test
input_modality
rgbir_composite_mode
prompt_class_ids
excluded_prompt_class_ids
fallback_class_id
score_threshold
prompt_variants
prompt_calls_per_patch
mask_filter_enabled
evaluation_split
```

在推荐主实验中：

```text
mask_filter_enabled = false
```

Exp3-D 还会额外保存：

```text
view_configs
```

## 12. 指标解读

Exp3 使用与 Exp1/Exp2 相同的指标口径。

论文主结果建议优先使用：

```text
dataset_overall_accuracy
dataset_mean_iou
dataset_mean_f1
dataset_frequency_weighted_iou
dataset_iou_per_class
dataset_confusion_matrix
```

`dataset_*` 指标基于所有成功样本的累计混淆矩阵计算，适合作为论文主表。

`average_*` 指标是逐图指标再平均，更适合分析图像间波动。

如果 `run_context.json` 中存在 skipped 或 failed 样本，则本次 `overall_metrics.json` 中的总体指标只代表成功样本，不能在论文中表述为完整数据集结果。

Exp3 分析不应只看 `dataset_mean_precision`。Exp1/Exp2 已经说明 precision 可能因为预测像素很少而虚高；更重要的是：

```text
mIoU
Mean F1
Mean Recall
各类别 IoU / Recall
预测类别分布
真实前景误分为 clutter/background 的比例
```

对于 Exp3-B / Exp3-C，还应增加一张类别级模态对比表：

| 类别 | RGB IoU | IRRG IoU | RGBIR IoU | 最优模态 | 解释 |
|---|---:|---:|---:|---|---|
| impervious surface | 待填 | 待填 | 待填 | 待填 | 是否受 NIR 影响 |
| building | 待填 | 待填 | 待填 | 待填 | 是否仍依赖 RGB 外观 |
| low vegetation | 待填 | 待填 | 待填 | 待填 | 是否受益于 NIR |
| tree | 待填 | 待填 | 待填 | 待填 | 是否改善树冠召回 |
| car | 待填 | 待填 | 待填 | 待填 | 是否 RGB 最稳定 |
| clutter/background | 待填 | 待填 | 待填 | 待填 | fallback 类别变化 |

这张表主要用于分析模态偏好。若要把“最优模态”变成正式方法，需要避免测试集调参问题。

## 13. 推荐运行顺序

建议按以下顺序运行：

development split 模态敏感性分析：

```bash
.venv_hf/bin/python exp3a_sam3_potsdam_rgb_modality.py

.venv_hf/bin/python exp3b_sam3_potsdam_irrg_modality.py

.venv_hf/bin/python exp3c_sam3_potsdam_rgbir_composite.py

.venv_hf/bin/python exp3d_sam3_potsdam_multiview_fusion.py
```

held-out split 最终评估：

```bash
.venv_hf/bin/python exp3a_sam3_potsdam_rgb_modality.py \
  --config exp3a_sam3_potsdam_rgb_modality_heldout.yaml

.venv_hf/bin/python exp3b_sam3_potsdam_irrg_modality.py \
  --config exp3b_sam3_potsdam_irrg_modality_heldout.yaml

.venv_hf/bin/python exp3c_sam3_potsdam_rgbir_composite.py \
  --config exp3c_sam3_potsdam_rgbir_composite_heldout.yaml

.venv_hf/bin/python exp3d_sam3_potsdam_multiview_fusion.py \
  --config exp3d_sam3_potsdam_multiview_fusion_heldout.yaml
```

正式运行前应确认：

```text
model.allow_hf_fallback: false
device.type: cuda
evaluation.test_images: []
evaluation.split: dev 或 heldout
mask_filter.enabled: false
Potsdam RGB / IRRG / RGBIR / noBoundary 标签文件路径完整
```

Exp3-D 需要同时具备 RGB、IRRG 和 RGBIR 三类输入文件；缺失任一视图的样本会被跳过。

## 14. 论文表述建议

英文表述可写为：

```text
Based on the Exp2 ablation results, Exp3 fixes the best-performing prompt strategy from Exp2-B rather than the mask-filtered Exp2-C variant. All Exp3 experiments use foreground visual prompt ensembles for the five non-background Potsdam classes, keep clutter/background only as a residual fallback class, and disable the fixed class-aware area filter because it slightly degraded the Exp2-B result. Following the official Potsdam label availability, the 24 participant-labeled tiles are used as a development split for modality sensitivity analysis, while the remaining 14 fully referenced tiles are used as a held-out split for final evaluation. Exp3-A uses RGB orthophotos as the modality baseline. Exp3-B uses IRRG inputs and Exp3-C converts RGBIR images into a NIR-R-G three-channel composite, not only to test whether they improve the overall score, but also to analyze class-specific modality preferences. Exp3-D performs equal-weight multi-view score fusion over RGB, IRRG, and RGBIR-derived views, while class-aware modality fusion is left as a follow-up design if clear class-level complementarity is observed on the development split.
```

中文表述可写为：

```text
基于 Exp2 的消融结果，Exp3 固定采用表现最好的 Exp2-B prompt 策略，而不是继承加入固定面积过滤的 Exp2-C。所有 Exp3 实验均对 Potsdam 五个前景类别使用视觉化 prompt ensemble，将 clutter/background 仅作为 residual fallback 类别，并关闭固定类别面积过滤，因为 Exp2-C 已显示该后处理会使 Exp2-B 结果略微下降。根据 Potsdam 官方标签可用性，本文将 24 张参赛者公开标签瓦片作为 development split，用于模态敏感性分析；将其余 14 张完整参考标签瓦片作为 held-out split，用于最终评估。Exp3-A 使用 RGB 正射影像作为模态基准；Exp3-B 使用 IRRG 影像，Exp3-C 将 RGBIR 四波段影像转换为 NIR-R-G 三通道 composite。Exp3-B 和 Exp3-C 不仅用于比较整体指标是否超过 RGB，也用于分析近红外相关输入对不同语义类别的选择性影响。Exp3-D 则对 RGB、IRRG 和 RGBIR composite 三个视图进行等权 score fusion；若 development split 上的单模态结果显示明显类别互补性，可进一步设计类别相关的模态选择或类别级融合权重，并在 held-out split 上最终评估。
```

## 15. 在论文主线中的意义

Exp3 将论文从 prompt 消融推进到多源遥感信息利用。

如果 Exp3-B、Exp3-C 或 Exp3-D 相比 Exp3-A 有改善，说明：

```text
SAM3 的开放词汇候选 mask 生成不仅受 prompt 影响，
也受输入模态和遥感光谱信息表达方式影响。
```

如果改善有限甚至下降，则说明：

```text
仅将多源遥感数据重排为三通道图像输入 SAM3，
未必足以让通用开放词汇模型真正利用遥感物理信息；
后续需要引入更显式的 DSM、NDVI、粗语义先验或水利领域约束。
```

因此，Exp3 为后续 Exp4 的 DSM / NDVI 先验融合提供实验依据。

如果 Exp3-B / Exp3-C 只在部分类别上提升，而整体指标没有明显提升，这并不意味着实验失败。它仍然可以支持以下论文论点：

```text
不同遥感输入模态对 SAM3 的开放词汇候选生成具有类别选择性影响；
简单整体替换输入模态未必稳定，
但类别相关的模态选择或遥感先验融合可能更合理。
```

## 16. 最终建议

当前最合理的 Exp3 路线是：

```text
Exp3-A-dev / Exp3-A-heldout：RGB + Exp2-B，作为模态基准；
Exp3-B-dev / Exp3-B-heldout：IRRG + Exp2-B，分析近红外假彩色输入的类别选择性影响；
Exp3-C-dev / Exp3-C-heldout：RGBIR NIR-R-G composite + Exp2-B，分析 RGBIR 派生输入的类别选择性影响；
Exp3-D1-dev / Exp3-D1-heldout：RGB/IRRG/RGBIR 多视图等权 score fusion + Exp2-B，检验朴素模态互补；
Exp3-D2：类别相关模态选择或类别级融合权重，作为后续设计或 Exp4 扩展；规则应由 development split 或遥感先验确定，再在 held-out split 上评估。
```

不建议把 Exp2-C 的固定面积过滤放入 Exp3 主实验。它可以保留为独立后处理消融结果，但不应作为模态实验的固定基础。

如果希望保持当前实验数量不膨胀，建议正文只运行 Exp3-D1，把 Exp3-D2 写为由 Exp3-B/C 类别级分析自然引出的后续方法设计。
