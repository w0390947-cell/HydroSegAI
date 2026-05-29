# 实验二：SAM3 + Potsdam Prompt 消融实验说明

## 1. 实验定位

Exp2 系列实验用于分析开放词汇分割模型在遥感密集语义分割中的关键推理因素。

它不是单纯为了调高 SAM3 在 Potsdam 上的分数，而是服务于论文中的核心问题：

```text
通用开放词汇分割模型直接迁移到遥感语义分割时，
prompt 构造、背景建模和 mask 后处理会如何影响最终结果？
```

Exp1 已经建立了基础零样本基线：

```text
RGB 输入 + Potsdam 官方 6 类 prompt + confidence fusion
```

Exp2 在 Exp1 的基础上逐步改变推理策略，用于判断问题主要来自：

```text
background prompt
类别 prompt 表达
prompt ensemble 带来的候选 mask 噪声
mask 质量过滤
```

## 2. 数据与评估设置

Exp2-A / Exp2-B / Exp2-C 使用相同数据：

| 项目 | 内容 |
|---|---|
| 输入影像 | `Potsdam/2_Ortho_RGB/2_Ortho_RGB` |
| 参考标签 | `Potsdam/5_Labels_all_noBoundary` |
| 标签语义 | Potsdam 标准 6 类 |
| ignore 区域 | noBoundary 标签中的黑色 `(0,0,0)`，映射为 `255` |
| 样本发现方式 | RGB 影像和 noBoundary 标签自动配对 |
| 当前样本数 | 默认自动发现 RGB 影像与 noBoundary 标签的完整交集；当前数据集为 38 张 |
| patch size | `1008` |
| stride | `672` |
| patch 方式 | 边缘对齐滑窗 |
| patch 融合 | SAM3 置信度加权融合 |

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

## 3. Exp2-A：No Background Prompt

### 3.1 实验目的

Exp1 的结果显示，大量像素被预测或回退为 `clutter/background`。其中一个可能原因是：

```text
background clutter 不是稳定、明确的开放词汇视觉概念。
```

Exp2-A 用于验证：

```text
是否应该主动向 SAM3 输入 clutter/background prompt？
```

### 3.2 实验变量

Exp2-A 保持 Exp1 的数据、切片、融合、fallback 和评估逻辑不变，只改变 prompt 输入。

主动 prompt 类别：

```text
0 impervious surface
1 building
2 low vegetation
3 tree
4 car
```

不主动 prompt：

```text
5 clutter/background
```

但 `clutter/background` 仍然保留为：

```text
评估类别
fallback 类别
```

也就是说，当某个被 patch 覆盖的位置没有任何有效正置信度 mask 时，该像素仍回退为 `clutter/background`。

### 3.3 对应文件

| 类型 | 路径 |
|---|---|
| 脚本 | `exp2a_sam3_potsdam_no_background_prompt.py` |
| 配置 | `exp2a_sam3_potsdam_no_background_prompt.yaml` |
| 结果目录 | `results_exp2a_sam3_potsdam_no_background_prompt/` |

### 3.4 运行命令

正式运行：

```bash
.venv_hf/bin/python exp2a_sam3_potsdam_no_background_prompt.py
```

如需显式指定配置或输出目录：

```bash
.venv_hf/bin/python exp2a_sam3_potsdam_no_background_prompt.py \
  --config exp2a_sam3_potsdam_no_background_prompt.yaml \
  --output-dir /home/anjou/PythonENV/Test_11/results_exp2a_sam3_potsdam_no_background_prompt
```

如需小样本检查，可在 YAML 的 `evaluation.test_images` 中显式填写样本 ID；留空则运行所有可配对样本。

## 4. Exp2-B：Visual Prompt Ensemble

### 4.1 实验目的

Potsdam 官方类别名中存在一些偏遥感数据集术语的表达，例如：

```text
impervious surface
low vegetation
background clutter
```

这些词不一定是通用开放词汇视觉模型最容易理解的自然图像概念。

Exp2-B 用于验证：

```text
更视觉化、更具体的 prompt ensemble 是否能改善候选 mask 生成？
```

### 4.2 实验变量

Exp2-B 继承 Exp2-A 的背景处理方式：

```text
不主动 prompt clutter/background
clutter/background 仍作为 fallback 和评估类别
```

同时对 5 个前景类别使用 prompt ensemble。

当前每类 5 个 prompt：

| 类别 | Prompt variants |
|---|---|
| `impervious surface` | `impervious surface`; `road and pavement`; `asphalt road`; `paved area`; `parking lot` |
| `building` | `building`; `buildings`; `rooftop`; `building roof`; `house roof` |
| `low vegetation` | `low vegetation`; `grass`; `lawn`; `low plants`; `ground vegetation` |
| `tree` | `tree`; `trees`; `tree crown`; `tall tree`; `urban trees` |
| `car` | `car`; `cars`; `vehicle`; `small vehicle`; `parked car` |

所有 prompt variants 仍映射回对应的 Potsdam 类别 ID。

例如：

```text
building
rooftop
building roof
```

都映射为：

```text
class 1: building
```

这些 prompt variants 是在运行测试集之前，依据 Potsdam 类别语义和通用视觉概念预先设计的固定集合；不是根据测试集指标、单张测试图像可视化结果或逐类测试表现反复筛选得到的。所有测试图像和后续实验均使用同一组 prompt variants。

### 4.3 计算量

Exp2-B 每个 patch 的 prompt 调用数为：

```text
5 类 x 每类 5 个 prompt = 25 次
```

在 38 张 Potsdam 图像、每张 81 个 patch 的设置下：

```text
总 patch 数 = 3078
总 prompt 调用数 = 76950
```

### 4.4 对应文件

| 类型 | 路径 |
|---|---|
| 脚本 | `exp2b_sam3_potsdam_visual_prompt_ensemble.py` |
| 配置 | `exp2b_sam3_potsdam_visual_prompt_ensemble.yaml` |
| 结果目录 | `results_exp2b_sam3_potsdam_visual_prompt_ensemble/` |

### 4.5 运行命令

正式运行：

```bash
.venv_hf/bin/python exp2b_sam3_potsdam_visual_prompt_ensemble.py
```

如需显式指定配置或输出目录：

```bash
.venv_hf/bin/python exp2b_sam3_potsdam_visual_prompt_ensemble.py \
  --config exp2b_sam3_potsdam_visual_prompt_ensemble.yaml \
  --output-dir /home/anjou/PythonENV/Test_11/results_exp2b_sam3_potsdam_visual_prompt_ensemble
```

## 5. Exp2-C：Prompt Ensemble + Mask Filtering

### 5.1 实验目的

Prompt ensemble 可能增加召回，但也可能引入更多噪声 mask。

Exp2-C 用于验证：

```text
开放词汇模型生成的候选 mask 是否需要质量过滤？
```

它在 Exp2-B 的基础上增加：

```text
类别感知的 mask 面积过滤
```

### 5.2 实验变量

Exp2-C 继承：

```text
Exp2-A 的 no background prompt
Exp2-B 的 foreground prompt ensemble
```

新增：

```text
class-aware mask area filtering
```

Exp2-C 的 `score_threshold` 与 Exp2-B 保持一致：

```text
score_threshold = 0.5
```

这样 Exp2-C 相比 Exp2-B 只新增 `class-aware mask area filtering`，结果差异可以更清晰地归因于 mask 面积过滤。

面积比例定义为：

```text
mask_area_ratio = mask 像素数 / patch 像素数
```

当前 patch 尺寸为：

```text
1008 x 1008 = 1,016,064 pixels
```

### 5.3 面积过滤规则

当前固定规则如下：

| 类别 | min area ratio | max area ratio |
|---|---:|---:|
| `impervious surface` | 0.001 | 0.80 |
| `building` | 0.0005 | 0.60 |
| `low vegetation` | 0.001 | 0.80 |
| `tree` | 0.0003 | 0.50 |
| `car` | 0.00001 | 0.05 |

这些阈值按 Potsdam 高分辨率影像中不同类别的常见尺度预先固定，不针对单张测试图像调参。

与 Exp2-B 的 prompt variants 一样，Exp2-C 的面积阈值属于实验前设定的类别尺度先验，而不是根据测试集结果调出的最优阈值。

### 5.4 过滤统计

Exp2-C 会保存每类 mask 过滤统计：

```text
input_masks
kept_masks
filtered_small
filtered_large
filtered_invalid
```

保存位置：

```text
results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/metrics/experiment_metadata.json
```

### 5.5 对应文件

| 类型 | 路径 |
|---|---|
| 脚本 | `exp2c_sam3_potsdam_prompt_ensemble_mask_filter.py` |
| 配置 | `exp2c_sam3_potsdam_prompt_ensemble_mask_filter.yaml` |
| 结果目录 | `results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter/` |

### 5.6 运行命令

正式运行：

```bash
.venv_hf/bin/python exp2c_sam3_potsdam_prompt_ensemble_mask_filter.py
```

如需显式指定配置或输出目录：

```bash
.venv_hf/bin/python exp2c_sam3_potsdam_prompt_ensemble_mask_filter.py \
  --config exp2c_sam3_potsdam_prompt_ensemble_mask_filter.yaml \
  --output-dir /home/anjou/PythonENV/Test_11/results_exp2c_sam3_potsdam_prompt_ensemble_mask_filter
```

## 6. Exp2 系列对比关系

| 实验 | background prompt | prompt 设置 | mask 过滤 | 主要问题 |
|---|---|---|---|---|
| Exp1 | yes | 官方 6 类 prompt | 无额外过滤 | 直接 zero-shot 基线 |
| Exp2-A | no | 官方前景类 prompt | 无额外过滤 | 背景是否应作为 residual |
| Exp2-B | no | 前景类 prompt ensemble | 无额外过滤 | prompt 构造是否影响候选 mask |
| Exp2-C | no | 前景类 prompt ensemble | class-aware area filtering，score threshold 与 Exp2-B 一致 | 候选 mask 是否需要质量控制 |

## 7. 先验固定与测试集调参控制

为避免测试集调参或 prompt tuning 泄漏，Exp2-B 和 Exp2-C 的设计遵循以下约束：

```text
Exp2-B 的 prompt variants 在运行测试集之前确定；
Exp2-C 的 mask 面积过滤阈值在运行测试集之前确定；
所有测试图像使用同一组 prompt variants 和同一组过滤阈值；
不针对单张测试图像、逐图指标或测试集总体指标临时修改 prompt 或阈值。
```

因此，Exp2-B 应被解释为“基于类别语义的视觉化 prompt ensemble 消融”，而不是在测试集上搜索最优 prompt 组合；Exp2-C 应被解释为“基于类别尺度先验的固定面积过滤消融”，而不是在测试集上搜索最优后处理阈值。

这一点在论文中可以表述为：

```text
To avoid test-set-specific prompt or threshold tuning, all prompt variants and mask area thresholds were defined a priori according to the semantic meanings and typical spatial scales of the Potsdam classes, and were kept fixed for all test images.
```

中文可写为：

```text
为避免针对测试集进行 prompt 或阈值调参，所有 prompt 变体和 mask 面积阈值均依据 Potsdam 类别语义及典型空间尺度在实验前预先设定，并在全部测试图像上保持固定。
```

## 8. 输出文件结构

每个 Exp2 实验会保存：

```text
results_exp2*/predictions/
results_exp2*/visualizations/
results_exp2*/metrics/
results_exp2*/logs/
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
| `overall_metrics.json` | 数据集级指标、逐图平均指标、实验元数据 |
| `per_image_metrics.csv` | 每张图的 OA、mIoU、F1、precision、recall、FWIoU |
| `experiment_metadata.json` | 当前实验的 prompt 设置、fallback 类别、过滤规则等 |
| `config_snapshot.yaml` | 本次运行使用的 YAML 配置快照 |
| `run_context.json` | 本次运行的样本清单、成功样本、跳过样本、失败样本和指标统计范围 |

每张图还会在 `metrics/` 下保存对应的单图指标 JSON。若 `metrics.save_confusion_matrix: true`，会额外保存 `metrics/dataset_confusion_matrix.csv`。若 `output.save_predictions` 或 `output.save_visualizations` 为 `false`，对应的预测图或可视化图不会生成，但目录仍会创建。

## 9. 指标解读

论文主结果建议优先使用 `dataset_*` 指标。

核心指标：

```text
dataset_overall_accuracy
dataset_mean_iou
dataset_mean_f1
dataset_mean_precision
dataset_mean_recall
dataset_frequency_weighted_iou
dataset_iou_per_class
dataset_f1_per_class
dataset_precision_per_class
dataset_recall_per_class
```

`average_*` 指标是逐图平均，可用于辅助分析不同图像之间的波动，不建议作为论文主结果。

Exp2 分析时应重点比较：

```text
clutter/background 预测比例是否下降；
impervious surface / building / vegetation / tree / car 的 IoU 是否改善；
prompt ensemble 是否提高召回；
mask filtering 是否减少噪声并提高 precision；
FWIoU 与 mIoU 是否出现方向不一致。
```

## 10. 推荐运行顺序

建议按以下顺序运行：

```bash
.venv_hf/bin/python exp2a_sam3_potsdam_no_background_prompt.py

.venv_hf/bin/python exp2b_sam3_potsdam_visual_prompt_ensemble.py

.venv_hf/bin/python exp2c_sam3_potsdam_prompt_ensemble_mask_filter.py
```

其中 Exp2-B 和 Exp2-C 的计算量明显大于 Exp2-A。

正式运行前应确认：

```text
model.allow_hf_fallback: false
device.type: cuda
evaluation.test_images: []  # 论文主实验默认运行全部可配对样本
```

如果 SAM3 模块不可用，脚本会直接抛出 `RuntimeError` 并输出 Python 可执行文件、当前工作目录、SAM3 导入错误和 sys.path 诊断信息；不会以成功状态静默结束。

## 11. 论文表述建议

英文表述可写为：

```text
To analyze the influence of reasoning strategies in open-vocabulary remote-sensing segmentation, we design a series of prompt ablation experiments on the Potsdam dataset. Exp2-A removes the active clutter/background prompt and treats background as a residual fallback class. Exp2-B further replaces single foreground prompts with visual prompt ensembles. Exp2-C adds class-aware mask area filtering while keeping the same confidence threshold as Exp2-B, so that the effect of mask filtering can be isolated. All experiments use the same RGB inputs, noBoundary labels, patching strategy, confidence-based fusion, and dataset-level evaluation protocol.
The prompt variants and mask area thresholds are defined a priori according to class semantics and typical object scales, and are kept fixed for all test images to avoid test-set-specific prompt or threshold tuning.
```

中文表述可写为：

```text
为分析开放词汇模型在遥感语义分割中的推理策略影响，本文在 Potsdam 数据集上设计了 Exp2 系列 prompt 消融实验。Exp2-A 去除主动背景 prompt，将 clutter/background 仅作为 residual fallback 类别；Exp2-B 在此基础上使用前景类别的视觉化 prompt ensemble；Exp2-C 在保持与 Exp2-B 相同置信度阈值的前提下，进一步加入类别感知的 mask 面积过滤，从而单独分析候选 mask 质量控制的影响。所有实验保持 RGB 输入、noBoundary 标签、切片策略、置信度融合和数据集级评价协议一致。
为避免针对测试集进行 prompt 或阈值调参，所有 prompt 变体和 mask 面积阈值均依据类别语义和典型目标尺度预先设定，并在全部测试图像上保持固定。
```

## 12. 在论文主线中的意义

Exp2 系列服务于本文“遥感/水利推理分割技术体系”的要素分析。

它说明：

```text
开放词汇分割模型的表现不仅由模型本体决定，
还受到任务概念表达、背景建模、候选 mask 质量控制和融合策略的影响。
```

因此，Exp2 的结论将为后续 Exp3 / Exp4 / Exp5 提供依据：

```text
Exp3: 固定较优推理策略，分析多源遥感输入模态影响；
Exp4: 引入 DSM / NDVI 等遥感先验；
Exp5: 引入粗语义分割模型，与开放词汇候选 mask 协同。
```
