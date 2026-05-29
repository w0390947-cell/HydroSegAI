# 实验一：SAM3 + Potsdam 零样本遥感分割基线实验说明

## 1. 实验定位

Exp1 是整个论文实验体系中的基础零样本基线。

它回答的问题是：

```text
不进行遥感数据微调、不引入额外先验、不做 prompt 消融时，
通用开放词汇分割模型 SAM3 直接迁移到 Potsdam 遥感语义分割任务上的表现如何？
```

因此，Exp1 不是为了得到最高分，而是为了建立一个可复现的起点。

后续 Exp2、Exp3、Exp4 等实验都应以 Exp1 为参照，逐步分析：

```text
prompt 构造是否影响结果
背景类别是否适合主动 prompt
多源遥感影像是否能提供额外信息
NDVI、DSM 等先验是否能改善推理分割
```

## 2. 对应文件

| 类型 | 路径 |
|---|---|
| 实验脚本 | `exp1_sam3_potsdam_zeroshot_baseline.py` |
| 实验配置 | `exp1_sam3_potsdam_zeroshot_baseline.yaml` |
| 结果目录 | 由 YAML `paths.output_dir` 指定 |
| 结果说明 | `实验一_SAM3_Potsdam_零样本结果说明.md` |
| 结果分析 | `实验一_SAM3_Potsdam_零样本结果分析.md` |

## 3. 数据设置

Exp1 当前只使用 Potsdam 数据集中的 RGB 影像和 noBoundary 标签。

| 项目 | 内容 |
|---|---|
| 输入影像 | `Potsdam/2_Ortho_RGB/2_Ortho_RGB` |
| 参考标签 | `Potsdam/5_Labels_all_noBoundary` |
| 影像文件模式 | `{image_id}_RGB.tif` |
| 标签文件模式 | `{image_id}_label_noBoundary.tif` |
| 样本发现方式 | RGB 影像和 noBoundary 标签自动配对 |
| 样本数 | 以运行时自动发现的文件交集为准；当前服务器数据预计为 38 张 |
| 评估范围 | 完整可配对样本 |

配置位置：

```yaml
evaluation:
  test_images: []
```

`test_images` 留空时，脚本会固定使用文件交集自动发现完整配对样本，而不是手写固定行列范围。这样可以避免候选范围中存在影像或标签缺失时造成不一致。

如需调试小样本，应显式填写 `test_images`：

```yaml
evaluation:
  test_images:
    - "top_potsdam_2_10"
    - "top_potsdam_5_11"
```

## 4. Potsdam 类别定义

Exp1 使用 Potsdam 标准 6 类语义标签。

| 类别 ID | 类别名 | 标签颜色 | 说明 |
|---:|---|---|---|
| 0 | `impervious surface` | 白色 `(255,255,255)` | 不透水面，道路、铺装地面等 |
| 1 | `building` | 蓝色 `(0,0,255)` | 建筑物 |
| 2 | `low vegetation` | 青色 `(0,255,255)` | 低矮植被、草地等 |
| 3 | `tree` | 绿色 `(0,255,0)` | 树木 |
| 4 | `car` | 黄色 `(255,255,0)` | 车辆 |
| 5 | `clutter/background` | 红色 `(255,0,0)` | 杂波或背景 |
| 255 | `ignore` | 黑色 `(0,0,0)` | noBoundary 标签中的忽略区域 |

黑色 `(0,0,0)` 只作为 ignore 区域处理，不参与指标计算。

## 5. 零样本 Prompt 设置

Exp1 对 Potsdam 6 个评估类别都主动输入一个文本 prompt。

| 类别 ID | 类别名 | Prompt |
|---:|---|---|
| 0 | `impervious surface` | `impervious surface` |
| 1 | `building` | `building` |
| 2 | `low vegetation` | `low vegetation` |
| 3 | `tree` | `tree` |
| 4 | `car` | `car` |
| 5 | `clutter/background` | `background clutter` |

这就是 Exp1 与 Exp2-A 的核心区别：

```text
Exp1 主动 prompt clutter/background
Exp2-A 不主动 prompt clutter/background
```

Exp1 的 prompt 设计尽量贴近 Potsdam 官方类别名称，不做 prompt ensemble，也不引入额外视觉化描述。

## 6. 图像切片策略

Potsdam 单张影像尺寸较大，不能直接整图输入 SAM3，因此 Exp1 使用 patch 滑窗推理。

| 参数 | 当前值 |
|---|---:|
| `patch_size` | `1008` |
| `stride` | `672` |
| patch 重叠 | 约 1/3 |
| 单张 6000 x 6000 影像 patch 数 | 约 81 个 |

切片方式为边缘对齐滑窗：

```text
中间区域按 stride 滑动；
右边界和下边界的最后一个 patch 与图像边缘对齐；
尽量避免大面积 padding。
```

这种方式比简单滑窗后在边缘补零更适合遥感大图，因为它减少了 padding 对模型视觉输入的干扰。

## 7. Patch 预测与融合逻辑

每个 patch 的处理流程如下：

```text
读取 patch
转换为 PIL RGB 图像
调用 Sam3Processor.set_image()
依次输入 6 个类别 prompt
收集每个类别返回的 masks、scores、boxes
按 score_threshold 过滤候选 mask
将所有候选 mask 映射回 Potsdam 类别 ID
```

当前 `score_threshold` 为：

```yaml
image_processing:
  score_threshold: 0.5
```

脚本已经确保该阈值同时作用于：

```text
SAM3Processor 内部候选过滤阶段
脚本自身的二次 score 过滤阶段
```

这样后续如果调整 `score_threshold`，不会出现 SAM3Processor 仍按默认 `0.5` 提前丢弃候选的问题。

Patch 融合采用两阶段置信度融合：

```text
第一阶段发生在单个 patch 内：
同一像素如果被多个类别 mask 覆盖，脚本保留置信度最高的类别。

第二阶段发生在多个 patch 合并时：
同一像素如果被多个重叠 patch 覆盖，脚本按类别累积来自各 patch 的置信度分数；
最终选择累计分数最高的类别作为该像素预测类别。
```

如果某个像素被 patch 覆盖，但没有任何类别产生有效正置信度 mask，则回退为：

```text
class 5: clutter/background
```

如果理论上出现某个像素没有被任何 patch 覆盖，则视为切片覆盖错误，脚本应报错或进入严格检查逻辑，而不应静默当作背景。

### 7.1 推理健康检查

为了避免 SAM3 推理链路异常被误当作真实 zero-shot 结果，脚本会对每张图记录推理健康统计：

```text
patch_calls           patch 推理调用次数
patch_failures        patch 级推理失败次数
prompt_calls          prompt 推理调用次数
prompt_successes      prompt 推理成功次数
prompt_failures       prompt 推理失败次数
empty_prompt_outputs  成功调用但没有有效输出的 prompt 次数
valid_masks           通过 score_threshold 的有效 mask 数量
prompt_failure_rate   prompt 推理失败率
```

单张图必须满足以下条件才会进入指标计算：

```text
patch_failures == 0
prompt_failure_rate <= 5%
valid_masks >= 1
```

如果不满足这些条件，脚本会将该样本记为失败样本，并跳过该样本的指标计算。这样可以区分两种情况：

```text
正常推理但局部像素无有效 mask：这些像素回退为 clutter/background
整张图推理链路异常或完全无有效 mask：该图失败，不纳入总体指标
```

## 8. 评估指标

Exp1 在排除 ignore 像素后计算指标。

ignore 条件：

```text
ground_truth == 255
```

当前保存的主要指标包括：

| 指标 | 说明 |
|---|---|
| Overall Accuracy | 所有有效像素上的总体准确率 |
| Mean IoU | 6 个类别 IoU 的平均值 |
| Mean F1 | 6 个类别 F1 的平均值 |
| Mean Precision | 6 个类别 precision 的平均值 |
| Mean Recall | 6 个类别 recall 的平均值 |
| Frequency Weighted IoU | 按类别像素频率加权的 IoU |
| Per-class IoU | 各类别 IoU |
| Per-class F1 | 各类别 F1 |
| Per-class Precision | 各类别 precision |
| Per-class Recall | 各类别 recall |
| Confusion Matrix | 类别混淆矩阵 |

总体指标分为两类：

```text
dataset_*  : 先累计所有图像混淆矩阵，再计算指标，推荐作为论文主结果
average_*  : 先计算单图指标，再对图像取平均，作为辅助参考
```

论文主表更推荐报告：

```text
dataset_overall_accuracy
dataset_mean_iou
dataset_mean_f1
dataset_frequency_weighted_iou
dataset_iou_per_class
```

其中 precision 和 recall 仍会完整保存，但在 Exp1 中主要作为辅助诊断指标使用，用于分析类别误检和漏检倾向，不作为论文主表的核心排序指标。

## 9. 输出内容

Exp1 会在 YAML `paths.output_dir` 指定的目录下保存结果。当前服务器配置为：

```yaml
paths:
  output_dir: "/home/anjou/PythonENV/Test_11/results_exp1_sam3_potsdam_zeroshot_baseline"
```

逻辑目录名仍为 `results_exp1_sam3_potsdam_zeroshot_baseline`。

主要目录结构：

```text
results_exp1_sam3_potsdam_zeroshot_baseline/
  predictions/
  visualizations/
  metrics/
  logs/
```

### 9.1 predictions

保存每张图的预测和标签。

典型文件包括：

```text
*_prediction.png
*_ground_truth.png
*_prediction_id.png
*_ground_truth_id.png
```

其中：

```text
*_prediction.png      使用 Potsdam 标准颜色着色，便于人工查看
*_ground_truth.png    使用 Potsdam 标准颜色着色，便于对比
*_prediction_id.png   保存类别 ID 图，便于后续程序读取
*_ground_truth_id.png 保存标签 ID 图，便于后续程序读取
```

### 9.2 visualizations

保存输入图、预测图、标签图的对比可视化：

```text
*_comparison.png
```

### 9.3 metrics

保存单图和总体指标：

```text
overall_metrics.json
per_image_metrics.csv
run_context.json
config_snapshot.yaml
*_metrics.json
```

其中：

```text
overall_metrics.json     数据集总体指标，并包含 run_context、inference_health、模型来源和实际配置
per_image_metrics.csv    每张图的关键指标表
run_context.json         本次运行的样本完整性记录，包括成功、跳过和失败样本
config_snapshot.yaml     本次运行使用的 YAML 配置快照
*_metrics.json           单张图的完整指标，并包含该图的 inference_stats
```

`overall_metrics.json` 的 `run_context.metrics_scope` 为：

```text
successful_images_only
```

这表示总体指标只基于成功完成推理健康检查并完成指标计算的样本。若存在缺失文件或推理失败样本，应同时查看 `run_context.json` 或 `overall_metrics.json` 中的 `run_context` 字段。

### 9.4 logs

保存运行日志：

```text
evaluation_log_*.txt
```

## 10. 运行命令

正式运行：

```bash
.venv_hf/bin/python exp1_sam3_potsdam_zeroshot_baseline.py
```

如需指定配置文件：

```bash
.venv_hf/bin/python exp1_sam3_potsdam_zeroshot_baseline.py \
  --config exp1_sam3_potsdam_zeroshot_baseline.yaml
```

## 11. 与后续实验的关系

Exp1 是后续实验的参照组。

| 实验 | 与 Exp1 的关系 |
|---|---|
| Exp2-A | 去掉主动 `clutter/background` prompt |
| Exp2-B | 在 Exp2-A 基础上加入 foreground prompt ensemble |
| Exp2-C | 在 Exp2-B 基础上加入 class-aware mask area filtering |
| Exp3-A | 在 Exp2-C 推理策略基础上重新跑 RGB 输入 |
| Exp3-B | 将输入从 RGB 替换为 IRRG |
| Exp3-C | 使用 RGBIR 派生的三通道 composite |
| Exp3-D | RGB、IRRG、RGBIR composite 多视图结果融合 |

因此，Exp1 在论文中的角色是：

```text
直接迁移基线
后续 prompt 消融和多源信息融合的参照起点
证明通用开放词汇分割模型直接用于遥感语义分割时存在明显 domain gap
```

## 12. 论文写作建议

Exp1 可以放在论文实验章节的第一组实验中，标题可写为：

```text
SAM3 zero-shot baseline on Potsdam
```

建议在论文中强调三点：

```text
1. Exp1 不使用 Potsdam 训练集微调，属于严格零样本迁移。
2. Exp1 使用 Potsdam 官方类别名称作为文本 prompt，避免人为调参。
3. Exp1 的低分并不代表实验失败，而是说明通用开放词汇分割模型直接迁移到遥感密集语义分割任务时存在显著适配问题。
```

这为后续研究问题提供依据：

```text
遥感和水利场景是否需要一套由任务理解、prompt 构造、多源先验、候选 mask 生成、空间约束融合共同组成的通用推理分割技术体系？
```

