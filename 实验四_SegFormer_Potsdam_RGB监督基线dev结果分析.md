# 实验四：SegFormer + Potsdam RGB 监督基线 dev 结果分析

## 1. 本次运行概况

本次实验已完成 `train_segformer_potsdam.py` 的 SegFormer RGB supervised baseline 训练，结果保存于：

```text
results_segformer_potsdam_rgb/
```

本实验的核心变量是：

```text
使用监督语义分割模型 SegFormer-B0，在 Potsdam RGB 图像上训练 6 类 dense semantic segmentation baseline。
```

它与 Exp1-Exp3 的 SAM3 系列实验不同：SAM3 系列是 zero-shot / prompt-driven mask 生成与融合；本实验是使用 Potsdam 训练标签进行监督学习。因此，本实验的目的不是继续验证 SAM3 prompt 是否有效，而是建立一个稳定的监督语义先验模型，作为后续 `SegFormer prior + SAM3 refinement` 或 `coarse-to-fine` 融合路线的基础。

本次运行使用的数据与配置如下：

| 项目 | 内容 |
|---|---|
| 训练脚本 | `train_segformer_potsdam.py` |
| 配置文件 | `segformer_potsdam_rgb.yaml` |
| 预训练模型 | `models/segformer/segformer-b0-finetuned-ade-512-512` |
| 模型结构 | SegFormer-B0 |
| 输入影像 | `Potsdam/2_Ortho_RGB/2_Ortho_RGB` |
| 参考标签 | `Potsdam/5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary` |
| 输入模态 | RGB |
| 类别数 | 6 |
| Ignore index | 255 |
| Patch size | `512 x 512` |
| Train stride | 512 |
| Val stride | 512 |
| 训练图像数 | 18 |
| 验证图像数 | 6 |
| 训练 patch 数 | 2592 |
| 验证 patch 数 | 864 |
| Epochs | 40 |
| Batch size | 4 |
| Optimizer | AdamW |
| Learning rate | 0.00006 |
| Weight decay | 0.01 |
| AMP | true |
| Device | cuda |

输出文件检查：

| 输出类型 | 路径 | 说明 |
|---|---|---|
| best checkpoint | `results_segformer_potsdam_rgb/checkpoints/best` | 验证 mIoU 最优模型 |
| latest checkpoint | `results_segformer_potsdam_rgb/checkpoints/latest` | 最后一轮模型 |
| training summary | `results_segformer_potsdam_rgb/metrics/training_summary.json` | 最优 epoch 与总体训练信息 |
| training history | `results_segformer_potsdam_rgb/metrics/training_history.json` | 40 轮训练与验证曲线 |
| run context | `results_segformer_potsdam_rgb/metrics/run_context.json` | split、patch 数和类别信息 |
| config snapshot | `results_segformer_potsdam_rgb/metrics/config_snapshot.yaml` | 本次运行配置快照 |

此外，已完成 best checkpoint 的 validation full-tile sliding-window 评估，结果保存于：

```text
results_segformer_potsdam_rgb/eval_val/
```

full-tile 评估使用 6 张 validation tile，逐张生成完整 6000×6000 预测图，再计算整图和数据集级指标。相比训练过程中的 patch-level validation，这一结果更接近最终实际使用方式，因此本文将 full-tile 结果作为主结果。

## 2. 模型加载与分类头重建

本次使用的本地预训练模型为：

```text
models/segformer/segformer-b0-finetuned-ade-512-512
```

该模型原本是在 ADE20K 150 类语义分割上 fine-tune 的 SegFormer-B0。训练脚本加载时将分类头重建为 Potsdam 6 类：

```text
ADE20K 150-class classifier -> Potsdam 6-class classifier
```

加载时出现分类头维度 mismatch 是预期行为：

```text
decode_head.classifier.weight: [150, 256, 1, 1] -> [6, 256, 1, 1]
decode_head.classifier.bias:   [150] -> [6]
```

这表示 backbone 和 decoder 主体权重被继承，最终分类层重新初始化并在 Potsdam 训练集上学习。

## 3. Full-tile validation 主结果

`eval_segformer_potsdam.py --split val` 已成功处理 6 张 validation 图像：

```text
top_potsdam_7_7_RGB
top_potsdam_7_8_RGB
top_potsdam_7_9_RGB
top_potsdam_7_10_RGB
top_potsdam_7_11_RGB
top_potsdam_7_12_RGB
```

输出文件检查：

| 输出类型 | 数量 | 说明 |
|---|---:|---|
| `results_segformer_potsdam_rgb/eval_val/predictions` | 24 | 6 张图，每张包含彩色预测、彩色 GT、预测 ID、GT ID |
| `results_segformer_potsdam_rgb/eval_val/visualizations` | 6 | 每张图一张整图对比可视化 |
| `results_segformer_potsdam_rgb/eval_val/metrics` | 10 | 6 个单图 JSON + `overall_metrics.json`、`per_image_metrics.csv`、`run_context.json`、`config_snapshot.yaml` |

数据集级 full-tile 指标如下：

| 指标 | 数值 |
|---|---:|
| Overall Accuracy | 0.8932 |
| Mean IoU | 0.7856 |
| Mean F1 | 0.8762 |
| Mean Precision | 0.8744 |
| Mean Recall | 0.8785 |
| Frequency Weighted IoU | 0.8130 |

逐图平均指标如下，作为补充参考：

| 指标 | 数值 |
|---|---:|
| Average Overall Accuracy | 0.8933 |
| Average Mean IoU | 0.7460 |
| Average Mean F1 | 0.8369 |

总体判断：

```text
SegFormer-B0 在 validation full-tile sliding-window 评估中取得 0.7856 mIoU 和 0.8932 OA，说明该监督模型已经能够稳定完成 Potsdam RGB 六类密集语义分割。
```

与训练时 patch-level validation best 指标相比，full-tile sliding-window 结果略有提升：

| 评估方式 | OA | mIoU | Mean F1 | FWIoU |
|---|---:|---:|---:|---:|
| Patch-level validation | 0.8894 | 0.7799 | 0.8727 | 0.8067 |
| Full-tile sliding-window validation | 0.8932 | 0.7856 | 0.8762 | 0.8130 |

这说明重叠滑窗融合没有造成性能退化，反而通过 logits 平均带来轻微平滑收益。

## 4. Full-tile heldout 结果

`eval_segformer_potsdam.py --split heldout` 已成功处理 14 张 heldout 图像：

```text
top_potsdam_2_13_RGB
top_potsdam_2_14_RGB
top_potsdam_3_13_RGB
top_potsdam_3_14_RGB
top_potsdam_4_13_RGB
top_potsdam_4_14_RGB
top_potsdam_4_15_RGB
top_potsdam_5_13_RGB
top_potsdam_5_14_RGB
top_potsdam_5_15_RGB
top_potsdam_6_13_RGB
top_potsdam_6_14_RGB
top_potsdam_6_15_RGB
top_potsdam_7_13_RGB
```

输出文件检查：

| 输出类型 | 数量 | 说明 |
|---|---:|---|
| `results_segformer_potsdam_rgb/eval_heldout/predictions` | 56 | 14 张图，每张包含彩色预测、彩色 GT、预测 ID、GT ID |
| `results_segformer_potsdam_rgb/eval_heldout/visualizations` | 14 | 每张图一张整图对比可视化 |
| `results_segformer_potsdam_rgb/eval_heldout/metrics` | 18 | 14 个单图 JSON + `overall_metrics.json`、`per_image_metrics.csv`、`run_context.json`、`config_snapshot.yaml` |

数据集级 heldout 指标如下：

| 指标 | 数值 |
|---|---:|
| Overall Accuracy | 0.9021 |
| Mean IoU | 0.7687 |
| Mean F1 | 0.8535 |
| Mean Precision | 0.8772 |
| Mean Recall | 0.8413 |
| Frequency Weighted IoU | 0.8251 |

逐图平均指标如下：

| 指标 | 数值 |
|---|---:|
| Average Overall Accuracy | 0.9020 |
| Average Mean IoU | 0.7563 |
| Average Mean F1 | 0.8395 |

总体判断：

```text
SegFormer-B0 在 heldout full-tile 上取得 0.7687 mIoU 和 0.9021 OA，说明监督 RGB baseline 具有较好的未见图像泛化能力。heldout mIoU 略低于 val，但 OA 和 FWIoU 更高，说明下降主要来自少数类别的宏平均影响，而不是大面积主类整体退化。
```

## 5. Val 与 heldout 对比

| Split | 图像数 | OA | mIoU | Mean F1 | FWIoU |
|---|---:|---:|---:|---:|---:|
| validation | 6 | 0.8932 | 0.7856 | 0.8762 | 0.8130 |
| heldout | 14 | 0.9021 | 0.7687 | 0.8535 | 0.8251 |
| heldout - val | - | +0.0089 | -0.0169 | -0.0227 | +0.0121 |

从总体指标看，heldout 并没有出现明显崩塌。相反，OA 和 FWIoU 上升，说明模型对 heldout 中大面积类别的预测仍然稳定。mIoU 和 Mean F1 下降，主要由 `clutter/background` 类拖低。

逐类 IoU 对比如下：

| 类别 | Val IoU | Heldout IoU | 变化 |
|---|---:|---:|---:|
| impervious surface | 0.8557 | 0.8576 | +0.0018 |
| building | 0.8962 | 0.9248 | +0.0286 |
| low vegetation | 0.6729 | 0.7767 | +0.1038 |
| tree | 0.7481 | 0.7860 | +0.0379 |
| car | 0.8936 | 0.9034 | +0.0098 |
| clutter/background | 0.6470 | 0.3635 | -0.2834 |

这个对比非常关键：

```text
heldout 上 0-4 五个主要前景类别全部保持或超过 validation；
唯一明显下降的是 clutter/background。
```

因此，heldout 的 mIoU 下降不应解读为 SegFormer 主体语义能力退化，而应解读为：

```text
当前模型对 heldout 中 clutter/background 的召回不足，倾向于把部分杂乱/背景区域吸收到前景主类中。
```

Heldout 逐类指标如下：

| 类别 | GT 占比 | 预测占比 | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 31.52% | 32.64% | 0.8576 | 0.9233 | 0.9075 | 0.9397 |
| building | 24.93% | 24.85% | 0.9248 | 0.9609 | 0.9625 | 0.9593 |
| low vegetation | 20.61% | 22.02% | 0.7767 | 0.8743 | 0.8464 | 0.9041 |
| tree | 17.09% | 16.27% | 0.7860 | 0.8802 | 0.9022 | 0.8593 |
| car | 1.59% | 1.61% | 0.9034 | 0.9493 | 0.9433 | 0.9553 |
| clutter/background | 4.26% | 2.61% | 0.3635 | 0.5332 | 0.7014 | 0.4301 |

其中 `clutter/background` 的 GT 占比为 4.26%，预测占比只有 2.61%，recall 为 0.4301，说明模型对该类明显欠预测。由于 `clutter/background` 是语义上最杂、边界也最不稳定的类别，这个问题在遥感监督分割中是可以预期的。

## 6. Heldout 逐图结果

| 图像 | OA | mIoU | Mean F1 | FWIoU |
|---|---:|---:|---:|---:|
| top_potsdam_2_13_RGB | 0.9094 | 0.7545 | 0.8407 | 0.8384 |
| top_potsdam_2_14_RGB | 0.9111 | 0.7704 | 0.8423 | 0.8373 |
| top_potsdam_3_13_RGB | 0.9087 | 0.7564 | 0.8346 | 0.8374 |
| top_potsdam_3_14_RGB | 0.9040 | 0.7754 | 0.8593 | 0.8267 |
| top_potsdam_4_13_RGB | 0.8957 | 0.7215 | 0.7947 | 0.8133 |
| top_potsdam_4_14_RGB | 0.8398 | 0.7166 | 0.8125 | 0.7276 |
| top_potsdam_4_15_RGB | 0.9092 | 0.7374 | 0.8197 | 0.8383 |
| top_potsdam_5_13_RGB | 0.9137 | 0.7617 | 0.8470 | 0.8437 |
| top_potsdam_5_14_RGB | 0.8871 | 0.7639 | 0.8565 | 0.7979 |
| top_potsdam_5_15_RGB | 0.8850 | 0.7376 | 0.8306 | 0.7962 |
| top_potsdam_6_13_RGB | 0.9347 | 0.7776 | 0.8427 | 0.8808 |
| top_potsdam_6_14_RGB | 0.9339 | 0.7729 | 0.8450 | 0.8784 |
| top_potsdam_6_15_RGB | 0.9132 | 0.8026 | 0.8838 | 0.8489 |
| top_potsdam_7_13_RGB | 0.8830 | 0.7402 | 0.8436 | 0.8026 |

Heldout 中 mIoU 最高的是 `top_potsdam_6_15_RGB`，mIoU 为 0.8026；最低的是 `top_potsdam_4_14_RGB`，mIoU 为 0.7166。整体逐图波动较小，14 张图中没有出现失败样本。

## 7. 训练过程结果

本次训练共运行 40 个 epoch，best checkpoint 出现在第 16 轮：

| 指标 | 数值 |
|---|---:|
| Best epoch | 16 |
| Best val Overall Accuracy | 0.8894 |
| Best val Mean IoU | 0.7799 |
| Best val Mean F1 | 0.8727 |
| Best val Mean Precision | 0.8717 |
| Best val Mean Recall | 0.8742 |
| Best val Frequency Weighted IoU | 0.8067 |
| Best train loss | 0.1783 |
| Best val loss | 0.4404 |

训练初始、最优与末轮对比如下：

| Epoch | Train loss | Val loss | Val OA | Val mIoU | Val Mean F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7370 | 0.6040 | 0.8069 | 0.5006 | 0.6055 |
| 16 | 0.1783 | 0.4404 | 0.8894 | 0.7799 | 0.8727 |
| 40 | 0.1137 | 0.5929 | 0.8854 | 0.7730 | 0.8677 |

前 8 个验证 mIoU 最好的 epoch 如下：

| Rank | Epoch | Train loss | Val loss | Val OA | Val mIoU | Val Mean F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 0.1783 | 0.4404 | 0.8894 | 0.7799 | 0.8727 |
| 2 | 12 | 0.2085 | 0.4128 | 0.8905 | 0.7791 | 0.8723 |
| 3 | 39 | 0.1129 | 0.5816 | 0.8866 | 0.7786 | 0.8719 |
| 4 | 14 | 0.1894 | 0.4564 | 0.8898 | 0.7784 | 0.8717 |
| 5 | 31 | 0.1313 | 0.5257 | 0.8876 | 0.7779 | 0.8713 |
| 6 | 33 | 0.1252 | 0.5484 | 0.8863 | 0.7766 | 0.8706 |
| 7 | 20 | 0.1621 | 0.4690 | 0.8868 | 0.7761 | 0.8700 |
| 8 | 34 | 0.1249 | 0.5487 | 0.8858 | 0.7759 | 0.8703 |

训练过程判断：

```text
SegFormer supervised baseline 在 dev validation patch 上已经形成稳定、可用的 dense semantic prior。full-tile 评估进一步确认该 prior 可以迁移到整图推理场景。
```

## 8. Validation full-tile 各类别结果

Full-tile validation 的逐类指标如下：

| 类别 | GT 占比 | 预测占比 | IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| impervious surface | 34.14% | 33.83% | 0.8557 | 0.9223 | 0.9264 | 0.9182 |
| building | 30.00% | 28.97% | 0.8962 | 0.9453 | 0.9620 | 0.9291 |
| low vegetation | 15.29% | 16.21% | 0.6729 | 0.8045 | 0.7819 | 0.8284 |
| tree | 12.02% | 12.45% | 0.7481 | 0.8559 | 0.8409 | 0.8715 |
| car | 1.59% | 1.57% | 0.8936 | 0.9438 | 0.9494 | 0.9383 |
| clutter/background | 6.97% | 6.97% | 0.6470 | 0.7856 | 0.7857 | 0.7856 |

类别级观察：

1. `building` 和 `car` 的表现最好，IoU 分别为 0.8962 和 0.8936，说明监督模型能够稳定学习建筑屋顶和小目标汽车。
2. `impervious surface` 达到 0.8557 IoU，证明道路、硬化地表等大面积类别已经被可靠建模。
3. `low vegetation` 的 IoU 为 0.6729，是主要前景类别中相对较低的一类，主要原因可能是低矮植被与树木、不透水表面之间存在边界和纹理混淆。
4. `tree` 的 IoU 为 0.7481，明显优于 SAM3 系列中几乎无法召回树冠的情况。
5. `clutter/background` 的 IoU 为 0.6470，是 6 类中最低，但该类本身语义杂、像素占比低，且 noBoundary 标签中边界被忽略，对该类稳定建模更难。

## 9. Validation 逐图结果

Full-tile validation 的逐图指标如下：

| 图像 | OA | mIoU | Mean F1 | FWIoU |
|---|---:|---:|---:|---:|
| top_potsdam_7_7_RGB | 0.8930 | 0.7415 | 0.8301 | 0.8132 |
| top_potsdam_7_8_RGB | 0.8901 | 0.7207 | 0.8156 | 0.8117 |
| top_potsdam_7_9_RGB | 0.9110 | 0.8296 | 0.9052 | 0.8391 |
| top_potsdam_7_10_RGB | 0.8457 | 0.7473 | 0.8476 | 0.7442 |
| top_potsdam_7_11_RGB | 0.9202 | 0.7526 | 0.8413 | 0.8635 |
| top_potsdam_7_12_RGB | 0.8996 | 0.6842 | 0.7815 | 0.8431 |

其中 `top_potsdam_7_9_RGB` 的 mIoU 最高，为 0.8296；`top_potsdam_7_12_RGB` 的 mIoU 最低，为 0.6842。不同 tile 之间的 mIoU 差异说明模型对局部场景分布、类别组成和 clutter/background 复杂度仍较敏感。

## 10. 与 SAM3 系列实验的对比

为了说明实验路线变化的意义，这里将 SegFormer full-tile validation 结果与目前 SAM3 dev 系列的主要结果放在一起参考。SegFormer 和 SAM3 的评估 split 与训练监督条件并不完全相同，因此这不是严格排行榜，但可以清楚反映方法能力差异。

| 实验 | 方法 | 输入 | 监督信号 | OA | mIoU | Mean F1 | FWIoU |
|---|---|---|---|---:|---:|---:|---:|
| Exp3-A | SAM3 + prompt ensemble | RGB | 无监督/zero-shot | 0.1677 | 0.1032 | 0.1781 | 0.1048 |
| Exp3-B | SAM3 + prompt ensemble | IRRG | 无监督/zero-shot | 0.1284 | 0.0629 | 0.1098 | 0.0548 |
| Exp3-C | SAM3 + prompt ensemble | RGBIR-derived NIR-R-G | 无监督/zero-shot | 0.1296 | 0.0632 | 0.1104 | 0.0575 |
| Exp4 val | SegFormer-B0 full-tile | RGB | Potsdam train labels | 0.8932 | 0.7856 | 0.8762 | 0.8130 |
| Exp4 heldout | SegFormer-B0 full-tile | RGB | Potsdam train labels | 0.9021 | 0.7687 | 0.8535 | 0.8251 |

最关键的差异不只是指标更高，而是错误模式发生了根本变化：

```text
SAM3 系列主要问题是候选 mask 覆盖不足和 prompt-类别对齐失败；
SegFormer 的主要问题则转为常规语义分割中的类别边界、相似类别混淆和小类泛化。
```

这说明当前项目下一阶段从 SAM3 prompt 硬调转向监督语义 prior 是合理的。

## 11. 训练曲线解读

从训练曲线看，第 1 轮到第 16 轮验证 mIoU 从 0.5006 提升到 0.7799，说明 SegFormer 很快适应了 Potsdam 的遥感 RGB 分布。

第 16 轮之后，训练 loss 继续下降：

```text
epoch 16 train loss = 0.1783
epoch 40 train loss = 0.1137
```

但验证 loss 从 0.4404 上升到 0.5929，验证 mIoU 从 0.7799 小幅下降到 0.7730。这说明 16 轮以后出现轻微过拟合或验证集收益饱和。由于 mIoU 下降幅度不大，模型整体仍稳定，但论文或后续实验中应使用：

```text
results_segformer_potsdam_rgb/checkpoints/best
```

而不是 latest checkpoint。

## 12. 对后续工作的启示

本实验基本确认：

```text
SegFormer 可以作为 Potsdam 上稳定的 dense semantic prior。
```

这对后续路线有三点启示：

1. 后续不应继续把 SAM3 当作单独完成六类 dense semantic segmentation 的主模型。
2. SegFormer 可以先给出完整、稳定的 6 类语义图，解决 SAM3 大面积漏检和 fallback 过多的问题。
3. SAM3 更适合作为候选 mask、边界细化或局部实例级修正模块，而不是直接替代监督语义分割模型。

推荐下一步：

```text
基于 SegFormer full-tile prediction / logits / uncertainty 设计 SAM3 refinement 实验。
```

下一步可以围绕 `clutter/background` 欠召回和边界细化设计后续实验：

1. 使用 SegFormer logits/softmax entropy 找出低置信区域。
2. 用 SAM3 只处理边界或低置信区域，而不是整图替代 SegFormer。
3. 检查 `clutter/background` 与建筑、道路、植被之间的主要混淆方向。
