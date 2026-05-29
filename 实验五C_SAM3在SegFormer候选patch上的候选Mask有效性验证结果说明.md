# 实验五C：SAM3 在 SegFormer candidate patches 上的候选 mask 有效性验证结果说明

## 1. 实验名称

```text
Exp5-C: SAM3 on SegFormer candidate patches
```

中文名称：

```text
SAM3 在 SegFormer candidate patches 上的候选 mask 有效性验证
```

## 2. 实验目的

本实验用于验证：

```text
在 SegFormer 已经定位出的疑难候选区域中，
SAM3 生成的候选 mask 是否能够覆盖 SegFormer 错误像素和 clutter/background 漏检像素。
```

该实验是 SegFormer + SAM3 融合前的验证实验，不直接作为最终融合方法。

## 3. 运行命令

```bash
python exp5c_sam3_candidate_patch_validation.py \
  --config exp5c_sam3_candidate_patch_validation.yaml
```

## 4. 配置文件

```text
exp5c_sam3_candidate_patch_validation.yaml
```

关键配置：

```yaml
evaluation:
  split: "heldout"
  max_images: -1
  max_patches_per_image: 6
  max_total_patches: 60
  min_candidate_ratio: 0.08
  sort_patches_by: "candidate_ratio"

image_processing:
  patch_size: 1008
  stride: 672
  score_threshold: 0.5
  restrict_masks_to_candidate: true

prompt_policy:
  prompt_class_ids: [0, 1, 2, 3, 4]
  excluded_prompt_class_ids: [5]
```

说明：

```text
本实验不主动 prompt clutter/background。
SAM3 输出 mask 在评价时被限制在 SegFormer candidate region 内。
```

## 5. 输入结果目录

SegFormer heldout 不确定性结果：

```text
results_segformer_potsdam_rgb/eval_heldout_uncertainty
```

Exp5-B candidate regions：

```text
results_exp5b_segformer_candidate_regions_heldout
```

## 6. 输出结果目录

```text
results_exp5c_sam3_candidate_patch_validation_heldout
```

主要输出文件：

```text
metrics/candidate_patch_proposals.csv
metrics/sam_patch_metrics.csv
metrics/sam_mask_metrics.csv
metrics/sam_candidate_validation_summary.json
patch_visualizations/*.png
logs/evaluation_log_20260523_230506.txt
```

## 7. 运行信息

SAM3 加载日志显示：

```text
使用本地 checkpoint:
/home/anjou/PythonENV/Test_11/sam3/checkpoints/sam3.1_multiplex.pt

CUDA 推理 dtype:
torch.float32

device:
cuda:0
```

运行范围：

```text
split: heldout
image count: 14
selected patches: 60
patch size: 1008
stride: 672
```

## 8. 总体指标

来自：

```text
metrics/sam_candidate_validation_summary.json
```

核心计数：

```text
patches: 60
SAM3 masks: 465
candidate pixels: 26,961,588
SegFormer error pixels: 10,602,086
clutter/background FN pixels: 3,107,873
SAM3 union pixels: 7,644,700
SAM3 error overlap: 2,508,005
SAM3 clutter FN overlap: 617,824
SAM3 candidate overlap: 7,644,700
```

核心指标：

```text
sam_error_coverage: 0.2366
sam_clutter_fn_coverage: 0.1988
sam_error_precision: 0.3281
sam_clutter_fn_precision: 0.0808
sam_candidate_overlap_ratio: 0.2835
```

## 9. 指标解释

```text
sam_error_coverage = 0.2366
```

表示在这 60 个候选 patch 内，SAM3 mask union 覆盖了约 23.66% 的 SegFormer 错误像素。

```text
sam_clutter_fn_coverage = 0.1988
```

表示 SAM3 mask union 覆盖了约 19.88% 的 clutter/background 漏检像素。

```text
sam_error_precision = 0.3281
```

表示 SAM3 mask union 中约 32.81% 的像素确实落在 SegFormer 错误区域。

```text
sam_clutter_fn_precision = 0.0808
```

表示 SAM3 mask union 中只有约 8.08% 的像素属于 clutter/background 漏检区域。

```text
sam_candidate_overlap_ratio = 0.2835
```

表示 candidate region 中约 28.35% 被 SAM3 mask union 覆盖。

## 10. patch 级统计

```text
zero-mask patches: 14 / 60
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

## 11. mask 类别统计

按 SAM3 prompt class 统计：

```text
class 0 impervious surface:
  masks: 30
  eval area: 7,707,872
  error precision: 0.3312
  clutter precision: 0.0794
  GT precision: 0.1285

class 1 building:
  masks: 13
  eval area: 91,208
  error precision: 0.0906
  clutter precision: 0.0351
  GT precision: 0.8140

class 2 low vegetation:
  masks: 19
  eval area: 395,766
  error precision: 0.3398
  clutter precision: 0.0078
  GT precision: 0.4691

class 3 tree:
  masks: 0

class 4 car:
  masks: 403
  eval area: 40,532
  error precision: 0.1868
  clutter precision: 0.0000
  GT precision: 0.8862
```

## 12. 结果摘要

本实验说明：

```text
SAM3 在 SegFormer candidate patches 中可以产生一定局部候选 mask，
但这些 mask 对错误区域的覆盖并不充分，对 clutter/background 漏检也不稳定。
```

当前结果更支持以下定位：

```text
SAM3 = 局部 candidate mask proposal generator
SegFormer = 主体 dense semantic segmentation model
融合方法 = 必须保守筛选 SAM3 mask 后再做局部 refinement
```

不建议直接采用：

```text
SAM3 mask 覆盖哪里，就把哪里改成 SAM3 prompt 类别。
```

因为 class 0 大 mask 语义精度偏低，容易造成大范围误修正。

## 13. 后续建议

下一步建议开展：

```text
Exp5-D：SAM3 candidate mask 筛选与保守 refinement 规则验证
```

初始规则建议：

```text
只在 SegFormer candidate region 内修正；
优先考虑 building / car 等高精度小目标或结构化对象；
限制超大 impervious surface mask 的直接修正；
结合 SegFormer confidence / entropy / margin 作为门控；
clutter/background 不由 SAM3 直接预测，而通过低置信、高熵、无可靠前景覆盖等条件推断。
```
