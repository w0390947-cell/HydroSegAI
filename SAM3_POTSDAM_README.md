# SAM3 零样本评估 - Potsdam 数据集

## 📋 概述

这个脚本用于测试 SAM3 对 Potsdam 数据集 6 类地物的零样本分割能力。

**评估目标**：
- 验证 SAM3 在遥感图像语义分割任务中的零样本性能
- 测试开放词汇文本提示在不同地物类别上的表现
- 为后续的 SegFormer + SAM3 融合实验提供基线对比

## 🎯 Potsdam 类别与文本提示映射

| 类别ID | 类别名称 | SAM3 文本提示 | 颜色 |
|--------|----------|---------------|------|
| 0 | 杂乱/背景 | `background clutter` | 白色 (255,255,255) |
| 1 | 不透水表面 | `impervious surface` | 灰色 (0,0,0) |
| 2 | 建筑物 | `building` | 蓝色 (0,0,255) |
| 3 | 低矮植被 | `low vegetation` | 青色 (0,255,255) |
| 4 | 树木 | `tree` | 绿色 (0,255,0) |
| 5 | 汽车 | `car` | 黄色 (255,255,0) |

## 🚀 快速开始

### 1. 基础使用

```bash
# 激活虚拟环境
source .venv_hf/bin/activate

# 运行评估脚本
python sam3_potsdam_evaluation.py
```

### 2. 首次运行检查清单

确保以下条件满足：

- [x] SAM3 权重文件存在：`sam3/checkpoints/sam3.1_multiplex.pt`
- [x] Potsdam 数据集在正确位置：`Potsdam/`
- [x] Python 依赖已安装：torch, PIL, opencv, matplotlib 等
- [x] GPU 可用（推荐，16GB 显存）

### 3. 环境配置

如果遇到依赖问题，请安装：

```bash
pip install torch torchvision opencv-python matplotlib scikit-learn pandas
pip install gdal  # 可选，用于读取 GeoTIFF
```

## 📂 输出结果结构

运行完成后，在 `results/` 目录下会生成：

```
results/
├── predictions/              # 预测结果
│   ├── top_potsdam_2_10_prediction.png
│   ├── top_potsdam_2_10_ground_truth.png
│   └── ...
├── visualizations/           # 可视化对比图
│   ├── top_potsdam_2_10_comparison.png
│   └── ...
├── metrics/                  # 评估指标
│   ├── overall_metrics.json  # 总体指标
│   └── per_image_metrics.csv # 每张图像的详细指标
└── logs/                     # 运行日志
    └── evaluation_log_YYYYMMDD_HHMMSS.txt
```

## 📊 评估指标说明

### 主要指标
- **Overall Accuracy (OA)**: 总体分类精度
- **Mean IoU**: 平均交并比
- **Mean F1**: 平均 F1 分数

### 每类别指标
- **Per-class IoU**: 每个类别的 IoU
- **Per-class F1**: 每个类别的 F1 分数
- **Per-class Precision/Recall**: 精确率和召回率

### 可视化结果
每张测试图像会生成一个对比图，包含：
1. 原始 RGB 图像
2. 真实标签（彩色）
3. SAM3 预测标签（彩色）
4. 预测叠加图（预测标签叠加在原图上）

## ⚙️ 配置参数

### 图像处理参数
```python
PATCH_SIZE = 1008   # SAM3 输入尺寸
STRIDE = 672        # 切片步长（重叠度约 1/3）
SCORE_THRESHOLD = 0.5  # mask 置信度阈值
```

### 测试图像选择
快速验证阶段使用 3 张代表性图像：
- `top_potsdam_2_10`: 简单城市场景
- `top_potsdam_5_11`: 中等复杂度场景
- `top_potsdam_7_9`: 复杂密集建筑场景

## 🔧 自定义配置

### 修改测试图像
编辑 `main()` 函数中的 `TEST_IMAGES` 列表：

```python
TEST_IMAGES = [
    "top_potsdam_2_10",
    "top_potsdam_3_11",  # 添加新图像
    # ...
]
```

### 调整文本提示
修改 `CLASS_INFO` 字典中的 `prompt` 字段：

```python
CLASS_INFO = {
    1: {"name": "impervious surface", "prompt": "road and pavement", ...},
    # ...
}
```

### 修改融合策略
在 `fuse_multiclass_masks()` 方法中调整融合逻辑。

### 切片参数调整
```python
# 更高质量（更慢）
PATCH_SIZE = 1008
STRIDE = 504  # 50% 重叠

# 更快速度
PATCH_SIZE = 1008
STRIDE = 1008  # 无重叠
```

## 📈 预期结果

### 零样本基线性能
基于 SAM3 的设计，预期在 Potsdam 数据集上的表现：

- **Overall Accuracy**: 40-60%
- **Mean IoU**: 25-45%

**性能分析**：
- **建筑物 (Building)**: 预期较好（SAM 在建筑物检测上训练充分）
- **汽车 (Car)**: 预期中等（SAM 有车辆概念）
- **植被 (Tree/Low Vegetation)**: 预期一般到中等
- **不透水面 (Impervious)**: 预期较差（需要更多道路概念）

### 可能遇到的问题
1. **类别混淆**: tree vs low vegetation
2. **背景误判**: clutter 被误分为其他类别
3. **小目标漏检**: car 类别召回率可能较低

## 🚨 故障排除

### 常见错误

#### 1. CUDA 内存不足
```bash
# 解决方案：减小 batch size 或使用 CPU
# 修改代码中的设备设置
self.device = "cpu"  # 强制使用 CPU
```

#### 2. 文件读取错误
```bash
# 确保数据路径正确
ls Potsdam/2_Ortho_RGB/2_Ortho_RGB/
ls Potsdam/5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary/
```

#### 3. GDAL 导入错误
```bash
# 安装 GDAL
pip install gdal
# 或者让代码自动使用 PIL 作为替代
```

#### 4. SAM3 模型加载失败
```bash
# 检查权重文件
ls -lh sam3/checkpoints/sam3.1_multiplex.pt

# 重新安装 SAM3
cd sam3
pip install -e .
```

## 🎯 下一步计划

完成零样本评估后，建议的后续实验：

### 阶段 2：提示工程优化
- 测试不同的文本提示变体
- 尝试组合提示（如 "building and house"）
- 调整置信度阈值

### 阶段 3：SegFormer + SAM3 融合
```python
# 使用 SegFormer 生成粗分割图
coarse_map = segformer_model(image)

# 基于 coarse_map 生成 box prompt
boxes = generate_boxes_from_coarse_map(coarse_map)

# SAM3 进行边界细化
refined_result = sam3_refine(image, boxes, text_prompts)
```

### 阶段 4：完整数据集评估
- 扩展到所有 24 张有标签图像
- 与其他方法对比
- 撰写实验报告

## 📚 参考资料

- SAM3 官方文档: `sam3/README.md`
- Potsdam 数据集文档: `Potsdam/Potsdam_Dataset_Documentation.md`
- 项目架构说明: `简短总结的架构.md`

## 💡 使用建议

1. **首次运行**: 建议先用 1 张图像测试，确保流程正常
2. **GPU 监控**: 使用 `nvidia-smi` 监控显存使用
3. **日志查看**: 查看 `logs/` 目录中的运行日志
4. **结果分析**: 重点关注可视化结果，理解模型行为

## 📞 支持

如有问题，请检查：
1. 运行日志文件
2. SAM3 和 Potsdam 数据集文档
3. 环境配置是否正确

---

**祝实验顺利！** 🚀