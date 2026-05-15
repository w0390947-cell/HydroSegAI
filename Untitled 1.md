# 面向水利智能感知的 SAM3 推理分割方案整理

## 1. 研究定位

本文建议聚焦为：

**面向水利工程智能感知的 SAM3 概念驱动推理分割方法研究**

或者：

**基于 SAM3 开放词汇分割的水利遥感监测与灾害识别方法研究**

核心观点是：水利场景中的目标分割不是单纯的视觉像素分类问题，而是需要结合任务语义、水文水力知识、地形约束、工程结构语义和灾害机理进行推理的复杂感知问题。

传统语义分割模型通常回答：

```text
这个像素属于哪一类？
```

SAM3 更适合回答：

```text
在当前文本概念或视觉提示下，图像/视频中哪些实例符合该概念？
```

因此，本文不应再按 “SAM2 + LoRA + 二值 mask” 来组织，而应围绕 SAM3 的 **开放词汇概念分割、文本提示、视觉提示、检测-分割联合建模、视频跟踪和多轮交互** 展开。

## 2. 从 SAM2 思路切换到 SAM3 思路

原来的 SAM2/SAM2.1 方案更偏向：

```text
人工或模型生成 point / box prompt
→ SAM2 输出目标 mask
→ LoRA 微调 image encoder 适配水利遥感域
```

SAM3/SAM3.1 的重点发生了变化：

```text
文本概念 / 视觉示例 / 点 / 框 / mask prompt
→ SAM3 检测所有符合概念的实例
→ 同时输出 boxes、masks、scores
→ 视频中进一步跟踪目标
```

也就是说，SAM3 不只是一个交互式 mask 细化器，而是一个 **以概念为中心的开放词汇实例检测与分割模型**。

对水利任务而言，可以直接用自然语言概念描述目标，例如：

```text
flooded road
river water
reservoir water surface
dam crack
seepage area
bare landslide scar
blocked river channel
building
tree
car
impervious surface
low vegetation
```

然后由 SAM3 在图像或视频中寻找并分割所有符合该概念的实例。

## 3. SAM3 模型架构理解

根据当前工作区中的 `sam3` 官方仓库，SAM3 的总体结构可以概括为：

```text
输入图像 / 视频
        │
        ▼
视觉编码器 Vision Encoder
        │
        ├───────────────┐
        ▼               ▼
文本/视觉/几何提示编码  图像特征
        │               │
        └───────融合────┘
                │
                ▼
开放词汇检测器 Detector
        │
        ├── boxes
        ├── concept scores
        ▼
分割头 Segmentation Head
        │
        ├── masks
        └── mask scores
```

视频任务中：

```text
检测结果 / prompt
        │
        ▼
Tracker / Object Multiplex
        │
        └── 跨帧目标跟踪与 mask 传播
```

SAM3.1 的重点是 **Object Multiplex**，即面向多目标视频跟踪的共享记忆机制。对静态遥感图像任务，主要使用 SAM3 image model；对洪水演进、无人机巡检视频、河道动态监测等任务，可以使用 video predictor。

## 4. 水利推理分割总体框架

建议把文章中的方法框架写成五层。

### 第一层：多源数据输入层

输入不应只写 RGB 图像，而应覆盖水利场景的多源数据：

```text
遥感 RGB 影像
无人机巡检图像
视频帧
SAR 影像
多光谱 / 高光谱影像
DEM / DSM / nDSM
GIS 水系矢量
河道、堤防、库区边界
历史水体边界
水位、流量、降雨时序
工程设计图纸与巡检记录
用户自然语言问题
```

SAM3 主要处理图像和视频视觉输入；DEM、GIS 和水文时序更适合作为 prompt 生成、结果校正和解释模块的外部知识。

### 第二层：任务理解与概念构造层

这一层可以由大语言模型或规则系统完成，核心任务是把水利问题转成 SAM3 可执行的概念提示。

示例：

```text
用户问题：找出可能由河道漫溢造成的新增积水区。

概念拆解：
1. current water
2. river water
3. flooded farmland
4. flooded road
5. permanent water body
6. dark roof or shadow
```

再结合历史水体、DEM 和河网关系，形成推理规则：

```text
新增淹没区 = 当前水体 - 历史常水位水体
优先保留与河道、沟渠、低洼区连通的区域
排除高程异常高、与水系完全不连通的暗色区域
```

### 第三层：SAM3 概念驱动检测与分割层

SAM3 接收文本或视觉提示后，输出：

```text
masks
boxes
scores
```

在水利图像中，可以把 SAM3 用作：

```text
开放词汇目标发现器
概念级实例分割器
文本提示驱动的候选区域生成器
交互式修正工具
视频目标跟踪器
```

与 SAM2 不同，SAM3 不必完全依赖外部 box prompt。它可以直接用文本概念触发检测和分割，这是本文方法的关键优势。

### 第四层：水利知识约束与融合层

SAM3 输出的 mask 仍然需要专业约束校正。

建议加入：

```text
水系连通性约束
DEM / DSM 高程约束
坡度约束
河道缓冲区约束
库区边界约束
历史水体差分
洪水演进时间一致性
工程结构拓扑约束
面积、形状、连通域过滤
```

例如，SAM3 可能把阴影、深色屋顶或湿地识别为水体。水利约束层应判断该区域是否：

```text
与河道或沟渠连通
位于低洼地带
符合近期水位和降雨过程
不属于历史永久水体
不在明显高程建筑物顶部
```

### 第五层：结果解释与决策输出层

最终输出不应只有 mask，还应包括：

```text
分割图
实例边界框
面积统计
类别或概念置信度
风险等级
水利约束检查结果
可解释文本
工程建议
```

示例解释：

```text
该区域被判定为新增淹没区，因为其在当前影像中被 SAM3 识别为 water，
不属于历史常水位水体，位于 DEM 低洼区，并与主河道缓冲区连通。
```

## 5. 典型应用场景

### 场景一：洪水淹没范围提取

目标概念可以设计为：

```text
flood water
flooded road
flooded farmland
river water
standing water
permanent water body
shadow
dark roof
```

处理流程：

```text
当前影像
→ SAM3 按 water / flooded road / flooded farmland 等概念生成 masks
→ 与历史水体边界差分
→ DEM 低洼区和河网连通性约束
→ 输出新增淹没范围、面积和风险对象
```

### 场景二：河道、水体与岸线动态识别

目标概念：

```text
main river
tributary
reservoir water
pond
wetland
sandbar
river bank
shoreline
```

SAM3 可根据文本概念生成候选 mask，再通过 GIS 河网和历史边界筛选主河道、水库和非目标水体。

### 场景三：堤坝、渠道、闸门等水工结构缺陷识别

目标概念：

```text
dam crack
concrete crack
seepage area
spalling concrete
exposed rebar
slope failure
erosion scar
```

这类任务更依赖近景巡检图像或无人机图像。SAM3 可先用文本概念定位疑似缺陷，再用点、框或示例图进行交互式细化。

### 场景四：水库库区与消落带识别

目标概念：

```text
reservoir water surface
drawdown zone
exposed shoreline
mudflat
vegetated bank
```

推理重点是结合水位、库容曲线、DEM 和历史水面线，判断当前水面边界和消落带范围是否合理。

### 场景五：山洪、泥石流、滑坡灾害隐患识别

目标概念：

```text
landslide scar
bare soil
debris flow deposit
blocked river channel
collapsed slope
temporary dam
```

SAM3 可用于开放词汇候选目标发现，再结合沟道形态、坡度、河网位置和历史灾害记录进行风险判断。

## 6. Potsdam 数据集下的 SAM3 适配方案

Potsdam 是城市高分辨率遥感语义分割数据集，当前工作区已有：

```text
Potsdam/2_Ortho_RGB/2_Ortho_RGB
Potsdam/3_Ortho_IRRG/3_Ortho_IRRG
Potsdam/4_Ortho_RGBIR/4_Ortho_RGBIR
Potsdam/1_DSM/1_DSM
Potsdam/5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary
```

Potsdam 类别可以映射成 SAM3 文本概念：

| class id | Potsdam 类别 | SAM3 text prompt |
| --- | --- | --- |
| 0 | clutter / background | background clutter |
| 1 | impervious surface | impervious surface |
| 2 | building | building |
| 3 | low vegetation | low vegetation |
| 4 | tree | tree |
| 5 | car | car |

### 6.1 为什么不能照搬 SAM2-LoRA 方案

SAM2 方案通常是：

```text
GT mask → box prompt → SAM2 输出二值 mask → LoRA 学边界
```

而 SAM3 原生支持文本概念检测与分割，所以 Potsdam 更适合改成：

```text
RGB patch
→ 对每个类别输入 text prompt
→ SAM3 输出该概念的 boxes / masks / scores
→ 多类别 mask 融合
→ 计算 mIoU / mF1 / OA
```

如果要进一步微调，则应围绕 SAM3 的官方训练框架，把 Potsdam 转成开放词汇检测/分割格式，而不是只做 SAM2 式的二值 prompt mask。

### 6.2 推荐的第一版实验：SAM3 零样本评估

第一版不要急着微调，先验证 SAM3 对 Potsdam 类别的零样本能力。

流程：

```text
1. 切分 Potsdam 6000×6000 大图为 1008 或 1024 patch
2. 对每个 patch 分别输入 6 个文本概念
3. 收集 SAM3 输出的 masks、boxes、scores
4. 将实例 masks 合成为 6 类语义图
5. 与 Potsdam 标签计算 OA、mIoU、mF1、per-class IoU
```

建议 prompt：

```text
building
tree
car
low vegetation
impervious surface
background clutter
```

可以做 prompt 消融：

```text
building vs rooftop vs building roof
tree vs tall tree
low vegetation vs grass
impervious surface vs road and pavement
car vs vehicle
```

### 6.3 推荐的第二版实验：SAM3 + 粗分割融合

SAM3 是开放词汇实例分割模型，但 Potsdam 的评价目标是 6 类语义分割。为了让结果更稳，建议使用粗语义分割模型提供类别先验。

推荐框架：

```text
RGB patch
→ SegFormer / DeepLabV3+ 输出 coarse 6 类语义图
→ 每类连通域生成候选 box
→ SAM3 使用 text prompt + box prompt 细化实例 mask
→ 结合 coarse logits 和 SAM3 scores 融合
→ 输出最终 6 类语义图
```

这种定位更合理：

```text
SegFormer 负责类别判别
SAM3 负责开放词汇概念确认和边界细化
DSM / nDSM 负责高度先验校正
```

### 6.4 推荐的第三版实验：SAM3 微调

如果零样本和融合实验效果不足，再考虑微调。

SAM3 官方训练入口在：

```text
sam3/sam3/train/train.py
```

训练文档在：

```text
sam3/README_TRAIN.md
```

SAM3 的训练配置面向开放词汇检测/分割数据，Potsdam 需要先转成类似 COCO/ODinW/Roboflow 风格的数据：

```text
image
annotations:
  bbox
  segmentation
  category / phrase
```

每个 Potsdam patch 中的类别连通域可以转成实例级 annotation：

```text
building 实例 mask
tree 实例 mask
car 实例 mask
impervious surface 区域 mask
low vegetation 区域 mask
```

注意：Potsdam 本身是语义标签，不是严格实例标签。建筑物、树木、车辆可以通过连通域近似构造实例；impervious surface 和 low vegetation 往往是大面积区域，实例边界不一定有明确语义。因此第一版微调应谨慎。

## 7. Potsdam 数据预处理流程

### 7.1 标签颜色映射

Potsdam 标签颜色可映射为：

| class id | 类别 | RGB |
| --- | --- | --- |
| 0 | clutter / background | 255,255,255 |
| 1 | impervious surface | 0,0,0 |
| 2 | building | 0,0,255 |
| 3 | low vegetation | 0,255,255 |
| 4 | tree | 0,255,0 |
| 5 | car | 255,255,0 |
| 255 | boundary / ignore | 255,0,0 |

建议优先使用：

```text
Potsdam/5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary
```

如果使用含 boundary 标签，则红色边界区域应设置为 ignore index 255。

### 7.2 切片建议

Potsdam 单图为 6000×6000。SAM3 的图像尺寸设置中常见输入大小为 1008，因此建议：

```text
patch_size = 1008 或 1024
stride = 504 或 512
```

如果显存不足：

```text
patch_size = 512
stride = 384
```

### 7.3 数据划分建议

推荐保持空间分离，避免相邻 patch 泄漏。

训练瓦片：

```text
2_10, 2_11, 2_12
3_10, 3_11, 3_12
4_10, 4_11, 4_12
5_10, 5_11, 5_12
6_7, 6_8, 6_9, 6_10
```

验证瓦片：

```text
6_11, 6_12
7_7, 7_8, 7_9, 7_10, 7_11, 7_12
```

## 8. SAM3 环境与权重

当前工作区已有：

```text
sam3/
```

这是 SAM3/SAM3.1 官方代码。SAM3 README 推荐：

```text
Python >= 3.12
PyTorch >= 2.7
CUDA >= 12.6
```

建议环境：

```bash
conda create -n sam3 python=3.12 -y
conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
cd sam3
pip install -e .
pip install -e ".[train]"
```

SAM3.1 权重位于 Hugging Face：

```text
facebook/sam3.1
```

需要先申请访问权限并登录：

```bash
pip install -U huggingface_hub
hf auth login
hf auth whoami
```

PowerShell 下载命令：

```powershell
mkdir D:\TestEnvironment\PythonENV2\Test_11\sam3\checkpoints

hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir D:\TestEnvironment\PythonENV2\Test_11\sam3\checkpoints

hf download facebook/sam3.1 config.json --local-dir D:\TestEnvironment\PythonENV2\Test_11\sam3\checkpoints
```

如果出现：

```text
Access denied. This repository requires approval.
```

说明 Hugging Face 账号尚未获得 `facebook/sam3.1` 仓库访问权限，需要在网页端申请并等待通过。

## 9. SAM3 推理流程

基本图像推理逻辑：

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)

image = Image.open("your_image.jpg")
state = processor.set_image(image)
output = processor.set_text_prompt(state=state, prompt="building")

masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]
```

如果使用本地权重，思路是显式传入 checkpoint：

```python
model = build_sam3_image_model(
    checkpoint_path="checkpoints/sam3.1_multiplex.pt",
    load_from_HF=False,
)
```

实际函数参数需以当前 `sam3/sam3/model_builder.py` 为准。

## 10. 实验设计

### 10.1 对比实验

建议设置：

| 方法 | 说明 |
| --- | --- |
| U-Net | 传统语义分割基线 |
| DeepLabV3+ | CNN 语义分割基线 |
| SegFormer | Transformer 遥感分割强基线 |
| SAM2 / SAM2.1 zero-shot | 旧版 promptable segmentation 对比 |
| SAM3 zero-shot text prompt | 开放词汇文本概念分割 |
| SAM3 text + box prompt | 加入候选框提示 |
| SegFormer + SAM3 refinement | 推荐最终方案 |
| SAM3 fine-tune | 可选进阶方案 |

### 10.2 消融实验

建议做：

```text
文本 prompt 消融
RGB / IRRG 输入消融
patch size 消融
score threshold 消融
SAM3 zero-shot vs SAM3 + coarse prompt
是否加入 DSM/nDSM 后处理
是否加入水利知识约束
```

### 10.3 评价指标

Potsdam 语义分割指标：

```text
OA
mIoU
mF1
per-class IoU
per-class F1
```

水利场景额外指标：

```text
水面面积相对误差
淹没区召回率
重点建筑物漏检率
河道连通性一致率
岸线边界误差
风险区召回率
```

## 11. 技术路线总览

建议把论文方法图画成：

```text
多源数据输入
RGB / UAV / video / SAR / DEM / GIS / 水文时序 / 用户问题
        │
        ▼
任务理解与概念生成
水利问题 → SAM3 text prompt / visual prompt / box prompt
        │
        ▼
SAM3 开放词汇检测与分割
输出 boxes / masks / scores
        │
        ▼
水利知识约束校正
DEM / 水系连通性 / 历史水体 / 工程边界 / 时序一致性
        │
        ▼
结果融合与解释
分割图 / 面积统计 / 风险等级 / 解释文本 / 工程建议
```

Potsdam 实验路线：

```text
Potsdam RGB + noBoundary label
→ 标签颜色转 class id
→ 大图切 patch
→ SAM3 text prompt 零样本分割
→ 多类别 mask 融合
→ mIoU / mF1 / OA 评价
→ 引入 SegFormer coarse map 生成 box prompt
→ SAM3 边界细化
→ 与基线模型对比
```

## 12. 论文创新点建议

### 创新点一：从语义分割转向概念驱动推理分割

传统遥感语义分割依赖固定类别，SAM3 允许通过自然语言概念动态指定目标，更适合水利应急场景中的开放目标识别。

### 创新点二：水利知识增强的 SAM3 结果校正

将 SAM3 输出与 DEM、水系连通性、历史水体边界、水位过程线和工程边界结合，减少阴影、屋顶、湿地等误检。

### 创新点三：粗语义模型与 SAM3 边界细化融合

使用 SegFormer 等模型负责稳定类别判别，SAM3 负责开放词汇概念确认和高质量 mask 细化。

### 创新点四：可解释分割输出

系统不仅输出 mask，还输出为什么该区域符合某个水利概念，增强工程应用可信度。

## 13. 当前最小可行方案

建议第一阶段按以下顺序推进：

```text
1. 配好 SAM3 环境
2. 获得 facebook/sam3.1 权重访问权限
3. 跑通 SAM3 image predictor 示例
4. 从 Potsdam 中选 1-2 张 RGB 瓦片
5. 切成 1008 或 1024 patch
6. 对 building / tree / car 等概念做 zero-shot 推理
7. 可视化 masks、boxes、scores
8. 与标签计算初步 IoU
9. 调整 prompt 和 score threshold
10. 再考虑 SegFormer + SAM3 refinement 或 SAM3 微调
```

最重要的一点是：

**SAM3 的优势不是复刻 SAM2-LoRA 的二值 mask 微调，而是利用开放词汇概念分割能力，把水利专业语义以文本和视觉提示的方式直接接入分割流程。**
