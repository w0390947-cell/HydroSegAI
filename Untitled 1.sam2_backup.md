你的思路是合理的，而且很适合写成一篇“水利领域智能感知 + 多模态推理”的应用型文章。但当前方案已经改用 SAM3，因此需要把原来的 SAM2-LoRA 叙述重整为 SAM3 的开放词汇概念分割、检测-分割联合建模与视频跟踪架构。

## 1. 文章定位建议

建议题目不要太泛，可以聚焦为：

**“面向水利工程智能感知的推理分割方法研究：大模型推理与专用图像分割模型融合框架”**

或者更应用化一点：

**“大模型驱动的推理分割在水利工程遥感监测与灾害识别中的应用展望”**

你的核心卖点应是：传统分割模型只能“看图分割”，而推理分割系统能够根据任务语义、上下文知识和专业约束，判断“应该分割什么、为什么分割、如何修正分割结果”。

近年来确实已经出现了“LLM/VLM + segmentation”的方向，例如 LISA、LISA++、SegLLM、LLM-Seg、RSVP 等工作，核心都是让大模型参与复杂语言指令理解、目标定位和分割掩膜生成。SegLLM 特别强调多轮交互和视觉-文本记忆，LLM-Seg 则强调把图像分割和大语言模型推理连接起来。([OpenReview][1])

## 2. 推荐的总体架构

你提出的“大模型负责推理 + 图像分割专用模型负责分割”可以进一步拆成四层：

### 第一层：多源数据输入层

水利领域不要只考虑普通 RGB 图像，建议写成多源输入：

遥感影像、无人机影像、巡检图像、视频帧、SAR 影像、DEM/DSM、高程数据、水位流量数据、降雨数据、历史洪水淹没图、工程设计图纸、GIS 矢量边界等。

这样文章会更像水利专业文章，而不是计算机视觉泛化文章。洪水、水体和水利设施监测中，遥感和深度学习已经被广泛用于水体、水淹区域、库区变化等任务；例如已有研究把遥感影像与水利工程语义结合用于水库提取和监测。([MDPI][2])

### 第二层：大模型推理层

大模型不要只写“生成提示词”，而应承担这些职责：

1. **任务解析**：把自然语言问题转化为分割任务。
   例如：“找出可能受洪水影响的低洼农田和被淹道路”。

2. **专业知识约束**：结合水文学、水力学、地形坡度、河道连通性判断分割目标是否合理。
   例如：被识别为“洪水”的区域如果与河道、水系、低洼区完全不连通，就可能是假阳性。

3. **分割提示生成**：为 SAM、SAM2、Mask2Former、UNet、DeepLab、SegFormer 等模型生成 box、point、text prompt 或候选区域。

4. **结果解释与修正**：对分割结果进行逻辑检查。
   例如：“该区域虽然呈蓝色，但位于屋顶阴影区，不应判定为水体”。

5. **多轮交互**：用户可以继续问：“只保留主河道，不要池塘”“把堤防缺损区域单独标出”。这正是推理分割相比传统分割的优势。

### 第三层：专用分割模型层

这里建议不要只写 SAM。可以分三类：

| 模型类型                                        | 适合任务                     | 建议写法               |
| ------------------------------------------- | ------------------------ | ------------------ |
| SAM / SAM2 / MobileSAM                      | 通用交互式分割、水体轮廓、堤岸、建筑物      | 适合作为基础分割器，但水利场景需微调 |
| UNet / DeepLabV3+ / SegFormer / Mask2Former | 洪水淹没、水体、滑坡、泥沙、裂缝、渗漏等语义分割 | 适合有标注数据的专业场景       |
| SAR / 多光谱 / 高光谱分割模型                         | 暴雨、云层遮挡、夜间洪水、水体识别        | 适合遥感水利监测           |

SAM 已经被用于洪水淹没制图和水体提取场景，例如 ArcGIS 示例展示了用 SamLoRA 微调 SAM 来提取洪水淹没区域，并使用 Sentinel-1 SAR 数据进行洪水监测。([ArcGIS Developers][3]) 这可以作为你文章里“通用分割模型需要水利领域适配”的依据。

### 第四层：水利专业后处理与决策层

这是最容易体现专业性的地方。建议加入：

水系连通性约束、DEM 高程约束、坡度约束、河道缓冲区约束、库区边界约束、历史水位线约束、洪水演进时间一致性约束、工程结构拓扑约束。

例如：

分割模型识别出一片“积水区”，大模型结合 DEM 判断其是否位于低洼区；结合水系拓扑判断是否与河道漫溢路径一致；结合降雨和水位数据判断是否符合洪水发生条件。这样就从“图像分割”提升为“水利推理分割”。

## 3. 最适合写的应用场景

我建议你不要泛泛列很多场景，而是重点写 4–6 个最有代表性的。

### 场景一：洪水淹没范围智能提取

这是最适合推理分割的场景。原因是洪水分割不仅看颜色，还要考虑地形、河网、道路、建筑、阴影、云雾、SAR 后向散射等因素。

传统模型容易把阴影、湿地、稻田、暗色屋顶误分为洪水。推理分割可以利用大模型判断：“该区域是否与河道连通？是否位于低洼区？是否符合洪水传播方向？”

已有工作正在探索 SAM2 用于遥感洪水检测，也有大量深度学习方法用于洪水区域分割。([ResearchGate][4])

### 场景二：河道、水体与岸线动态识别

可用于河道演变、水库水面面积变化、湖泊萎缩、河岸侵蚀监测。

这里的推理点是：
不是所有蓝色/暗色区域都是水体；不是所有水体都属于目标河道。大模型可以根据“主河道、支流、库区、滩涂、鱼塘、湿地”等水利语义进行筛选。

### 场景三：堤坝、渠道、闸门等水工建筑物缺陷分割

可以分割裂缝、渗水、剥蚀、露筋、沉陷、边坡滑塌、混凝土破损等。

这个方向和普通遥感水体分割不同，更偏“水利工程安全监测”。2025 年已有综述关注水工结构图像中的深度学习缺陷识别，并讨论了人工巡检、无人机、机器人、声呐/水下图像等多种图像采集方式。([ScienceDirect][5])

### 场景四：水库库区与消落带识别

水库调度、生态监测、水资源管理都需要库区边界、水面面积、消落带范围识别。

推理分割可以结合水位数据、历史库容曲线、岸坡高程、季节变化，判断当前水面边界是否合理。

### 场景五：山洪、泥石流、滑坡灾害隐患识别

水利专业常涉及山洪沟、沟道堵塞、滑坡堰塞体、河道阻塞物等。大模型可以根据“沟道形态 + 植被破坏 + 裸露土体 + 河道阻塞”进行语义推理，再交给分割模型细化边界。

已有研究将深度学习和遥感用于全球河道阻塞物快速检测，这说明“水利灾害目标识别 + 分割”有很强应用价值。([AgUpubs][6])

## 4. 文章可以提出的创新点

你可以把创新点设计成以下几类。

### 创新点一：从“语义分割”转向“推理分割”

传统语义分割回答的是：

“这个像素属于哪一类？”

推理分割回答的是：

“在当前水利任务语境下，哪些区域符合某种工程或灾害条件？”

比如“找出可能由河道漫溢造成的积水区域”，这就不是普通水体分割，而是结合水文机制的推理分割。

### 创新点二：引入水利专业知识约束

建议明确提出：

**水利知识增强的推理分割框架**

知识可以包括：

水系拓扑、河流连通性、DEM 高程、水位过程线、降雨时序、洪水传播方向、堤防工程边界、库区调度规则、历史淹没范围。

这会让文章明显区别于普通“VLM + SAM”综述。

### 创新点三：多模态融合

文章不要只写“图像 + 文本”，而应写成：

图像 + 文本指令 + GIS + DEM + 水文数据 + 工程资料。

例如：

用户输入：“请识别该遥感影像中超出常水位边界的新增淹没区，并排除永久水体。”

系统流程：

大模型理解“新增淹没区”和“永久水体”的区别；调用历史水体边界和当前影像；分割模型生成当前水体掩膜；后处理模块计算新增淹没范围；大模型解释结果。

### 创新点四：可解释性

水利工程应用非常看重可信性。你可以强调推理分割不仅输出 mask，还输出解释：

“该区域被判断为洪水淹没区，因为其位于河道下游低洼地带，与主河道连通，且在近期高水位过程中由非水体转变为水体。”

这比单纯 IoU 更符合工程决策需求。

## 5. 技术路线建议

你可以在文章中画一个框架图，流程如下：

**输入数据**
遥感影像 / 无人机图像 / 巡检图像 / DEM / GIS / 水文时序 / 用户问题

↓

**大模型推理模块**
任务理解、目标定义、专业知识调用、分割提示生成

↓

**图像分割模块**
SAM/SAM2/SegFormer/UNet/Mask2Former 生成初始 mask

↓

**水利约束校正模块**
连通性、高程、坡度、水系、历史边界、时序一致性检查

↓

**结果输出模块**
分割图、面积统计、风险等级、解释文本、工程建议

这个架构比“大模型 + 分割模型”更完整，也更像能落地的水利专业系统。

## 6. 实验设计建议

文章最好不要只写理论框架，建议设计几个实验任务：

| 实验任务      | 数据                              | 指标                      |
| --------- | ------------------------------- | ----------------------- |
| 洪水淹没区分割   | Sentinel-1 SAR、Sentinel-2、无人机影像 | IoU、F1、Precision、Recall |
| 河道水体提取    | 多时相遥感影像                         | mIoU、边界 F1、水面面积误差       |
| 堤坝裂缝/渗漏分割 | 巡检图像、无人机图像                      | Dice、IoU、缺陷识别率          |
| 库区消落带识别   | 遥感 + 水位数据 + DEM                 | 面积误差、边界误差               |
| 复杂语言指令分割  | 自建水利指令数据集                       | 指令理解准确率、mask 质量、人工评分    |

建议必须加入对比实验：

1. 传统分割模型：UNet、DeepLabV3+、SegFormer。
2. 通用 SAM 或 SAM2。
3. 只用大模型提示 + SAM。
4. 你的“水利知识增强推理分割框架”。

这样才能证明大模型推理层确实有用。

## 7. 需要注意的问题

第一个问题是：**大模型容易幻觉**。
水利工程是严肃领域，不能让大模型凭空判断“堤坝有风险”或“该区域会被淹”。所以大模型应主要负责推理、解释和调度，不应直接替代水文水动力模型。

第二个问题是：**SAM 类模型并不天然适合水利场景**。
SAM 对自然图像能力强，但对 SAR 影像、浑浊水体、消落带、阴影、浅滩、植被覆盖水面等场景可能效果不稳定。因此建议写“需要 LoRA 微调、领域适配或与专用分割网络融合”。

第三个问题是：**水利场景边界本身具有模糊性**。
例如洪水边界、湿地边界、浅水区边界、泥沙水体边界都不一定有清晰视觉边缘。文章应强调不确定性估计，而不是只输出一个确定 mask。

第四个问题是：**评价指标不能只用 IoU**。
水利应用还关心面积误差、淹没深度估计误差、是否漏检关键建筑物、是否误判重要工程设施。因此建议加入工程指标，例如：

水面面积相对误差、受灾建筑漏检率、河道连通性一致率、库容估算误差、风险区召回率。

第五个问题是：**需要构建水利领域指令数据集**。
普通分割数据集只有类别标签，不能训练“推理分割”。你需要设计类似这样的指令：

“请分割出可能由河道漫溢形成的新增积水区。”
“请只保留主河道水体，不包括鱼塘和水库。”
“请识别堤坝迎水坡上疑似渗漏区域。”
“请分割库区消落带，不包括永久水面。”
“请找出阻塞河道行洪断面的堆积物。”

这会成为文章中非常重要的建设性内容。

## 8. 我认为最值得写的文章主线

建议你把文章主线写成：

**水利图像分割任务具有强专业语义、强空间约束和强机理背景，传统分割模型难以处理复杂指令和专业判别。推理分割通过引入大模型，将自然语言理解、专业知识推理、视觉提示生成和分割结果解释结合起来，可以提升洪水监测、水体提取、工程缺陷识别和水利灾害识别的智能化水平。**

文章结构可以这样安排：

1. 引言：水利智能感知需求与传统分割局限。
2. 推理分割概念：从语义分割到语言驱动、知识增强的推理分割。
3. 模型架构：大模型推理层 + 专用图像分割层 + 水利知识约束层。
4. 典型应用：洪水、水体、堤坝、库区、灾害隐患。
5. 关键技术：多模态数据融合、领域提示、SAM 微调、知识约束、可解释输出。
6. 挑战：数据集、泛化性、幻觉、实时性、评价体系。
7. 展望：水利数字孪生、智能巡检、应急决策支持。

## 9. 我的总体看法

这个选题有潜力，但要避免写成“把大模型和 SAM 拼在一起”。真正有价值的方向是：

**让大模型理解水利任务，让分割模型完成像素级边界，让水利知识约束结果，让系统输出可解释、可验证、可用于工程决策的分割结果。**

也就是说，文章的重点不应只是“模型架构”，而应是：

**水利专业知识如何进入推理分割流程。**

[1]: https://openreview.net/forum?id=Pm1NXHgzyf&utm_source=chatgpt.com "SegLLM: Multi-round Reasoning Segmentation with Large ..."
[2]: https://www.mdpi.com/2072-4292/16/2/328?utm_source=chatgpt.com "Combining Satellite Images and the Hydraulic Engineering ..."
[3]: https://developers.arcgis.com/python/latest/samples/finetuning-sam-for-flood-inundation-mapping/?utm_source=chatgpt.com "Fine-tuning Segment Anything Model (SAM) for flood ..."
[4]: https://www.researchgate.net/publication/392959657_Segment_Anything_Model_2_SAM_2_for_Accurate_Flood_Detection_in_Remote_Sensing_Imagery?utm_source=chatgpt.com "Segment Anything Model 2 (SAM 2) for Accurate Flood ..."
[5]: https://www.sciencedirect.com/science/article/pii/S1226798825005252?utm_source=chatgpt.com "Deep learning-based defect identification in hydraulic ..."
[6]: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024WR039692?utm_source=chatgpt.com "Deep Learning and Remote‐Sensed Observations Reveal ..."



下面给你一套**对 SAM2 / SAM2.1 做 LoRA 微调的完整可落地流程**。我会按“水利图像分割任务”来写，例如洪水淹没区、水体、河道、堤坝裂缝、渗漏区域、岸线、消落带等。

SAM2 官方仓库已经提供推理、checkpoint、notebook，并在 2024-09-30 后释放了 SAM2.1 checkpoint 与训练/微调代码；官方要求 Python ≥ 3.10、torch ≥ 2.5.1、torchvision ≥ 0.20.1。SAM2 本身支持图像和视频分割，输入可以是 point、box 或 mask prompt。([GitHub](https://github.com/facebookresearch/sam2))

------

# 一、先明确：为什么用 LoRA 微调 SAM2

SAM2 是通用分割模型，零样本能力强，但在水利专业场景里会遇到明显问题：

比如：

阴影、湿地、稻田、暗色屋顶容易被误识别为水体；
SAR 洪水图像和普通 RGB 图像分布差异大；
浑浊水体、泥沙水体、浅水区边界模糊；
堤坝裂缝、渗漏、混凝土剥蚀属于小目标、细长目标；
消落带、水岸线、河漫滩具有强时空变化。

所以不建议全量微调 SAM2，而建议：

**冻结 SAM2 主体参数，只在部分 Linear 层注入 LoRA，让模型学习水利领域的视觉特征。**

LoRA 的核心思想是冻结原始权重，只训练低秩增量矩阵。已有 SAM2 相关工作也采用 LoRA 适配 encoder、memory encoder、memory attention 等模块，以较少参数完成领域适配。([arXiv](https://arxiv.org/html/2509.12105v1))

------

# 二、推荐训练目标

你要先决定 SAM2-LoRA 微调后希望解决哪类任务。

## 方案 A：单类二值分割

最适合入门，也最适合水利论文实验。

例如：

水体 vs 背景
洪水淹没区 vs 非淹没区
裂缝 vs 非裂缝
渗漏区 vs 非渗漏区
滑坡裸露区 vs 背景

输出 mask 是二值图。

这是最推荐的第一版。

## 方案 B：多类语义分割

例如：

永久水体
新增淹没区
湿地
阴影
建筑物
道路
农田

SAM2 原生不是标准语义分割模型，它更偏 promptable segmentation，所以多类任务一般要转成多个二值 mask：

每个类别单独生成一个 prompt 和 mask，再组合成多类结果。

## 方案 C：视频/时序分割

例如：

洪水演进视频
无人机巡检视频
河道水位变化视频
堤坝裂缝巡检视频

SAM2 的优势在视频，因为它有 memory 机制。官方说明 SAM2 是图像和视频统一分割模型，视频中有 streaming memory 设计。([GitHub](https://github.com/facebookresearch/sam2))

但如果你是第一次做，建议先从**图像 LoRA 微调**做起。

------

# 三、环境准备

建议使用 Linux / Ubuntu。官方也建议 Windows 用户优先使用 WSL + Ubuntu。([GitHub](https://github.com/facebookresearch/sam2))

## 1. 创建 Conda 环境

```bash
conda create -n sam2_lora python=3.10 -y
conda activate sam2_lora
```

## 2. 安装 PyTorch

根据你的 CUDA 版本安装。例如 CUDA 12.1：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果输出 `True`，说明 GPU 可用。

## 3. 安装 SAM2

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

官方安装方式就是 clone 仓库后 `pip install -e .`。([GitHub](https://github.com/facebookresearch/sam2))

如果你需要跑 notebook：

```bash
pip install -e ".[notebooks]"
```

## 4. 安装 LoRA 与训练依赖

```bash
pip install peft accelerate opencv-python pillow matplotlib tqdm scikit-learn albumentations
pip install segmentation-models-pytorch
```

如果你不想用 `peft`，也可以自己写 LoRA 层。建议论文实验阶段自己写，便于说明“LoRA 注入位置”。

------

# 四、下载 SAM2 checkpoint

以 SAM2.1 Hiera Large 为例：

```bash
mkdir -p checkpoints
cd checkpoints

wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

cd ..
```

常见模型有：

```text
sam2.1_hiera_tiny.pt
sam2.1_hiera_small.pt
sam2.1_hiera_base_plus.pt
sam2.1_hiera_large.pt
```

建议：

| 显卡条件     | 推荐模型                  |
| ------------ | ------------------------- |
| 8GB 显存     | sam2.1_hiera_tiny / small |
| 12–16GB 显存 | sam2.1_hiera_base_plus    |
| 24GB 以上    | sam2.1_hiera_large        |
| 论文主实验   | base_plus 或 large        |
| 快速调试     | tiny                      |

------

# 五、准备水利分割数据集

你的数据建议统一成如下结构：

```text
dataset/
  train/
    images/
      0001.jpg
      0002.jpg
    masks/
      0001.png
      0002.png
  val/
    images/
      0101.jpg
      0102.jpg
    masks/
      0101.png
      0102.png
  test/
    images/
      0201.jpg
    masks/
      0201.png
```

mask 要求：

```text
背景 = 0
目标区域 = 255
```

例如：

洪水淹没区为 255，其他区域为 0。

## 标注建议

水利任务里，标注质量非常关键。建议你标注时区分：

永久水体
新增积水
阴影
湿地
稻田
河道
池塘
水库
道路积水
建筑物积水

即使最终训练二值分割，也建议原始标注尽量细，后面可以合并类别。

## 图像尺寸建议

SAM2 默认会做图像预处理，但训练时建议统一 resize：

```text
1024 × 1024
```

如果是裂缝、渗漏等小目标，可以切 patch：

```text
512 × 512
768 × 768
1024 × 1024
```

遥感影像不要直接整幅输入，建议切片：

```text
原始遥感图：10000 × 10000
切片：1024 × 1024
overlap：128 或 256
```

------

# 六、LoRA 应该加在哪里

SAM2 主要由这些部分组成：

```text
image encoder
prompt encoder
mask decoder
memory encoder
memory attention
```

对于图像分割微调，最推荐：

```text
image_encoder 中的 attention q_proj / v_proj
```

或者：

```text
image_encoder 中 q_proj / k_proj / v_proj / out_proj
```

对于视频/时序水利任务，可以进一步加到：

```text
memory_attention
memory_encoder
```

已有 FS-SAM2 工作也选择对 SAM2 的 memory 和 encoder 模块使用 LoRA，而不是从头训练整个模型。([arXiv](https://arxiv.org/html/2509.12105v1))

## 推荐注入策略

### 入门版

只加：

```text
image_encoder attention 的 q_proj 和 v_proj
```

优点：稳定、省显存、不容易过拟合。

### 进阶版

加：

```text
image_encoder attention 的 q_proj, k_proj, v_proj, out_proj
```

优点：表达能力更强。

### 视频版

加：

```text
image_encoder + memory_attention + memory_encoder
```

优点：适合洪水演进、巡检视频、河道动态变化。

### 不建议一开始训练的部分

```text
prompt_encoder
```

prompt encoder 通常不需要动。

```text
mask_decoder
```

mask decoder 可以少量解冻，但不建议第一版就动。因为 SAM2 的 mask decoder 已经有很强的类别无关分割能力，过度微调可能破坏泛化能力。

------

# 七、LoRA 层的基本代码

你可以新建文件：

```text
lora.py
```

内容如下：

```python
import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.original_linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original_linear(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
```

这个模块的含义是：

```text
原始输出 = W x
LoRA 增量 = B A x
最终输出 = W x + alpha/r × B A x
```

原始 SAM2 参数冻结，只训练 A 和 B。

------

# 八、自动给 SAM2 注入 LoRA

新建：

```text
inject_lora.py
import torch.nn as nn
from lora import LoRALinear


def inject_lora_to_sam2(model, target_keywords=("q_proj", "v_proj"), r=8, alpha=16, dropout=0.05):
    replaced = []

    for name, module in model.named_modules():
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child_module, nn.Linear):
                if any(key in full_name for key in target_keywords):
                    setattr(
                        module,
                        child_name,
                        LoRALinear(
                            child_module,
                            r=r,
                            alpha=alpha,
                            dropout=dropout
                        )
                    )
                    replaced.append(full_name)

    print("LoRA injected into:")
    for name in replaced:
        print("  ", name)

    return model
```

如果你不确定 SAM2 内部层名，可以先打印：

```python
for name, module in model.named_modules():
    if "proj" in name or "attn" in name:
        print(name, type(module))
```

不同 SAM2 版本内部命名可能略有差异，所以这一步非常重要。

------

# 九、构建 Dataset

新建：

```text
dataset.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class WaterSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=1024):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        self.names = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        image_path = os.path.join(self.image_dir, name)
        mask_path = os.path.join(
            self.mask_dir,
            os.path.splitext(name)[0] + ".png"
        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "name": name
        }
```

------

# 十、训练时 prompt 怎么生成

SAM2 是 promptable segmentation，训练时最好给它提供 prompt。常用 prompt 有三种：

## 1. Box prompt

从 mask 自动计算 bounding box：

```python
def mask_to_box(mask):
    y, x = torch.where(mask[0] > 0.5)
    if len(x) == 0 or len(y) == 0:
        return torch.tensor([0, 0, 1, 1], dtype=torch.float32)

    x1, x2 = x.min(), x.max()
    y1, y2 = y.min(), y.max()

    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
```

这是最稳定的训练方式。

## 2. Point prompt

从目标区域内部随机采样正点，从背景区域随机采样负点。

适合交互式分割。

## 3. Mask prompt

把低分辨率 mask 作为提示输入。

适合细化边界，但训练更复杂。

第一版建议用：

```text
box prompt + mask loss
```

------

# 十一、损失函数设计

建议使用：

```text
BCE Loss + Dice Loss
```

代码：

```python
import torch
import torch.nn.functional as F


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    numerator = 2 * (pred * target).sum(dim=(1, 2, 3))
    denominator = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    loss = 1 - numerator / denominator
    return loss.mean()


def bce_dice_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dsc = dice_loss(pred, target)
    return bce + dsc
```

对于水体、洪水淹没区这类面积较大的目标，Dice 很重要。

对于裂缝、渗漏、小目标，建议加入 Focal Loss：

```text
BCE + Dice + Focal
```

------

# 十二、训练脚本框架

这里给你一个“工程级伪代码结构”。因为 SAM2 官方仓库内部 API 会随版本变化，实际函数名可能需要根据你本地版本调整，但训练逻辑就是这个。

新建：

```text
train_sam2_lora.py
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import WaterSegDataset
from inject_lora import inject_lora_to_sam2
from loss import bce_dice_loss

from sam2.build_sam import build_sam2


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable / 1e6:.3f}M / {total / 1e6:.3f}M")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_file = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"

    model = build_sam2(config_file, checkpoint, device=device)

    freeze_all(model)

    model = inject_lora_to_sam2(
        model,
        target_keywords=("q_proj", "v_proj"),
        r=8,
        alpha=16,
        dropout=0.05
    )

    model.to(device)
    model.train()

    count_trainable_params(model)

    train_dataset = WaterSegDataset(
        image_dir="dataset/train/images",
        mask_dir="dataset/train/masks",
        image_size=1024
    )

    val_dataset = WaterSegDataset(
        image_dir="dataset/val/images",
        mask_dir="dataset/val/masks",
        image_size=1024
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-4
    )

    scaler = torch.cuda.amp.GradScaler()

    epochs = 50
    save_dir = "outputs/sam2_lora_water"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # 这里需要根据你使用的 SAM2 API 写 forward
                # 训练逻辑是：
                # 1. image encoder 提取 image embedding
                # 2. prompt encoder 编码 box / point prompt
                # 3. mask decoder 输出 predicted mask
                # pred_masks = model_forward_with_box_prompt(model, images, masks)
                #
                # pred_masks shape: [B, 1, H, W]

                pred_masks = model_forward_with_box_prompt(model, images, masks)

                loss = bce_dice_loss(pred_masks, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "lora_state_dict": {
                    k: v.cpu()
                    for k, v in model.state_dict().items()
                    if "lora_" in k
                },
                "optimizer": optimizer.state_dict()
            },
            os.path.join(save_dir, f"sam2_lora_epoch_{epoch + 1}.pth")
        )


if __name__ == "__main__":
    main()
```

注意这里的：

```python
model_forward_with_box_prompt
```

需要你根据本地 SAM2 版本的 predictor/model API 补上。官方仓库更新较快，不同版本的内部调用略有差异，所以更稳妥的做法是先用官方训练代码跑通，然后把 LoRA 注入进去。官方仓库明确提供了训练/微调代码入口。([GitHub](https://github.com/facebookresearch/sam2))

------

# 十三、推荐的实际改法：基于 SAM2 官方训练代码插 LoRA

如果你想更稳，建议不要从零写训练 forward，而是这样做：

## 第一步：先跑通官方训练配置

进入 SAM2：

```bash
cd sam2
```

查看训练说明：

```bash
ls training
```

通常官方训练代码会包含：

```text
training/
  README.md
  configs/
  dataset/
  trainer/
```

你的第一目标是：
**不加 LoRA，先用官方配置跑通一次小数据训练。**

## 第二步：在 build model 后面插入 LoRA

找到类似：

```python
model = build_sam2(...)
```

或：

```python
model = instantiate(cfg.model)
```

后面加入：

```python
from inject_lora import inject_lora_to_sam2

for p in model.parameters():
    p.requires_grad = False

model = inject_lora_to_sam2(
    model,
    target_keywords=("q_proj", "v_proj"),
    r=8,
    alpha=16,
    dropout=0.05
)
```

## 第三步：确认 optimizer 只优化 LoRA

把 optimizer 参数改成：

```python
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)
```

## 第四步：保存时只保存 LoRA

```python
lora_state_dict = {
    k: v for k, v in model.state_dict().items()
    if "lora_" in k
}

torch.save(lora_state_dict, "sam2_lora_water.pth")
```

这样你的 checkpoint 会很小。

------

# 十四、超参数建议

## 水体 / 洪水淹没区

```text
模型：sam2.1_hiera_base_plus
LoRA rank：8 或 16
alpha：16 或 32
dropout：0.05
batch size：2–4
学习率：1e-4
epoch：30–80
loss：BCE + Dice
输入尺寸：1024
prompt：box prompt
```

## 堤坝裂缝 / 渗漏 / 小目标

```text
模型：sam2.1_hiera_base_plus 或 large
LoRA rank：16
alpha：32
dropout：0.05–0.1
batch size：1–2
学习率：5e-5
epoch：80–150
loss：BCE + Dice + Focal
输入尺寸：1024 或 patch 512
prompt：point + box
```

## SAR 洪水分割

```text
模型：sam2.1_hiera_base_plus
输入：SAR 单通道复制成三通道，或 VH/VV/ratio 组成三通道
LoRA rank：16
学习率：1e-4
loss：BCE + Dice
增强：随机噪声、对比度增强、speckle noise
```

------

# 十五、验证指标

不要只看 loss。建议用：

```text
IoU
Dice / F1
Precision
Recall
Boundary F1
面积相对误差
漏检率
误检率
```

水利场景建议额外加入：

## 洪水淹没区

```text
淹没面积误差 = |预测面积 - 真值面积| / 真值面积
```

## 河道 / 水体

```text
水体面积误差
岸线边界误差
河道连通性一致率
```

## 堤坝裂缝

```text
裂缝召回率
细长目标 Dice
骨架误差
```

## 应急场景

```text
高风险区域 Recall
重点建筑物漏检率
道路中断识别准确率
```

------

# 十六、推理流程

训练完成后，推理时流程如下：

```text
1. 加载原始 SAM2
2. 注入相同结构的 LoRA 层
3. 加载 LoRA checkpoint
4. 输入图像
5. 输入 prompt：box / point / text-derived box
6. 输出 mask
7. 后处理：连通域过滤、孔洞填充、面积筛选、GIS 约束
```

示例：

```python
model = build_sam2(config_file, checkpoint, device=device)

model = inject_lora_to_sam2(
    model,
    target_keywords=("q_proj", "v_proj"),
    r=8,
    alpha=16,
    dropout=0.05
)

lora_ckpt = torch.load("sam2_lora_water.pth", map_location="cpu")
model.load_state_dict(lora_ckpt, strict=False)

model.eval()
```

------

# 十七、水利领域后处理建议

SAM2-LoRA 输出 mask 后，不建议直接作为最终结果。水利任务最好加专业后处理。

## 1. 连通域过滤

去掉小块误检：

```text
面积 < 50 像素的小区域删除
```

## 2. 河网连通性约束

洪水区一般应与：

```text
河道
沟渠
低洼区
历史水体
排水系统
```

存在空间关系。

完全孤立的大块“水体”可能是阴影、屋顶或农田。

## 3. DEM 高程约束

洪水淹没区通常更可能出现在低洼区。

如果一个区域高程明显高于周围河道水位，却被识别为洪水，应降低置信度。

## 4. 历史水体差分

用于区分：

```text
永久水体
新增淹没区
季节性湿地
```

公式上可以写：

```text
新增淹没区 = 当前水体 mask - 历史常水位水体 mask
```

## 5. 边界平滑

对遥感水体边界可做：

```text
morphology close
morphology open
contour smoothing
```

但裂缝任务不要过度平滑，否则细节会丢失。

------

# 十八、训练中常见问题

## 问题 1：mask 全黑

原因可能是：

```text
学习率太大
正负样本极不平衡
prompt 生成错误
mask 读取后变成 0/1 错误
输出尺寸和 GT 尺寸不一致
```

解决：

```text
检查 mask 是否真有 255
降低 lr 到 5e-5
加入 Dice Loss / Focal Loss
可视化每个 batch
```

## 问题 2：模型只会分割大水体，不会分割细小裂缝

原因：

```text
小目标占比太低
resize 后裂缝消失
loss 对小目标不敏感
```

解决：

```text
切 patch
提高输入分辨率
使用 Focal Loss
增加裂缝样本权重
使用 point prompt
```

## 问题 3：训练效果好，测试效果差

原因：

```text
数据来源单一
遥感季节差异大
不同传感器差异大
标注风格不一致
LoRA rank 太大导致过拟合
```

解决：

```text
降低 rank
增加数据增强
跨区域划分 train/test
不要让同一区域相邻切片同时进入 train 和 test
```

## 问题 4：显存不够

解决：

```text
换 sam2.1_hiera_tiny 或 small
batch size = 1
开启 mixed precision
冻结更多模块
只训练 q_proj / v_proj
使用 gradient accumulation
```

## 问题 5：LoRA 没有效果

检查：

```python
for name, p in model.named_parameters():
    if p.requires_grad:
        print(name, p.shape)
```

如果没有输出，说明 LoRA 没注入成功。

------

# 十九、推荐实验对比

如果你要写论文，建议设置这些对照组：

| 方法                   | 说明                 |
| ---------------------- | -------------------- |
| U-Net                  | 经典分割基线         |
| DeepLabV3+             | 语义分割基线         |
| SegFormer              | Transformer 分割基线 |
| SAM2 zero-shot         | 不微调，只给 prompt  |
| SAM2 full fine-tune    | 可选，显存允许才做   |
| SAM2-LoRA              | 你的方法             |
| SAM2-LoRA + 水利后处理 | 最终方法             |

这样可以证明：

1. LoRA 比 zero-shot SAM2 更适合水利场景；
2. LoRA 比全量微调更省参数；
3. 水利知识后处理能减少误检；
4. 推理分割系统比单纯视觉分割更适合工程应用。

------

# 二十、建议你在论文中这样描述方法

可以写成：

> 本文采用参数高效微调策略对 SAM2 进行水利场景适配。具体而言，冻结 SAM2 原始图像编码器、提示编码器和掩膜解码器的大部分参数，仅在图像编码器注意力模块的查询投影和值投影层中引入低秩适配矩阵。训练过程中，以水体、洪水淹没区或水工结构缺陷的标注掩膜自动生成边界框提示，利用 BCE-Dice 联合损失优化 LoRA 参数。推理阶段，加载原始 SAM2 权重与训练得到的 LoRA 权重，并结合水系连通性、高程约束和历史水体边界进行结果校正，从而获得更符合水利工程语义的分割结果。

------

# 二十一、我建议你的第一版最小可行方案

不要一开始做太复杂。第一版建议这样：

```text
任务：洪水淹没区二值分割
数据：Sentinel-1 SAR 或无人机洪水图像
模型：sam2.1_hiera_base_plus
微调方式：只在 image_encoder 的 q_proj/v_proj 加 LoRA
prompt：由 GT mask 自动生成 box prompt
loss：BCE + Dice
评价：IoU、Dice、Recall、面积误差
后处理：连通域过滤 + 历史水体差分 + DEM 约束
```

这套方案最容易跑通，也最容易写成水利领域文章。

------

# 二十二、整体流程总览

```text
Step 1：准备水利图像和二值 mask
Step 2：安装 SAM2 官方仓库和 PyTorch
Step 3：下载 SAM2.1 checkpoint
Step 4：跑通官方 SAM2 推理
Step 5：跑通官方微调代码
Step 6：冻结 SAM2 原始参数
Step 7：向 image encoder 的 q_proj/v_proj 注入 LoRA
Step 8：只训练 LoRA 参数
Step 9：使用 BCE + Dice Loss 优化
Step 10：保存 LoRA 权重
Step 11：推理时加载 SAM2 + LoRA
Step 12：加入水利专业后处理
Step 13：与 U-Net、SegFormer、SAM2 zero-shot 做对比
```

最重要的一点是：**SAM2-LoRA 微调本身只是视觉适配，真正体现水利专业价值的是“LoRA 分割结果 + 水系/DEM/历史水体/工程边界约束”的组合。**





下面给你一套**专门面向 Potsdam 数据集的 SAM2 LoRA 微调完整流程**。我建议你的技术路线不要把 Potsdam 当成普通二值分割数据，而是把它处理成：

**多类别语义分割任务 → 拆成多个“类别级 promptable mask”任务 → 用 SAM2 + LoRA 学习遥感场景中的对象边界与类别语义。**

Potsdam 数据集本身是城市遥感语义分割数据集，包含 RGB、IRRG、RGBIR、DSM 和像素级标签；单块影像尺寸为 **6000×6000**，GSD 为 **5 cm**，共有 **6 个主要地物类别 + 1 个边界/忽略类**。 这决定了你的训练流程必须做**切片、标签颜色映射、边界忽略、类别均衡和 prompt 构造**。

------

# 一、总体技术路线

建议采用下面这条路线：

```text
Potsdam 原始大图
→ 读取 RGB / IRRG / RGBIR / DSM
→ 标签颜色转 class id
→ 忽略 boundary / don't care 区域
→ 6000×6000 大图切成 512 或 1024 patch
→ 每个 patch 根据类别生成二值 mask
→ 根据二值 mask 自动生成 box / point prompt
→ SAM2 image encoder 注入 LoRA
→ 冻结 SAM2 主体，只训练 LoRA 和可选 mask decoder 小部分参数
→ 逐类别训练或多类别联合训练
→ 推理时滑窗预测
→ 拼接回 6000×6000
→ 计算 mIoU、F1、OA、per-class IoU
```

SAM2 官方仓库已经提供训练和微调代码，官方说明其训练代码支持用户在自己的图像或视频数据集上训练/微调 SAM2；同时官方安装要求是 Python ≥ 3.10、torch ≥ 2.5.1、torchvision ≥ 0.20.1。([GitHub](https://github.com/facebookresearch/sam2?utm_source=chatgpt.com))

------

# 二、你应该选择哪种微调范式

Potsdam 是**多类别语义分割数据集**，但 SAM2 是**promptable segmentation model**，不是天然的 DeepLab / SegFormer 那种一次输出 6 类 logits 的语义分割模型。

所以有三种做法。

## 方案 A：逐类别二值 LoRA 微调，最稳

把 6 个类别分别转成二值 mask：

```text
类别 0：clutter/background vs others
类别 1：impervious surface vs others
类别 2：building vs others
类别 3：low vegetation vs others
类别 4：tree vs others
类别 5：car vs others
```

训练时每次随机选一个类别，把该类别作为前景，其他类别作为背景。

优点是最符合 SAM2 的 promptable segmentation 逻辑。

缺点是推理时需要对 6 个类别分别预测，再融合。

我建议你优先使用这个方案。

## 方案 B：多类别联合训练，一个 batch 里包含 class prompt

训练时输入：

```text
image patch
target class id
class binary mask
box prompt / point prompt
```

本质上仍然是二值 mask 训练，只是一个模型适配所有类别。

优点是只有一个 LoRA 权重。

缺点是需要设计类别采样策略，否则 car 这种小目标很容易被忽略。

## 方案 C：改 SAM2 mask decoder，让它输出 6 类 logits

这个更像普通语义分割，但会破坏 SAM2 的原始 promptable 结构，工程复杂度明显更高。

不建议作为第一版。

------

# 三、Potsdam 标签处理原则

Potsdam 的标签颜色不是普通灰度 class id。你需要先把 RGB 标签颜色转成类别 id。文档给出的颜色映射是：

| class id | 类别                  | RGB         |
| -------- | --------------------- | ----------- |
| 0        | Clutter / Background  | 255,255,255 |
| 1        | Impervious Surface    | 0,0,0       |
| 2        | Building              | 0,0,255     |
| 3        | Low Vegetation        | 0,255,255   |
| 4        | Tree                  | 0,255,0     |
| 5        | Car                   | 255,255,0   |
| 6        | Boundary / Don't Care | 255,0,0     |

Potsdam 提供含边界标签和不含边界标签；含边界标签中的红色边界区域是类别 6，在评估时应忽略，文档也建议使用不含边界标签进行训练。

因此我建议：

```text
训练：优先使用 5_Labels_for_participants_no_Boundary
验证：如果使用含 boundary 标签，则 class 6 设为 ignore_index=255
评估：不要把 boundary 当成正常类别参与 mIoU
```

------

# 四、推荐目录结构

假设你的原始数据是这样：

```text
Potsdam/
├── 2_Ortho_RGB/
│   └── 2_Ortho_RGB/
│       ├── top_potsdam_2_10_RGB.tif
│       └── ...
├── 3_Ortho_IRRG/
│   └── 3_Ortho_IRRG/
├── 4_Ortho_RGBIR/
│   └── 4_Ortho_RGBIR/
├── 1_DSM/
│   └── 1_DSM/
├── 5_Labels_for_participants_no_Boundary/
│   └── 5_Labels_for_participants_no_Boundary/
│       ├── top_potsdam_2_10_label_noBoundary.tif
│       └── ...
```

建议处理成训练目录：

```text
sam2_lora_potsdam/
├── data/
│   ├── raw/
│   │   └── Potsdam/
│   ├── patches/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   ├── labels/
│   │   │   └── masks/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   ├── labels/
│   │   │   └── masks/
│   │   └── test/
│   ├── splits/
│   │   ├── train_tiles.txt
│   │   ├── val_tiles.txt
│   │   └── test_tiles.txt
├── sam2/
├── lora/
│   ├── lora_layers.py
│   ├── inject_lora.py
│   └── train_potsdam_lora.py
├── checkpoints/
└── outputs/
```

------

# 五、训练 / 验证瓦片划分

Potsdam 有标签的瓦片是 24 块，典型划分是 16 块训练、8 块验证。

我建议采用空间上相对分开的划分，避免相邻 patch 泄漏。

## 推荐划分

```text
train:
2_10, 2_11, 2_12
3_10, 3_11, 3_12
4_10, 4_11, 4_12
5_10, 5_11, 5_12
6_7, 6_8, 6_9, 6_10

val:
6_11, 6_12
7_7, 7_8, 7_9, 7_10, 7_11, 7_12
```

保存为：

```text
data/splits/train_tiles.txt
2_10
2_11
2_12
3_10
3_11
3_12
4_10
4_11
4_12
5_10
5_11
5_12
6_7
6_8
6_9
6_10
data/splits/val_tiles.txt
6_11
6_12
7_7
7_8
7_9
7_10
7_11
7_12
```

------

# 六、环境安装

## 1. 创建环境

```bash
conda create -n sam2_potsdam python=3.10 -y
conda activate sam2_potsdam
```

## 2. 安装 PyTorch

以 CUDA 12.1 为例：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

检查 GPU：

```bash
python - <<'EOF'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
EOF
```

## 3. 安装 SAM2

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
```

官方 SAM2 安装方式就是 clone 仓库后 `pip install -e .`，并推荐在 Linux / WSL Ubuntu 环境下使用。([GitHub](https://github.com/facebookresearch/sam2?utm_source=chatgpt.com))

## 4. 安装遥感和训练依赖

```bash
pip install numpy opencv-python pillow tqdm matplotlib scikit-learn
pip install albumentations rasterio tifffile einops
pip install hydra-core omegaconf
pip install tensorboard
```

GDAL 有时比较麻烦，推荐优先用 `rasterio` 或 `tifffile` 读取 Potsdam 的 tif。

------

# 七、下载 SAM2.1 checkpoint

建议使用 `sam2.1_hiera_base_plus` 作为第一版。

```bash
mkdir -p checkpoints
cd checkpoints

wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

cd ..
```

对应 config 通常是：

```text
sam2/configs/sam2.1/sam2.1_hiera_b+.yaml
```

显存选择建议：

| 显存         | 模型                      |
| ------------ | ------------------------- |
| 8GB          | sam2.1_hiera_tiny / small |
| 12–16GB      | sam2.1_hiera_base_plus    |
| 24GB+        | sam2.1_hiera_large        |
| 调试         | tiny                      |
| 正式论文实验 | base_plus 或 large        |

------

# 八、Potsdam 原始标签转 class id

新建：

```text
scripts/convert_label_color_to_id.py
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


COLOR_TO_ID = {
    (255, 255, 255): 0,  # clutter/background
    (0, 0, 0): 1,        # impervious surface
    (0, 0, 255): 2,      # building
    (0, 255, 255): 3,    # low vegetation
    (0, 255, 0): 4,      # tree
    (255, 255, 0): 5,    # car
    (255, 0, 0): 255,    # boundary / ignore
}


def rgb_label_to_id(label_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = label_rgb.shape
    label_id = np.full((h, w), 255, dtype=np.uint8)

    for color, class_id in COLOR_TO_ID.items():
        color_arr = np.array(color, dtype=np.uint8)
        match = np.all(label_rgb == color_arr, axis=-1)
        label_id[match] = class_id

    return label_id


def main(label_dir, out_dir):
    label_dir = Path(label_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(label_dir.glob("*.tif"))

    for p in tqdm(paths):
        # cv2 读取是 BGR，所以转 RGB
        label_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if label_bgr is None:
            raise FileNotFoundError(p)

        label_rgb = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2RGB)
        label_id = rgb_label_to_id(label_rgb)

        out_name = p.stem.replace("_label_noBoundary", "").replace("_label", "") + "_label_id.png"
        cv2.imwrite(str(out_dir / out_name), label_id)


if __name__ == "__main__":
    main(
        label_dir="data/raw/Potsdam/5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary",
        out_dir="data/processed/labels_id"
    )
```

运行：

```bash
python scripts/convert_label_color_to_id.py
```

检查：

```python
import cv2
import numpy as np

x = cv2.imread("data/processed/labels_id/top_potsdam_2_10_label_id.png", cv2.IMREAD_GRAYSCALE)
print(np.unique(x))
```

理论上应该看到：

```text
[0 1 2 3 4 5]
```

如果你用的是含边界标签，可能看到：

```text
[0 1 2 3 4 5 255]
```

------

# 九、切片处理

Potsdam 单张图是 6000×6000，文档也提醒单文件较大，需要分块处理，建议 512×512 或 256×256，并按需读取。

对 SAM2 来说，我建议：

```text
patch_size = 1024
stride = 512 或 768
```

原因：

```text
1024 接近 SAM2 的常用输入尺度
stride < patch_size 可以减少边界断裂
car 等小目标不容易被切坏
```

如果显存不足：

```text
patch_size = 512
stride = 384
```

新建：

```text
scripts/make_potsdam_patches.py
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_label_id(path):
    label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise FileNotFoundError(path)
    return label


def crop_with_padding(arr, x, y, size, pad_value=0):
    h, w = arr.shape[:2]
    crop = arr[y:min(y+size, h), x:min(x+size, w)]

    if arr.ndim == 3:
        out = np.full((size, size, arr.shape[2]), pad_value, dtype=arr.dtype)
        out[:crop.shape[0], :crop.shape[1], :] = crop
    else:
        out = np.full((size, size), pad_value, dtype=arr.dtype)
        out[:crop.shape[0], :crop.shape[1]] = crop

    return out


def make_patches(
    rgb_dir,
    label_id_dir,
    split_txt,
    out_img_dir,
    out_label_dir,
    patch_size=1024,
    stride=512,
    min_valid_ratio=0.5
):
    rgb_dir = Path(rgb_dir)
    label_id_dir = Path(label_id_dir)
    out_img_dir = Path(out_img_dir)
    out_label_dir = Path(out_label_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    with open(split_txt, "r", encoding="utf-8") as f:
        tiles = [line.strip() for line in f if line.strip()]

    for tile in tqdm(tiles, desc=f"patching {split_txt}"):
        r, c = tile.split("_")
        img_path = rgb_dir / f"top_potsdam_{r}_{c}_RGB.tif"
        label_path = label_id_dir / f"top_potsdam_{r}_{c}_label_id.png"

        image = read_rgb(img_path)
        label = read_label_id(label_path)

        h, w = label.shape

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                img_patch = crop_with_padding(image, x, y, patch_size, pad_value=0)
                label_patch = crop_with_padding(label, x, y, patch_size, pad_value=255)

                valid = label_patch != 255
                valid_ratio = valid.mean()

                if valid_ratio < min_valid_ratio:
                    continue

                name = f"potsdam_{tile}_x{x}_y{y}.png"

                cv2.imwrite(
                    str(out_img_dir / name),
                    cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(str(out_label_dir / name), label_patch)


if __name__ == "__main__":
    make_patches(
        rgb_dir="data/raw/Potsdam/2_Ortho_RGB/2_Ortho_RGB",
        label_id_dir="data/processed/labels_id",
        split_txt="data/splits/train_tiles.txt",
        out_img_dir="data/patches/train/images",
        out_label_dir="data/patches/train/labels",
        patch_size=1024,
        stride=512
    )

    make_patches(
        rgb_dir="data/raw/Potsdam/2_Ortho_RGB/2_Ortho_RGB",
        label_id_dir="data/processed/labels_id",
        split_txt="data/splits/val_tiles.txt",
        out_img_dir="data/patches/val/images",
        out_label_dir="data/patches/val/labels",
        patch_size=1024,
        stride=512
    )
```

运行：

```bash
python scripts/make_potsdam_patches.py
```

------

# 十、是否使用 RGB、IRRG、RGBIR、DSM？

Potsdam 提供 RGB、IRRG、RGBIR 和 DSM，DSM 对建筑物、树木等立体目标区分有帮助。

但 SAM2 预训练输入默认是 3 通道 RGB。因此第一版建议：

```text
第一版：只用 RGB
第二版：尝试 IRRG 伪 RGB
第三版：RGB + DSM 融合，但不要直接改 SAM2 输入层
```

## 为什么不要第一版就用 RGBIR 或 DSM？

SAM2 的 image encoder 第一层预期是 3 通道输入。你如果直接改成 4 通道或 5 通道，会涉及改 patch embedding / stem 权重，复杂度变高，而且破坏预训练权重。

## 推荐的 DSM 融合方式

更稳的做法是：

```text
RGB 输入 SAM2
DSM 用于后处理、prompt 生成或辅助类别筛选
```

例如：

```text
building/tree/car 训练时，DSM 可用于筛掉明显不可能的区域；
impervious surface / low vegetation 可结合 nDSM 做后处理。
```

论文里可以写成：

```text
本文第一阶段采用 RGB-only SAM2-LoRA；
第二阶段将 DSM/nDSM 作为空间先验用于结果修正，而非直接修改 SAM2 输入层。
```

------

# 十一、构造 SAM2 训练样本

你的每个训练样本应该是：

```python
{
    "image": image_patch,          # RGB, H×W×3
    "label": label_patch,          # H×W, class id 0-5 or 255
    "target_class": cls,           # 0-5
    "binary_mask": mask,           # H×W, 当前类别为 1，其余为 0
    "box": box,                    # 当前类别实例/区域的 box prompt
    "points": points,              # 可选 point prompt
}
```

关键是：**从语义标签生成类别级二值 mask**。

例如当前类别是 building：

```python
binary_mask = (label == 2)
```

其他类别全部视为背景：

```python
background = (label != 2) and (label != 255)
```

ignore 区域不参与 loss：

```python
ignore = (label == 255)
```

------

# 十二、Potsdam Dataset 类

新建：

```text
lora/potsdam_dataset.py
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


CLASS_NAMES = {
    0: "clutter_background",
    1: "impervious_surface",
    2: "building",
    3: "low_vegetation",
    4: "tree",
    5: "car",
}


def mask_to_box(mask, jitter=10):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    h, w = mask.shape

    x1 = max(0, x1 - random.randint(0, jitter))
    y1 = max(0, y1 - random.randint(0, jitter))
    x2 = min(w - 1, x2 + random.randint(0, jitter))
    y2 = min(h - 1, y2 + random.randint(0, jitter))

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def sample_points(binary_mask, ignore_mask=None, num_pos=1, num_neg=1):
    h, w = binary_mask.shape

    pos_y, pos_x = np.where(binary_mask > 0)
    if ignore_mask is None:
        neg_y, neg_x = np.where(binary_mask == 0)
    else:
        neg_y, neg_x = np.where((binary_mask == 0) & (~ignore_mask))

    points = []
    labels = []

    if len(pos_x) > 0:
        ids = np.random.choice(len(pos_x), size=min(num_pos, len(pos_x)), replace=False)
        for i in ids:
            points.append([pos_x[i], pos_y[i]])
            labels.append(1)

    if len(neg_x) > 0:
        ids = np.random.choice(len(neg_x), size=min(num_neg, len(neg_x)), replace=False)
        for i in ids:
            points.append([neg_x[i], neg_y[i]])
            labels.append(0)

    if len(points) == 0:
        return None, None

    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int64)


class PotsdamPromptDataset(Dataset):
    def __init__(
        self,
        image_dir,
        label_dir,
        image_size=1024,
        classes=(0, 1, 2, 3, 4, 5),
        min_fg_pixels=64,
        use_box=True,
        use_points=False
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size
        self.classes = list(classes)
        self.min_fg_pixels = min_fg_pixels
        self.use_box = use_box
        self.use_points = use_points

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.samples = []

        for img_path in self.image_paths:
            label_path = self.label_dir / img_path.name
            if not label_path.exists():
                continue

            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if label is None:
                continue

            for cls in self.classes:
                fg = (label == cls)
                if fg.sum() >= self.min_fg_pixels:
                    self.samples.append((img_path, label_path, cls))

        print(f"Loaded {len(self.samples)} class-level samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path, cls = self.samples[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        ignore = (label == 255)
        binary = (label == cls).astype(np.uint8)

        box = mask_to_box(binary, jitter=20) if self.use_box else None
        if box is None:
            box = np.array([0, 0, 1, 1], dtype=np.float32)

        points, point_labels = sample_points(binary, ignore, num_pos=1, num_neg=1)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        binary = torch.from_numpy(binary).unsqueeze(0).float()
        ignore = torch.from_numpy(ignore.astype(np.uint8)).unsqueeze(0).bool()

        sample = {
            "image": image,
            "mask": binary,
            "ignore": ignore,
            "box": torch.from_numpy(box).float(),
            "class_id": torch.tensor(cls).long(),
            "name": img_path.name,
        }

        if points is not None:
            sample["points"] = torch.from_numpy(points).float()
            sample["point_labels"] = torch.from_numpy(point_labels).long()

        return sample
```

------

# 十三、LoRA 注入模块

新建：

```text
lora/lora_layers.py
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r=8, alpha=16, dropout=0.05):
        super().__init__()

        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.original_linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original_linear(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
```

新建：

```text
lora/inject_lora.py
import torch.nn as nn
from lora_layers import LoRALinear


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def inject_lora(
    model,
    target_keywords=("q_proj", "v_proj"),
    r=8,
    alpha=16,
    dropout=0.05
):
    replaced = []

    for module_name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name

            if isinstance(child, nn.Linear):
                if any(k in full_name for k in target_keywords):
                    setattr(
                        module,
                        child_name,
                        LoRALinear(
                            child,
                            r=r,
                            alpha=alpha,
                            dropout=dropout
                        )
                    )
                    replaced.append(full_name)

    print("LoRA injected into:")
    for x in replaced:
        print("  ", x)

    if len(replaced) == 0:
        print("WARNING: no layer was replaced. Check module names.")

    return model


def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(f"Trainable params: {trainable:,}")
    print(f"Total params: {total:,}")
    print(f"Trainable ratio: {100 * trainable / total:.4f}%")
```

------

# 十四、先检查 SAM2 模块名称

不同 SAM2 版本内部层名可能有差异，所以你必须先打印模块名。

新建：

```text
scripts/inspect_sam2_modules.py
from sam2.build_sam import build_sam2

config_file = "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"

model = build_sam2(config_file, checkpoint, device="cuda")

for name, module in model.named_modules():
    if any(k in name.lower() for k in ["q", "k", "v", "proj", "attn", "linear"]):
        print(name, type(module))
```

运行：

```bash
python scripts/inspect_sam2_modules.py > sam2_modules.txt
```

然后搜索：

```bash
grep -E "q_proj|v_proj|attn|proj" sam2_modules.txt
```

如果你的 SAM2 层名不是 `q_proj/v_proj`，就把 `target_keywords` 改成真实层名，例如：

```python
target_keywords=("attn.q", "attn.v")
```

或：

```python
target_keywords=("qkv",)
```

------

# 十五、训练损失函数

新建：

```text
lora/losses.py
import torch
import torch.nn.functional as F


def masked_bce_with_logits(logits, targets, ignore_mask=None):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    if ignore_mask is not None:
        valid = ~ignore_mask
        loss = loss * valid.float()
        return loss.sum() / valid.float().sum().clamp_min(1.0)

    return loss.mean()


def masked_dice_loss(logits, targets, ignore_mask=None, eps=1e-6):
    probs = torch.sigmoid(logits)

    if ignore_mask is not None:
        valid = (~ignore_mask).float()
        probs = probs * valid
        targets = targets * valid

    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def focal_loss_with_logits(logits, targets, ignore_mask=None, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1 - probs) * (1 - targets)
    loss = alpha * (1 - pt) ** gamma * bce

    if ignore_mask is not None:
        valid = ~ignore_mask
        loss = loss * valid.float()
        return loss.sum() / valid.float().sum().clamp_min(1.0)

    return loss.mean()


def potsdam_loss(logits, targets, ignore_mask=None, use_focal=True):
    bce = masked_bce_with_logits(logits, targets, ignore_mask)
    dice = masked_dice_loss(logits, targets, ignore_mask)

    if use_focal:
        focal = focal_loss_with_logits(logits, targets, ignore_mask)
        return bce + dice + 0.5 * focal

    return bce + dice
```

建议：

```text
大目标：BCE + Dice
小目标 car：BCE + Dice + Focal
```

因为 Potsdam 中 car 类非常小，若不用 focal 或类别重采样，模型很可能忽略 car。

------

# 十六、训练脚本的核心结构

这里给你一个**可作为工程骨架使用的训练脚本**。需要说明的是，SAM2 官方训练 API 可能随版本略有变化，最稳做法是在官方 `training/train.py` 流程里插入 LoRA；但下面这个脚本把核心逻辑写清楚了，你可以基于你本地 SAM2 版本补齐 forward。

新建：

```text
lora/train_potsdam_lora.py
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("sam2")
sys.path.append("lora")

from sam2.build_sam import build_sam2
from potsdam_dataset import PotsdamPromptDataset
from inject_lora import freeze_model, inject_lora, print_trainable_parameters
from losses import potsdam_loss


def save_lora_only(model, path):
    state = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    torch.save(state, path)


def load_lora_only(model, path, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)


def sam2_forward_placeholder(model, images, boxes):
    """
    这里需要替换成你本地 SAM2 的训练 forward。

    目标：
    输入:
        images: [B, 3, H, W]
        boxes:  [B, 4]
    输出:
        logits: [B, 1, H, W]

    建议优先参考 sam2/training/model/sam2_train.py
    或官方 training 配置中的 forward 方式。
    """
    raise NotImplementedError(
        "请根据你本地 SAM2 官方 training API 补齐 forward。"
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_file = "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
    checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"

    model = build_sam2(
        config_file=config_file,
        ckpt_path=checkpoint,
        device=device
    )

    freeze_model(model)

    model = inject_lora(
        model,
        target_keywords=("q_proj", "v_proj"),
        r=8,
        alpha=16,
        dropout=0.05
    )

    model.to(device)
    model.train()

    print_trainable_parameters(model)

    train_dataset = PotsdamPromptDataset(
        image_dir="data/patches/train/images",
        label_dir="data/patches/train/labels",
        image_size=1024,
        classes=(0, 1, 2, 3, 4, 5),
        min_fg_pixels=64,
        use_box=True,
        use_points=False
    )

    val_dataset = PotsdamPromptDataset(
        image_dir="data/patches/val/images",
        label_dir="data/patches/val/labels",
        image_size=1024,
        classes=(0, 1, 2, 3, 4, 5),
        min_fg_pixels=64,
        use_box=True,
        use_points=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-4
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    os.makedirs("outputs/potsdam_sam2_lora", exist_ok=True)

    epochs = 50

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            ignore = batch["ignore"].to(device)
            boxes = batch["box"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = sam2_forward_placeholder(model, images, boxes)
                loss = potsdam_loss(
                    logits=logits,
                    targets=masks,
                    ignore_mask=ignore,
                    use_focal=True
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")

        save_lora_only(
            model,
            f"outputs/potsdam_sam2_lora/lora_epoch_{epoch+1}.pth"
        )


if __name__ == "__main__":
    main()
```

------

# 十七、更推荐的实际做法：基于官方 training 代码插入 LoRA

SAM2 官方训练代码组织在 `training` 文件夹下，包括 dataset、model、utils、loss、optimizer、trainer、train.py 等模块；官方 README 说明 `training/train.py` 是训练入口。([GitHub](https://github.com/facebookresearch/sam2/blob/main/training/README.md?utm_source=chatgpt.com))

因此真正落地时，建议你这样改：

## 第一步：先确认官方训练入口能跑

```bash
cd sam2
python training/train.py -h
```

## 第二步：在官方构建模型后插入 LoRA

找到官方训练代码里构建模型的位置，通常在配置实例化后会得到类似：

```python
model = ...
```

在其后加入：

```python
from lora.inject_lora import freeze_model, inject_lora, print_trainable_parameters

freeze_model(model)

model = inject_lora(
    model,
    target_keywords=("q_proj", "v_proj"),
    r=8,
    alpha=16,
    dropout=0.05
)

print_trainable_parameters(model)
```

## 第三步：optimizer 只训练 LoRA

把 optimizer 的参数来源改成：

```python
params = [p for p in model.parameters() if p.requires_grad]
```

而不是：

```python
model.parameters()
```

## 第四步：保存 LoRA 权重

训练保存 checkpoint 时，额外保存：

```python
lora_state = {
    k: v.cpu()
    for k, v in model.state_dict().items()
    if "lora_A" in k or "lora_B" in k
}

torch.save(lora_state, "potsdam_sam2_lora_only.pth")
```

这样你的 LoRA 权重很小，论文也更容易说明“参数高效微调”。

------

# 十八、SAM2 forward 应该怎么接 box prompt

SAM2 的核心训练路径一般是：

```text
image encoder
→ prompt encoder
→ mask decoder
→ mask logits
```

你需要实现的逻辑是：

```python
image_embeddings = model.image_encoder(images)

sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
    points=None,
    boxes=boxes,
    masks=None
)

low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
    image_embeddings=image_embeddings,
    image_pe=model.sam_prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=False,
    repeat_image=False,
    high_res_features=...
)

pred_masks = interpolate(low_res_masks, size=(H, W))
```

但 SAM2.1 的 high-res features、memory 模块和官方训练 wrapper 可能会封装得更复杂，所以我建议不要硬写裸 forward，而是参考：

```text
sam2/training/model/sam2_train.py
sam2/training/loss_fns.py
sam2/training/trainer.py
```

你的目标不是重写 SAM2，而是在官方训练 wrapper 上：

```text
冻结参数
注入 LoRA
替换 dataset
替换 loss 或使用官方 loss
替换保存方式
```

------

# 十九、类别采样策略

Potsdam 最大的问题是类别不平衡。

```text
impervious surface、building、low vegetation、tree 面积较大
car 面积极小
clutter/background 语义复杂
```

如果按 patch 随机采样，car 类训练样本会严重不足。

建议：

## 1. 类别均衡采样

构建 samples 时，不是每个 patch 均匀采样，而是：

```text
每个类别维护一个 sample list
每个 batch 尽量包含多个类别
car 类过采样
```

伪代码：

```python
class_weights = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 4.0,  # car
}
```

## 2. 设置不同 min_fg_pixels

```text
大类：min_fg_pixels = 512
car：min_fg_pixels = 16 或 32
```

否则 car patch 会被过滤掉。

## 3. car 类使用更小 patch

可以额外制作一套：

```text
patch_size = 512
stride = 256
```

用于 car 训练。

------

# 二十、超参数推荐

## 方案 1：第一版稳健实验

```text
模型：sam2.1_hiera_base_plus
输入：RGB
patch_size：1024
stride：512
LoRA 注入：image_encoder q_proj + v_proj
LoRA rank：8
alpha：16
dropout：0.05
batch_size：2
lr：1e-4
weight_decay：1e-4
epochs：50
loss：BCE + Dice + 0.5 Focal
prompt：box prompt
ignore_index：255
```

## 方案 2：显存不足

```text
模型：sam2.1_hiera_small
patch_size：512
batch_size：2 或 4
LoRA rank：8
lr：1e-4
gradient accumulation：4
```

## 方案 3：论文正式实验

```text
模型：sam2.1_hiera_base_plus / large
输入：RGB、IRRG 分别做实验
LoRA rank：8、16 做消融
注入位置：qv / qkv / qkvo 做消融
prompt：box、point+box 做消融
loss：BCE+Dice vs BCE+Dice+Focal 做消融
```

------

# 二十一、验证与评价

Potsdam 是多类别语义分割，因此最终指标不能只看二值 Dice。

你应该报告：

```text
OA: Overall Accuracy
mIoU: mean Intersection over Union
mF1: mean F1-score
per-class IoU
per-class F1
```

文档里也给出了类似 IoU、mean IoU、F1、mean F1 的评价思路。

## 评价代码

```python
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_metrics(pred, gt, num_classes=6, ignore_index=255):
    pred = pred.flatten()
    gt = gt.flatten()

    valid = gt != ignore_index
    pred = pred[valid]
    gt = gt[valid]

    cm = confusion_matrix(gt, pred, labels=list(range(num_classes)))

    oa = np.diag(cm).sum() / np.maximum(cm.sum(), 1)

    ious = []
    f1s = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        iou = tp / max(tp + fp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)

        ious.append(iou)
        f1s.append(f1)

    return {
        "OA": oa,
        "mIoU": float(np.mean(ious)),
        "mF1": float(np.mean(f1s)),
        "IoU_per_class": ious,
        "F1_per_class": f1s,
    }
```

------

# 二十二、推理时如何从 6 个二值 mask 合成语义图

因为训练是类别级二值 mask，所以推理时需要：

```text
对每个类别 c:
    生成该类别的 prompt
    得到 mask probability P_c
最终 label = argmax_c P_c
```

但这里有一个问题：推理时没有 GT mask，不能从 GT 生成 box prompt。

所以你需要设计推理 prompt 来源。

## 推理 prompt 来源一：全图网格 box

对每个 patch 使用多个固定 box：

```text
整图 box: [0, 0, W-1, H-1]
四分块 box
滑动小 box
```

适合大类：

```text
impervious surface
building
low vegetation
tree
```

不太适合 car。

## 推理 prompt 来源二：候选区域生成

先用 SAM2 自动 mask generator 生成候选 mask，再用 LoRA 适配后的模型细化。然后用分类规则或轻量分类头把候选 mask 分到 6 类。

这是更符合 SAM2 的方式。

流程：

```text
SAM2 automatic mask generator
→ 得到 object proposals
→ 对每个 proposal crop/mask 进行类别判别
→ 合成 semantic map
```

## 推理 prompt 来源三：由传统模型提供 prompt

用一个轻量 SegFormer / UNet 先得到粗分割，再把每类连通域转成 box prompt，交给 SAM2-LoRA 细化边界。

这是我最推荐的论文方案：

```text
SegFormer coarse semantic map
→ class connected components
→ box prompts
→ SAM2-LoRA boundary refinement
→ final semantic map
```

这样能解决 SAM2 没有语义输出的问题，也能突出 SAM2-LoRA 的边界细化优势。

------

# 二十三、推荐论文式模型框架

对于 Potsdam，最合理的模型框架不是“单独 SAM2-LoRA 做 6 类语义分割”，而是：

```text
粗语义分割分支：SegFormer / DeepLabV3+
边界细化分支：SAM2-LoRA
prompt 生成器：由粗分割连通域生成 box / point
融合模块：按类别概率和 mask 置信度融合
```

具体为：

```text
RGB patch
→ SegFormer 输出 6 类 coarse logits
→ 每个类别提取连通域
→ 每个连通域生成 box prompt
→ SAM2-LoRA 输出高质量 mask
→ 将 mask 写回对应类别
→ 得到最终 6 类语义分割图
```

这样有三个优点：

1. SegFormer 负责“这是什么类”；
2. SAM2-LoRA 负责“边界在哪里”；
3. LoRA 负责让 SAM2 适应 Potsdam 遥感风格。

这比强行让 SAM2 直接做语义分割更合理。

------

# 二十四、实验对比设计

建议至少做这些实验：

| 实验组                           | 说明               |
| -------------------------------- | ------------------ |
| U-Net                            | 传统基线           |
| DeepLabV3+                       | 强语义分割基线     |
| SegFormer                        | 遥感分割常用强基线 |
| SAM2 zero-shot + box prompt      | 不微调             |
| SAM2-LoRA + GT box prompt        | 验证 LoRA 上限     |
| SAM2-LoRA + coarse box prompt    | 实际推理设置       |
| SegFormer + SAM2-LoRA refinement | 推荐最终方法       |

注意：

```text
SAM2-LoRA + GT box prompt
```

只能作为“上限实验”，不能作为真实推理性能，因为真实推理没有 GT box。

真实部署要用：

```text
coarse segmentation generated box
```

或：

```text
automatic mask proposals
```

------

# 二十五、消融实验设计

建议写这些消融：

## 1. LoRA rank 消融

```text
r = 4, 8, 16, 32
```

一般：

```text
r=4 参数少但表达不足
r=8 稳健
r=16 精度更高但过拟合风险增加
r=32 不一定值得
```

## 2. 注入位置消融

```text
q_proj + v_proj
q_proj + k_proj + v_proj
q_proj + k_proj + v_proj + out_proj
image encoder only
image encoder + mask decoder
```

第一版建议：

```text
q_proj + v_proj
```

正式实验可加：

```text
image encoder + mask decoder
```

## 3. prompt 类型消融

```text
box prompt
point prompt
box + point prompt
coarse mask prompt
```

Potsdam 中 building、impervious surface 用 box 效果较稳定；car 可能需要 point + box。

## 4. 输入模态消融

```text
RGB
IRRG
RGBIR 转 3 通道组合
RGB + DSM 后处理
```

由于 SAM2 默认 3 通道输入，RGBIR 不建议直接 4 通道输入。可以比较：

```text
RGB = R,G,B
IRRG = NIR,R,G
```

## 5. patch size 消融

```text
512
768
1024
```

car 类可能更喜欢 512；building/tree 可能更喜欢 1024。

------

# 二十六、常见问题与解决方案

## 问题 1：LoRA 训练 loss 降了，但 mIoU 不高

原因通常是：

```text
SAM2 学到的是二值 mask，不知道多类别之间如何竞争
类别融合策略不合理
同一区域多个类别 mask 重叠
```

解决：

```text
使用 coarse segmentation 提供类别先验
使用 per-class probability 做 argmax
对重叠区域按 mask score / coarse logits 融合
```

## 问题 2：car 类很差

原因：

```text
car 面积太小
1024 patch 中 car 占比极低
SAM2 大尺度输入容易忽略小物体
```

解决：

```text
car 类过采样
car 类 min_fg_pixels 降低到 16
增加 512 patch 训练
使用 point + box prompt
loss 加 Focal
```

## 问题 3：building 和 impervious surface 混淆

原因：

```text
屋顶和硬化地面光谱相似
只用 RGB 时高度信息缺失
```

解决：

```text
使用 DSM/nDSM 后处理
高于地面的规则区域更偏 building
近地面大面积区域更偏 impervious surface
```

Potsdam 文档也指出 nDSM 表示距地面高度，有利于区分建筑物、树木等立地对象。

## 问题 4：tree 和 low vegetation 混淆

原因：

```text
RGB 光谱相似
高度差异才是关键
```

解决：

```text
IRRG 输入
DSM/nDSM 辅助
后处理中用高度阈值区分 tree 和 low vegetation
```

## 问题 5：边界区域影响训练

解决：

```text
优先使用 noBoundary 标签
如果用含 boundary 标签，把 RGB=(255,0,0) 映射为 ignore_index=255
loss 和 metric 都忽略 255
```

------

# 二十七、最终推荐的最小可行实验方案

你可以先按这个跑通：

```text
数据：Potsdam RGB + noBoundary label
类别：6 类
切片：1024×1024, stride=512
训练方式：类别级二值 mask
prompt：GT mask 自动生成 box prompt
模型：sam2.1_hiera_base_plus
LoRA：image encoder q_proj/v_proj
rank：8
alpha：16
loss：BCE + Dice + 0.5 Focal
epoch：50
评估：per-class IoU、mIoU、mF1、OA
```

然后做实际推理版本：

```text
SegFormer coarse segmentation
→ connected components
→ box prompts
→ SAM2-LoRA refinement
→ final semantic map
```

------

# 二十八、可以写进论文的方法描述

你可以这样写：

> 针对 Potsdam 高分辨率遥感影像尺寸大、类别边界复杂和小目标占比低等特点，本文将 SAM2 的通用提示式分割能力与 LoRA 参数高效微调相结合。首先，将 Potsdam 的 RGB 标签颜色映射为 6 类语义标签，并将 boundary / don't care 区域设置为忽略标签。随后，将 6000×6000 原始瓦片切分为 1024×1024 重叠 patch。训练阶段，将多类别语义分割任务转换为类别级二值 mask 预测任务，由每个类别 mask 自动生成 box prompt，并冻结 SAM2 主体参数，仅在 image encoder 的 attention projection 层中注入 LoRA 参数。模型采用 BCE、Dice 和 Focal 组成的联合损失进行优化，以缓解遥感小目标和类别不平衡问题。推理阶段，利用粗分割结果生成类别候选框，并调用 SAM2-LoRA 对各类别边界进行细化，最终融合得到 6 类语义分割图。

------

# 二十九、完整执行顺序

```text
1. 安装 Python / PyTorch / SAM2
2. 下载 sam2.1_hiera_base_plus checkpoint
3. 整理 Potsdam 原始目录
4. 读取 noBoundary 标签
5. 标签 RGB 颜色转 class id
6. 按 train_tiles.txt / val_tiles.txt 划分瓦片
7. 6000×6000 大图切成 1024 patch
8. 构建类别级二值 prompt dataset
9. 检查 SAM2 模块名
10. 冻结 SAM2 主体
11. 对 image encoder q_proj/v_proj 注入 LoRA
12. 只优化 LoRA 参数
13. 使用 BCE + Dice + Focal 训练
14. 每个 epoch 保存 LoRA-only checkpoint
15. 用验证集计算 per-class IoU / mIoU / mF1 / OA
16. 推理时滑窗预测并拼接
17. 与 U-Net / DeepLabV3+ / SegFormer / SAM2 zero-shot 对比
18. 做 rank、prompt、patch size、输入模态消融
```

------

# 三十、我对你这个任务的关键建议

最重要的一点是：**不要把 SAM2-LoRA 直接当成普通 6 类语义分割网络使用。**

更专业、更容易成功的定位是：

```text
SAM2-LoRA = 遥感目标 mask 细化器 / promptable segmentation adapter
SegFormer 或其他语义模型 = 类别判别器
Potsdam 标签 = 用于构造类别级 prompt-mask 训练样本
```

这样你的方法会更合理，也更容易在论文中解释：

```text
传统语义分割模型负责类别语义；
SAM2-LoRA 负责高质量边界；
LoRA 负责遥感域适配；
DSM / IRRG 负责提供高度和植被先验。
```

这套逻辑比“直接拿 SAM2 微调 Potsdam”更稳，也更符合 Potsdam 多类别遥感语义分割的特点。
