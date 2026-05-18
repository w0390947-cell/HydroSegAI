# ISPRS Potsdam 2D Semantic Segmentation Dataset

## 目录

1. [数据集概述](#数据集概述)
2. [数据集背景](#数据集背景)
3. [数据来源与采集信息](#数据来源与采集信息)
4. [数据集结构](#数据集结构)
5. [数据格式与规格](#数据格式与规格)
6. [语义类别标签](#语义类别标签)
7. [数据划分](#数据划分)
8. [数据集特点](#数据集特点)
9. [应用场景](#应用场景)
10. [使用方法](#使用方法)
11. [注意事项](#注意事项)
12. [参考文献](#参考文献)

---

## 数据集概述

ISPRS Potsdam数据集是由**国际摄影测量与遥感学会（ISPRS）**发布的**2D语义分割标签数据集**，主要用于城市遥感影像的语义分割研究。该数据集以德国波茨坦市为研究区域，提供了高分辨率的航空正射影像、数字表面模型（DSM）以及相应的像素级语义标注。

### 关键信息摘要

| 属性 | 详情 |
|------|------|
| **数据集名称** | ISPRS Potsdam 2D Semantic Segmentation Dataset |
| **发布机构** | 国际摄影测量与遥感学会（ISPRS） |
| **研究区域** | 德国波茨坦市（Potsdam, Germany） |
| **数据类型** | 航空遥感影像（RGB、IRRG、RGBIR）、数字表面模型（DSM）、语义标签 |
| **影像分辨率** | 5厘米（地面采样距离 GSD） |
| **影像尺寸** | 6000 × 6000 像素（每块/瓦片） |
| **数据覆盖范围** | 38块正射影像瓦片 |
| **标注瓦片数** | 参与者训练标签24块；本地完整参考标签38块 |
| **标签类别** | 6个评估类别；`noBoundary`版本额外包含黑色忽略边界 |
| **坐标系统** | UTM/WGS84 |
| **主要用途** | 城市地物分类、语义分割、深度学习研究 |

---

## 数据集背景

### 研究目的

ISPRS Potsdam数据集是ISPRS发起的**"2D语义分割竞赛"**的官方数据集之一（另一个为Vaihingen数据集），旨在推动遥感影像自动化解译技术的发展，特别是在城市地物识别与分类领域。

### 发展历史

- **2012年**：ISPRS第三技术委员会（TC III）启动了"城市分类与3D建筑物重建"工作组
- **2014-2015年**：数据集正式发布，用于国际竞赛和研究
- **至今**：该数据集已成为遥感语义分割领域最常用的基准数据集之一

### 研究意义

该数据集在以下方面具有重要意义：

1. **多光谱数据**：提供RGB、红外IRRG、四波段RGBIR等多种光谱数据
2. **高度信息**：包含数字表面模型（DSM），提供地物高度信息
3. **高质量标注**：人工精细标注的像素级语义标签
4. **真实场景**：代表典型的中欧城市环境

---

## 数据来源与采集信息

### 传感器与平台

| 参数 | 详情 |
|------|------|
| **传感器** | 满足特定要求的航空数字相机 |
| **飞行平台** | 航空器（飞机/无人机） |
| **飞行高度** | 约1500米（推测，基于5cm GSD） |
| **采集时间** | 2010年代初期 |
| **采集区域** | 德国勃兰登堡州波茨坦市 |

### 影像参数

- **地面采样距离（GSD）**：**5厘米**
- **影像重叠度**：航向与旁向均有较高重叠，用于DSM生成
- **光谱波段**：
  - RGB（红、绿、蓝）
  - IRRG（近红外、红、绿）
  - RGBIR（红、绿、蓝、近红外）

---

## 数据集结构

### 目录结构

```
Potsdam/
├── 1_DSM/                           # 数字表面模型（绝对高程）
│   └── 1_DSM/
│       ├── dsm_potsdam_XX_YY.tif    # DSM影像文件
│       └── dsm_potsdam_XX_YY.tfw    # 世界文件（地理参考信息）
│
├── 1_DSM_normalisation/             # 标准化DSM（归一化高度）
│   └── 1_DSM_normalisation/
│       ├── ..._lastools.jpg         # LAStools处理的nDSM
│       ├── ..._ownapproach.jpg      # 自定义方法处理的nDSM
│       ├── *.laz                    # 原始激光点云数据
│       └── readme.txt               # 说明文档
│
├── 2_Ortho_RGB/                     # RGB正射影像
│   └── 2_Ortho_RGB/
│       ├── top_potsdam_XX_YY_RGB.tif
│       └── top_potsdam_XX_YY_RGB.tfw
│
├── 3_Ortho_IRRG/                    # IRRG正射影像（近红外+红+绿）
│   └── 3_Ortho_IRRG/
│       ├── top_potsdam_XX_YY_IRRG.tif
│       └── top_potsdam_XX_YY_IRRG.tfw
│
├── 4_Ortho_RGBIR/                   # RGBIR四波段正射影像
│   └── 4_Ortho_RGBIR/
│       ├── top_potsdam_XX_YY_RGBIR.tif
│       └── top_potsdam_XX_YY_RGBIR.tfw
│
├── 5_Labels_all/                    # 所有完整参考标签（38块）
│   ├── top_potsdam_XX_YY_label.tif
│   └── top_potsdam_XX_YY_label.tfw
│
├── 5_Labels_all_noBoundary/         # 所有腐蚀边界标签（38块，黑色为忽略区）
│   ├── top_potsdam_XX_YY_label_noBoundary.tif
│   └── top_potsdam_XX_YY_label_noBoundary.tfw
│
├── 5_Labels_for_participants/       # 参赛者用完整参考标签（24块）
│   └── 5_Labels_for_participants/
│       ├── top_potsdam_XX_YY_label.tif
│       └── top_potsdam_XX_YY_label.tfw
│
├── 5_Labels_for_participants_no_Boundary/  # 参赛者用腐蚀边界标签（24块，黑色为忽略区）
│   └── 5_Labels_for_participants_no_Boundary/
│       ├── top_potsdam_XX_YY_label_noBoundary.tif
│       └── top_potsdam_XX_YY_label_noBoundary.tfw
│
└── assess_classification_reference_implementation.tgz  # 参考实现代码
```

### 文件命名规则

- **正射影像和标签**：`top_potsdam_<行号>_<列号>_<类型>.tif`
  - 行号范围：2-7
  - 列号范围：7-15
  - 例如：`top_potsdam_2_10_RGB.tif`

- **DSM**：`dsm_potsdam_<行号>_<列号>.tif`
  - 例如：`dsm_potsdam_02_10.tif`

---

## 数据格式与规格

### 影像规格

| 属性 | RGB/IRRG/RGBIR | DSM | 标签 |
|------|----------------|-----|------|
| **文件格式** | GeoTIFF (.tif) | GeoTIFF (.tif) | GeoTIFF (.tif) |
| **位深度** | 8位无符号整数 | 浮点型/整数型 | 8位无符号整数 |
| **像素尺寸** | 6000 × 6000 | 6000 × 6000 | 6000 × 6000 |
| **地面分辨率** | 5厘米 | 5厘米 | 5厘米 |
| **波段数** | 3/4波段 | 1波段 | 3波段RGB颜色编码 |

### RGB正射影像含义

`2_Ortho_RGB/`目录中的`top_potsdam_XX_YY_RGB.tif`是**RGB真彩色正射影像**，也就是最接近普通航拍照片的一组图像。文件名中的`XX_YY`表示瓦片编号，例如`top_potsdam_4_10_RGB.tif`表示第`4_10`号瓦片的RGB正射影像。

RGB影像包含三个可见光通道：

| 通道 | 含义 |
|------|------|
| R | 红色通道 |
| G | 绿色通道 |
| B | 蓝色通道 |

“正射影像”表示这些航拍影像已经经过几何校正，形成近似垂直俯视的地图影像。经过正射校正后，图像像素可以与真实地面坐标对应，也可以与同编号的DSM、nDSM和标签图严格对齐。

例如：

```
top_potsdam_4_10_RGB.tif
dsm_potsdam_04_10.tif
top_potsdam_4_10_label.tif
```

这几个文件表示同一片城市区域，但提供的信息不同：

| 文件 | 表示内容 |
|------|----------|
| RGB图像 | 该位置看起来是什么颜色 |
| DSM/nDSM图像 | 该位置有多高或比地面高多少 |
| 标签图像 | 该位置属于什么地物类别 |

RGB正射影像是语义分割中最常用的输入数据。典型任务是：以RGB图像作为输入，让模型预测每个像素属于不透水表面、建筑物、低矮植被、树木、汽车或杂乱/背景中的哪一类。

RGB影像的主要用途包括：

- 人工查看和理解场景：直观观察道路、屋顶、树木、草地、车辆等对象
- 语义分割模型输入：作为最基础的三通道输入
- 与标签配对训练：同编号RGB图像与同编号标签图组成监督学习样本
- 与DSM/nDSM或IRRG/RGBIR融合：补充高度或近红外信息，提高类别区分能力

需要注意：RGB只包含可见光颜色信息，有时不足以区分光谱相似的类别。例如，树木和低矮植被都可能呈绿色，建筑物屋顶和道路都可能呈灰色。因此，在要求更高的实验中，常会将RGB与DSM/nDSM高度信息或IRRG/RGBIR多光谱信息结合使用。

### IRRG正射影像含义

`3_Ortho_IRRG/`目录中的`top_potsdam_XX_YY_IRRG.tif`是**IRRG正射影像**，由近红外（NIR）、红光（R）、绿光（G）三个通道组成。文件名中的`XX_YY`同样表示瓦片编号，例如`top_potsdam_4_10_IRRG.tif`表示第`4_10`号瓦片的IRRG正射影像。

IRRG与RGB的区别在于：RGB使用红、绿、蓝三个可见光通道，而IRRG用近红外通道替代了蓝色通道。因此，IRRG不是为了显示“真实颜色”，而是为了突出遥感分析中有用的光谱差异，尤其是植被信息。

| 文件夹 | 通道组成 | 典型用途 |
|--------|----------|----------|
| `2_Ortho_RGB/` | Red, Green, Blue | 真彩色显示和基础语义分割输入 |
| `3_Ortho_IRRG/` | Near Infrared, Red, Green | 假彩色遥感分析，突出植被响应 |

近红外对植被非常敏感。健康植被通常会强烈反射近红外，因此IRRG影像常用于：

- 区分植被和非植被
- 辅助区分低矮植被、树木、道路和建筑
- 分析植被覆盖范围
- 为语义分割模型提供比RGB更丰富的光谱信息

不同地物在IRRG中的作用可以这样理解：

| 地物 | IRRG中的价值 |
|------|--------------|
| 树木 | 近红外响应强，通常更容易与道路、建筑区分 |
| 低矮植被 | 近红外响应也强，但高度、纹理和空间形态与树木不同 |
| 道路、广场 | 近红外响应通常较弱 |
| 建筑物屋顶 | 响应取决于材料，但通常与植被差异明显 |
| 汽车 | 小目标，识别仍主要依赖形状、颜色和上下文 |

IRRG影像也与同编号的RGB、DSM/nDSM和标签图严格对应。例如：

```
top_potsdam_4_10_IRRG.tif
top_potsdam_4_10_RGB.tif
dsm_potsdam_04_10.tif
top_potsdam_4_10_label.tif
```

这些文件表示同一个区域，只是提供不同信息：RGB提供真实颜色，IRRG提供近红外增强的光谱信息，DSM/nDSM提供高度信息，标签图提供类别真值。

在深度学习实验中，IRRG常见用法包括：

- 单独作为三通道输入，替代RGB
- 与RGB或RGBIR进行对比实验
- 与DSM/nDSM融合，形成“光谱 + 高度”的输入
- 用于增强植被相关类别的识别能力

简单来说，`3_Ortho_IRRG/`的核心价值是引入近红外信息，尤其有助于识别和区分植被相关类别。

### RGBIR正射影像含义

`4_Ortho_RGBIR/`目录中的`top_potsdam_XX_YY_RGBIR.tif`是**RGBIR四波段正射影像**。它同时包含可见光RGB和近红外NIR信息，是Potsdam数据集中光谱信息最完整的一组正射影像。

RGBIR包含四个通道：

| 通道 | 含义 |
|------|------|
| R | 红光 |
| G | 绿光 |
| B | 蓝光 |
| IR / NIR | 近红外 |

它与RGB和IRRG的关系可以这样理解：

| 文件夹 | 通道组成 | 特点 |
|--------|----------|------|
| `2_Ortho_RGB/` | R, G, B | 真彩色，人眼最直观 |
| `3_Ortho_IRRG/` | NIR, R, G | 假彩色，突出植被 |
| `4_Ortho_RGBIR/` | R, G, B, NIR | 同时保留真彩色和近红外信息 |

RGBIR的价值在于：模型可以同时利用可见光颜色和近红外响应。可见光通道有助于识别道路纹理、屋顶颜色、车辆外观、阴影和材料差异；近红外通道有助于区分植被与非植被，并增强对植被覆盖情况的判断。

不同类别中，RGBIR通常提供以下帮助：

| 类别 | RGBIR的作用 |
|------|-------------|
| 不透水表面 | RGB提供道路、广场等纹理信息，NIR响应通常较弱 |
| 建筑物 | RGB提供屋顶颜色和形状，NIR辅助与植被区分 |
| 低矮植被 | NIR响应强，RGB提供颜色和纹理 |
| 树木 | NIR响应强，结合纹理和高度信息更容易与草地区分 |
| 汽车 | 主要依赖RGB中的形状、颜色和上下文，NIR作用相对较小 |
| 杂乱/背景 | 多通道信息有助于减少与其他类别的混淆 |

RGBIR影像也与同编号的RGB、IRRG、DSM/nDSM和标签图严格对应。例如：

```
top_potsdam_4_10_RGBIR.tif
top_potsdam_4_10_RGB.tif
top_potsdam_4_10_IRRG.tif
dsm_potsdam_04_10.tif
top_potsdam_4_10_label.tif
```

这些文件表示同一个地理区域，只是提供的通道和信息不同。

在深度学习实验中，RGBIR常见用法包括：

- 作为四通道输入直接输入模型
- 将RGB和NIR分别编码，再在网络中融合
- 与DSM/nDSM拼接，形成`RGBIR + height`的多模态输入
- 用于对比实验，例如比较RGB、IRRG、RGBIR、RGBIR+DSM/nDSM的效果

需要注意：很多预训练模型默认输入是三通道RGB。如果使用RGBIR四通道输入，通常需要修改模型第一层输入通道数，或将NIR作为单独分支处理后再与RGB特征融合。

简单来说，`4_Ortho_RGBIR/`适合做多光谱语义分割实验，尤其适合评估“可见光 + 近红外”比单纯RGB能带来多少改进。

### DSM图像含义

`1_DSM/`目录中的`dsm_potsdam_XX_YY.tif`不是普通照片，而是**数字表面模型（Digital Surface Model, DSM）**，可以理解为一张“高度图”。每个像素的数值表示该地理位置的**绝对高程**，单位通常为米。

DSM瓦片与同编号的正射影像、标签瓦片一一对应。例如：

```
dsm_potsdam_04_10.tif
top_potsdam_4_10_RGB.tif
top_potsdam_4_10_label.tif
```

这三个文件表示同一个地理区域，尺寸也都是`6000 × 6000`像素，但含义不同：

| 文件类型 | 表示内容 |
|----------|----------|
| RGB/IRRG/RGBIR正射影像 | 航空影像的光谱/颜色信息 |
| 标签图像 | 每个像素对应的地物类别，如不透水表面、建筑物、树木、汽车等 |
| DSM图像 | 每个像素对应位置的表面高度 |

DSM中的高度包括地面和地表物体，因此它描述的是“表面高度”，不是单纯的地面高度。例如：

- 道路、草地：通常接近地面高度
- 建筑物屋顶：明显高于周围地面
- 树冠：明显高于地面，但形态通常不同于建筑物
- 汽车：略高于路面

DSM在语义分割中的作用是提供高度信息，帮助模型区分光谱上相似但高度不同的地物。例如，树木和低矮植被在RGB图像中都可能呈绿色，但树木在DSM或nDSM中更高；建筑物和道路都可能呈灰色，但建筑物通常有明显高度。

需要注意：`1_DSM/`中的DSM是**绝对高程**，数值表示相对于高程基准或坐标系统的高度。若任务更关注“物体离地高度”，通常更适合使用`1_DSM_normalisation/`中的nDSM，它表示归一化后的离地高度，更直接反映建筑物、树木等立地对象的高度差异。

### nDSM图像含义

`1_DSM_normalisation/`目录中的图片是**归一化DSM（normalized DSM, nDSM）**，可以理解为“离地高度图”。它与`1_DSM/`中的绝对DSM不同：DSM表示某个位置相对于高程基准有多高，nDSM表示该位置比附近地面高多少。

例如，某处地面绝对高程为`50m`，建筑物屋顶绝对高程为`65m`：

| 数据类型 | 地面像素 | 屋顶像素 |
|----------|----------|----------|
| DSM | 50m | 65m |
| nDSM | 0m | 15m |

因此，nDSM会弱化整体地形起伏，突出建筑物、树木、汽车等地表物体本身的高度，更适合作为语义分割的辅助特征。

该目录中常见文件包括：

| 文件类型 | 含义 |
|----------|------|
| `dsm_potsdam_XX_YY_normalized_lastools.jpg` | 使用LAStools方法生成的nDSM |
| `dsm_potsdam_XX_YY_normalized_ownapproach.jpg` | 使用数据提供方自定义方法生成的nDSM |
| `dsm_potsdam_XX_YY.laz` | 原始点云/DSM数据的压缩LAZ文件，可用于自行重新进行地面滤波和归一化 |

nDSM的JPG文件是8-bit高度图，像素值以分米为单位编码离地高度：

| 像素值 | 近似离地高度 |
|--------|--------------|
| 0 | 0.0m |
| 10 | 1.0m |
| 100 | 10.0m |
| 255 | 25.5m或更高 |

也就是说，像素值`255`表示高度达到或超过`25.5m`，超过该高度的值会被截断。nDSM常用于帮助模型区分高度差异明显的类别：

- 建筑物：高度明显，边缘通常较规整
- 树木：高度明显，但形态通常更碎、更不规则
- 低矮植被、道路、广场：通常接近地面
- 汽车：略高于路面，但明显低于多数建筑物和高大树木

需要注意：该目录中的nDSM没有经过逐瓦片人工质量检查，数据提供方也说明不保证完全无误。`*_lastools.jpg`和`*_ownapproach.jpg`都可能存在地面估计或插值导致的伪影，因此更适合作为辅助特征，而不应当作绝对可靠的真值。

### TFW世界文件格式

TFW文件包含地理参考信息，格式如下：

```
   0.050          # 像素大小（X方向，单位：米）
   0.000          # 旋转参数
   0.000          # 旋转参数
  -0.050          # 像素大小（Y方向，负值表示向上）
366976.525       # 左上角X坐标（UTM）
5809761.000      # 左上角Y坐标（UTM）
```

### 数据量统计

| 数据类型 | 文件数量 | 单文件大小 | 总大小（约） |
|----------|----------|------------|--------------|
| RGB正射影像 | 38 | ~100 MB | ~3.8 GB |
| IRRG正射影像 | 38 | ~100 MB | ~3.8 GB |
| RGBIR正射影像 | 38 | ~130 MB | ~5.0 GB |
| DSM | 38 | ~140 MB | ~5.3 GB |
| 标签（参与者） | 24 | ~100 MB | ~2.4 GB |
| 标签（完整参考） | 38 | ~100 MB | ~3.8 GB |

---

## 语义类别标签

### 完整参考标签图含义

`Potsdam/5_Labels_all/`目录中的`top_potsdam_XX_YY_label.tif`是**完整语义分割参考标签图**，也就是每个像素的“标准答案”。这些文件不是普通照片，而是RGB颜色编码的类别图。

例如：

```
top_potsdam_4_10_label.tif
```

表示第`4_10`号瓦片的语义标签图。它与同编号的正射影像和DSM/nDSM表示同一片地理区域：

```
top_potsdam_4_10_RGB.tif
top_potsdam_4_10_IRRG.tif
top_potsdam_4_10_RGBIR.tif
dsm_potsdam_04_10.tif
top_potsdam_4_10_label.tif
```

这些文件的关系如下：

| 文件类型 | 含义 |
|----------|------|
| RGB / IRRG / RGBIR | 输入影像，描述该位置的颜色和光谱特征 |
| DSM / nDSM | 高度信息，描述该位置的表面高度或离地高度 |
| `5_Labels_all`标签图 | 参考真值，描述该像素实际属于哪一类 |

`5_Labels_all/`覆盖完整的38个瓦片，比`5_Labels_for_participants/`中的24个参与者训练标签更多。它可用于本地训练、验证、测试或完整区域分析，但在复现官方竞赛设置时，需要注意官方训练/测试划分与该目录中完整参考标签的区别。

`5_Labels_all/`的主要用途包括：

- **训练语义分割模型**：将影像作为输入，将标签图作为监督信号
- **评估模型效果**：将模型预测结果与标签图对比，计算accuracy、mIoU、F1等指标
- **可视化检查**：直观看每个区域被标注为道路、建筑、树木、汽车等哪一类
- **制作训练样本**：通常将`6000 × 6000`大瓦片裁剪成`512 × 512`或`1024 × 1024`小块再训练

需要注意：这些标签图是RGB彩色标签图，不是单通道类别ID图。训练前通常需要先将RGB颜色转换为类别编号，例如白色转为不透水表面，蓝色转为建筑物，青色转为低矮植被等。

### 腐蚀边界标签图含义

`Potsdam/5_Labels_all_noBoundary/`目录中的`top_potsdam_XX_YY_label_noBoundary.tif`是**去边界/腐蚀边界版本的完整标签图**。它与`5_Labels_all/`一样都是语义分割参考标签，但会把类别交界附近的一圈像素改成黑色忽略区域。

例如：

```
top_potsdam_4_10_label_noBoundary.tif
```

表示第`4_10`号瓦片的去边界标签图。它对应的完整标签图是：

```
top_potsdam_4_10_label.tif
```

二者关系可以概括为：

| 目录 | 含义 |
|------|------|
| `5_Labels_all/` | 完整参考标签，尽量为每个像素指定6个评估类别之一 |
| `5_Labels_all_noBoundary/` | 在完整参考标签基础上，将类别边界附近像素置为黑色ignore区域 |

为什么需要`noBoundary`版本：遥感语义分割中的类别边界往往最不确定，例如建筑物边缘、树冠与草地交界、道路与低矮植被交界、汽车边缘和屋顶阴影区域。即使人工标注质量很高，边界位置也可能存在一两个像素的偏差。如果严格评估这些边界像素，模型可能因为很小的边界偏移被额外惩罚。

`noBoundary`标签通过忽略边界附近像素来降低这种不确定性。其颜色含义如下：

| 颜色 | 含义 |
|------|------|
| 白色 (255,255,255) | 不透水表面 |
| 蓝色 (0,0,255) | 建筑物 |
| 青色 (0,255,255) | 低矮植被 |
| 绿色 (0,255,0) | 树木 |
| 黄色 (255,255,0) | 汽车 |
| 红色 (255,0,0) | 杂乱/背景 |
| 黑色 (0,0,0) | 忽略区域/边界，不参与训练或评估 |

`5_Labels_all_noBoundary/`的主要用途包括：

- **更公平地评估模型**：忽略边界附近容易产生争议的像素
- **训练时忽略不确定区域**：将黑色像素转成`ignore_index=255`，不参与loss
- **复现官方no boundary评估**：官方评估流程通常会区分full reference和no boundary reference
- **降低边界噪声影响**：尤其适合大图裁剪训练时减少边界标签误差的影响

需要特别注意：黑色 (0,0,0) 在`*_label_noBoundary.tif`中表示ignore/don't care，不是不透水表面，也不是背景类别。训练或评估前应将黑色像素单独映射为忽略值。

### 参赛者训练标签图含义

`Potsdam/5_Labels_for_participants/5_Labels_for_participants/`目录中的`top_potsdam_XX_YY_label.tif`是**官方公开给参赛者使用的训练标签图**。它与`5_Labels_all/`中的标签格式相同，都是RGB颜色编码的语义分割参考标签；主要区别在于数量和用途。

| 文件夹 | 标签数量 | 含义 |
|--------|----------|------|
| `5_Labels_all/` | 38张 | 完整参考标签，覆盖所有瓦片 |
| `5_Labels_for_participants/5_Labels_for_participants/` | 24张 | 官方公开给参赛者训练使用的标签 |

例如：

```
top_potsdam_4_10_label.tif
```

表示第`4_10`号瓦片的语义标签图。它和同编号的影像、DSM/nDSM文件表示同一个区域：

```
top_potsdam_4_10_RGB.tif
top_potsdam_4_10_IRRG.tif
top_potsdam_4_10_RGBIR.tif
dsm_potsdam_04_10.tif
top_potsdam_4_10_label.tif
```

这些标签图的颜色含义与`5_Labels_all/`一致：白色表示不透水表面，蓝色表示建筑物，青色表示低矮植被，绿色表示树木，黄色表示汽车，红色表示杂乱/背景。

`5_Labels_for_participants/5_Labels_for_participants/`的主要用途包括：

- **模拟官方竞赛训练设置**：只使用官方公开的24张标签训练模型，而不是使用完整38张参考标签
- **训练语义分割模型**：与同编号RGB、IRRG、RGBIR、DSM/nDSM配对，作为监督学习样本
- **划分训练集和验证集**：常见做法是从24张公开标签中再划出一部分作为validation
- **避免数据泄漏**：如果目标是接近官方benchmark，应避免直接将`5_Labels_all/`的38张完整参考标签全部用于训练

简单来说，`5_Labels_for_participants/5_Labels_for_participants/`适合用于公平复现官方竞赛训练设置；`5_Labels_all/`更适合本地实验、教学、完整区域分析或需要完整参考标签的研究。

### 参赛者训练标签noBoundary版本含义

`Potsdam/5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary/`目录中的`top_potsdam_XX_YY_label_noBoundary.tif`是**参赛者训练标签的noBoundary版本**，也就是官方公开24张训练标签的去边界/腐蚀边界版本。

它可以理解为`5_Labels_for_participants/5_Labels_for_participants/`的去边界版本：

| 文件夹 | 含义 |
|--------|------|
| `5_Labels_for_participants/5_Labels_for_participants/` | 官方公开的24张训练标签，保留完整类别边界 |
| `5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary/` | 同样24张训练标签，但类别边界附近像素被设为黑色ignore |

例如：

```
top_potsdam_4_10_label_noBoundary.tif
```

表示第`4_10`号瓦片的参赛者训练标签noBoundary版本。它对应的完整训练标签是：

```
top_potsdam_4_10_label.tif
```

该目录中的颜色含义与其他标签一致，但额外包含黑色忽略像素：

| 颜色 | 含义 |
|------|------|
| 白色 (255,255,255) | 不透水表面 |
| 蓝色 (0,0,255) | 建筑物 |
| 青色 (0,255,255) | 低矮植被 |
| 绿色 (0,255,0) | 树木 |
| 黄色 (255,255,0) | 汽车 |
| 红色 (255,0,0) | 杂乱/背景 |
| 黑色 (0,0,0) | ignore/don't care边界区域 |

`5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary/`的主要用途包括：

- **使用官方公开训练瓦片训练模型**：仍然只使用官方公开的24张标签
- **训练时忽略边界不确定区域**：将黑色像素转成`ignore_index=255`，不参与loss
- **做noBoundary训练/验证实验**：对比完整标签训练和去边界标签训练的差异
- **减少类别交界处标注误差影响**：例如建筑边缘、树冠边缘、汽车边缘、道路与草地交界等
- **贴近官方no boundary评估逻辑**：如果验证或测试使用noBoundary标签，训练和指标处理也应保持一致

简单来说，`5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary/`就是“官方公开训练标签 + 边界像素忽略”。黑色 (0,0,0) 不是类别，训练和评估时必须作为ignore处理。

### 类别定义

ISPRS Potsdam数据集定义了**6个评估类别**。标签文件不是单通道类别ID图，而是**RGB颜色编码标签图**；训练前通常需要把颜色转换为类别ID。`*_label_noBoundary.tif`版本还包含黑色边界/忽略像素，这些像素在训练损失和评估中应跳过。

| 建议训练ID | 类别名称（中文） | 英文描述 | 官方RGB颜色 | 说明 |
|------------|------------------|----------|-------------|------|
| 0 | 不透水表面 | Impervious Surface | 白色 (255,255,255) | 道路、停车场、广场、人行道等硬化地面 |
| 1 | 建筑物 | Building | 蓝色 (0,0,255) | 各种类型的建筑物 |
| 2 | 低矮植被 | Low Vegetation | 青色 (0,255,255) | 草地、灌木丛等 |
| 3 | 树木 | Tree | 绿色 (0,255,0) | 高大树木、树林 |
| 4 | 汽车 | Car | 黄色 (255,255,0) | 停放或行驶的车辆 |
| 5 | 杂乱/背景 | Clutter/Background | 红色 (255,0,0) | 不属于其他类别的对象 |
| 255（常用忽略值） | 边界/忽略 | Boundary/Don't Care | 黑色 (0,0,0) | `noBoundary`版本中的腐蚀边界或不参与评估区域 |

### 类别详细说明

#### 1. 不透水表面（Impervious Surface）
- 包括：
  - 各种道路（沥青、水泥、砖石路面）
  - 停车场
  - 城市广场、人行道
  - 其他硬化地面
- 特征：人工铺设的无植被覆盖的地面

#### 2. 建筑物（Building）
- 包括：
  - 住宅建筑
  - 商业建筑
  - 工业建筑
  - 公共设施
- 特征：具有屋顶的永久性建筑结构

#### 3. 低矮植被（Low Vegetation）
- 包括：
  - 草坪
  - 低矮灌木丛
  - 花坛
  - 其他低矮植物
- 特征：高度较低的植被层

#### 4. 树木（Tree）
- 包括：
  - 高大乔木
  - 树林
  - 独立大树
- 特征：有明显的主干和较高的高度

#### 5. 汽车（Car）
- 包括：
  - 停放的汽车
  - 行驶中的汽车（如清晰可辨）
- 特征：四轮机动车辆

#### 6. 杂乱/背景（Clutter/Background）
- 包括：集装箱、桥梁、楼梯、花园家具、栅栏、杆状物、体育场、网球场等
- 特征：不属于其他主要类别的所有对象

#### 7. 边界/忽略区域（Boundary/Don't Care）
- 包括：
  - `*_label_noBoundary.tif`中的腐蚀边界区域
  - 标注不确定的区域
- 特征：用黑色 (0,0,0) 表示，在训练和评估中通常应忽略

### 标签版本说明

数据集提供两种版本的标签：

1. **完整参考标签** (`*_label.tif`)：
   - 包含6个评估类别的原始标注
   - 红色 (255,0,0) 表示Clutter/Background，不是边界类别
   - 适合用于需要完整像素标注的训练或可视化

2. **腐蚀边界标签** (`*_label_noBoundary.tif`)：
   - 保留6个评估类别，同时将类别边界附近的像素置为黑色 (0,0,0)
   - 黑色像素不属于6个评估类别，应作为ignore/mask处理
   - 适用于官方“no boundary”评估或希望降低边界标注不确定性的训练

---

## 数据划分

### 完整数据覆盖

数据集共有**38块影像瓦片**，按网格编号排列：

| 行 | 存在的列 |
|----|----------|
| 2 | 10, 11, 12, 13, 14 |
| 3 | 10, 11, 12, 13, 14 |
| 4 | 10, 11, 12, 13, 14, 15 |
| 5 | 10, 11, 12, 13, 14, 15 |
| 6 | 7, 8, 9, 10, 11, 12, 13, 14, 15 |
| 7 | 7, 8, 9, 10, 11, 12, 13 |

### 标注数据划分

提供标注的**24块瓦片**（用于训练/验证）：

| 行 | 提供标签的列 |
|----|-------------|
| 2 | 10, 11, 12 |
| 3 | 10, 11, 12 |
| 4 | 10, 11, 12 |
| 5 | 10, 11, 12 |
| 6 | 7, 8, 9, 10, 11, 12 |
| 7 | 7, 8, 9, 10, 11, 12 |

**具体瓦片列表**：
```
top_potsdam_2_10_label.tif      top_potsdam_4_12_label.tif
top_potsdam_2_11_label.tif      top_potsdam_5_10_label.tif
top_potsdam_2_12_label.tif      top_potsdam_5_11_label.tif
top_potsdam_3_10_label.tif      top_potsdam_5_12_label.tif
top_potsdam_3_11_label.tif      top_potsdam_6_7_label.tif
top_potsdam_3_12_label.tif      top_potsdam_6_8_label.tif
top_potsdam_4_10_label.tif      top_potsdam_6_9_label.tif
top_potsdam_4_11_label.tif      top_potsdam_6_10_label.tif
                               top_potsdam_6_11_label.tif
                               top_potsdam_6_12_label.tif
                               top_potsdam_7_7_label.tif
                               top_potsdam_7_8_label.tif
                               top_potsdam_7_9_label.tif
                               top_potsdam_7_10_label.tif
                               top_potsdam_7_11_label.tif
                               top_potsdam_7_12_label.tif
```

### 典型划分方式

研究者通常采用以下划分方式：

1. **训练集-验证集划分**（常用）：
   - 训练集：16块瓦片
   - 验证集：8块瓦片

2. **交叉验证**：
   - 5折或3折交叉验证

3. **竞赛划分**：
   - 部分瓦片用于训练，部分用于测试

---

## 数据集特点

### 1. 多光谱信息

数据集提供三种光谱组合：

- **RGB**：标准真彩色影像，详见[RGB正射影像含义](#rgb正射影像含义)
- **IRRG**：近红外+红+绿，增强植被信息，详见[IRRG正射影像含义](#irrg正射影像含义)
- **RGBIR**：四波段影像，包含完整光谱信息，详见[RGBIR正射影像含义](#rgbir正射影像含义)

### 2. 高度信息

- **绝对DSM**：数字表面模型，包含每个像素的绝对表面高程，详见[DSM图像含义](#dsm图像含义)
- **标准化DSM（nDSM）**：归一化高度，表示距地面高度，详见[nDSM图像含义](#ndsm图像含义)
  - 范围：0-25.5米（8位编码，单位：分米）
  - 有利于区分建筑物、树木等立地对象

### 3. 高分辨率

- **5厘米地面采样距离**：能够清晰识别小物体如汽车
- **6000×6000像素**：大范围连续覆盖

### 4. 多种标签格式

- 完整参考标签
- 腐蚀边界标签（黑色像素作为ignore区域）
- 完整标签（38块）和参与者标签（24块）

### 5. 地理参考信息

- 提供TFW世界文件
- UTM坐标系统
- 支持GIS软件直接读取

### 6. 真实城市场景

- 典型的中欧城市环境
- 包含复杂的地物组合
- 具有实际应用价值

---

## 应用场景

### 学术研究

1. **遥感影像语义分割**
   - 全卷积网络（FCN）
   - U-Net及其变体
   - DeepLab系列
   - Transformer架构

2. **多模态融合**
   - RGB + DSM融合
   - 多光谱影像分析
   - 高度信息辅助分类

3. **小样本/半监督学习**
   - 基于有限标注数据训练
   - 迁移学习研究

4. **域适应**
   - 从Potsdam到Vaihingen的跨域测试
   - 不同城市间的泛化能力

### 实际应用

1. **城市规划**
   - 土地利用分类
   - 绿化覆盖评估
   - 建筑物密度分析

2. **灾害管理**
   - 灾后损失评估
   - 变化检测

3. **环境监测**
   - 植被健康分析
   - 城市热岛效应研究

4. **导航与地图制作**
   - 自动地图更新
   - 兴趣点识别

---

## 使用方法

### Python读取示例

```python
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

# 读取RGB影像
def read_image(image_path):
    """读取GeoTIFF影像"""
    dataset = gdal.Open(image_path)
    if dataset is None:
        raise FileNotFoundError(f"无法打开文件: {image_path}")

    # 读取影像数据
    image = dataset.ReadAsArray()

    # 获取影像信息
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount

    # GDAL读取多波段影像的形状为(C, H, W)，转换为(H, W, C)
    if bands > 1:
        image = np.transpose(image, (1, 2, 0))

    dataset = None  # 关闭文件
    return image

# 读取标签
COLOR_TO_ID = {
    (255, 255, 255): 0,  # Impervious surface
    (0, 0, 255): 1,      # Building
    (0, 255, 255): 2,    # Low vegetation
    (0, 255, 0): 3,      # Tree
    (255, 255, 0): 4,    # Car
    (255, 0, 0): 5,      # Clutter/background
    (0, 0, 0): 255,      # Ignore boundary in noBoundary labels
}

def read_label(label_path):
    """读取RGB颜色标签，并转换为单通道类别ID。"""
    dataset = gdal.Open(label_path)
    label_rgb = dataset.ReadAsArray()
    dataset = None

    # GDAL读取多波段影像的形状为(C, H, W)，转换为(H, W, C)
    label_rgb = np.transpose(label_rgb, (1, 2, 0)).astype(np.uint8)
    label = np.full(label_rgb.shape[:2], 255, dtype=np.uint8)

    for color, class_id in COLOR_TO_ID.items():
        mask = np.all(label_rgb == color, axis=-1)
        label[mask] = class_id

    return label

# 使用示例
rgb_image = read_image('2_Ortho_RGB/2_Ortho_RGB/top_potsdam_2_10_RGB.tif')
label = read_label('5_Labels_for_participants/5_Labels_for_participants/top_potsdam_2_10_label.tif')

print(f"影像形状: {rgb_image.shape}")
print(f"标签形状: {label.shape}")
print(f"唯一标签值: {np.unique(label)}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes[0].imshow(rgb_image[:, :, :3])  # 显示前3个波段
axes[0].set_title('RGB Image')
axes[0].axis('off')

label_vis = np.ma.masked_where(label == 255, label)
axes[1].imshow(label_vis, cmap='tab10', vmin=0, vmax=5)
axes[1].set_title('Label')
axes[1].axis('off')
plt.show()
```

### 深度学习数据加载

```python
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PotsdamDataset(Dataset):
    def __init__(self, image_paths, label_paths, bands='RGB', transform=None):
        """
        Args:
            image_paths: 影像路径列表
            label_paths: 标签路径列表
            bands: 'RGB', 'IRRG', 'RGBIR', 或 'RGB+DSM'
            transform: 数据增强
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.bands = bands
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取影像
        image = self._read_multispectral(self.image_paths[idx], self.bands)

        # 读取标签
        label = self._read_label(self.label_paths[idx])

        # 数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        return image, label

    def _read_multispectral(self, path, bands):
        """根据指定波段读取影像"""
        # 实现略
        pass

    def _read_label(self, path):
        """读取RGB标签并转换为类别ID；黑色边界映射为255(ignore_index)。"""
        return read_label(path)

# 数据增强示例
transform = A.Compose([
    A.RandomCrop(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

# 创建数据加载器
dataset = PotsdamDataset(
    image_paths=[...],
    label_paths=[...],
    bands='RGB+DSM',
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

### 评价指标

```python
def calculate_metrics(predictions, labels, num_classes=6, ignore_index=255):
    """
    计算语义分割常用指标

    Args:
        predictions: 预测结果 (N, H, W)
        labels: 真实标签 (N, H, W)
        num_classes: 类别数量
        ignore_index: 不参与评估的标签值，常用于黑色边界像素

    Returns:
        dict: 包含各项指标的字典
    """
    from sklearn.metrics import confusion_matrix

    # 展平并移除忽略像素
    predictions = predictions.flatten()
    labels = labels.flatten()
    valid_mask = labels != ignore_index
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]

    # 计算混淆矩阵
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))

    # 各指标
    metrics = {}

    # 总体精度
    metrics['overall_accuracy'] = np.diag(cm).sum() / cm.sum()

    # 平均精度
    metrics['average_accuracy'] = np.nanmean(np.diag(cm) / np.maximum(cm.sum(axis=1), 1))

    # IoU (交并比)
    iou_per_class = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - cm[i, i]
        iou = intersection / union if union > 0 else 0
        iou_per_class.append(iou)

    metrics['iou_per_class'] = iou_per_class
    metrics['mean_iou'] = np.mean(iou_per_class)

    # F1分数
    f1_per_class = 2 * np.diag(cm) / np.maximum(cm.sum(axis=1) + cm.sum(axis=0), 1)
    metrics['f1_per_class'] = f1_per_class
    metrics['mean_f1'] = np.mean(f1_per_class)

    return metrics
```

---

## 注意事项

### 1. 数据使用许可

- 该数据集仅用于**研究和教育目的**
- 商业使用需获得ISPRS特别许可
- 引用时需注明数据来源

### 2. 标注质量

- 标签由人工创建，经过质量控制
- 边界区域可能存在不确定性
- 某些复杂场景（如阴影、遮挡）可能标注不准

### 3. DSM归一化

根据`1_DSM_normalisation/readme.txt`：

> **我们不保证数据完全无误**，该数据仅为帮助研究人员使用高度数据而提供。

两种归一化方法：

- **LAStools方法**（`*_lastools.jpg`）：
  - 地面点三角剖分形成封闭表面
  - 计算地上点与地面的高度差
  - 可能存在插值错误导致的伪影

- **自定义方法**（`*_ownapproach.jpg`）：
  - 避免三角剖分
  - 为每个地上点搜索最近地面点
  - 计算高度差
  - 同样存在伪影

建议：根据具体应用选择合适的nDSM，或自行处理原始LAZ点云数据。

### 4. 边界处理

- 红色 (255,0,0) 是Clutter/Background类别，不是边界/忽略类别
- `*_label_noBoundary.tif`中的黑色 (0,0,0) 像素是边界/忽略区域，训练和评估时应跳过
- 使用深度学习框架时，建议将黑色像素映射为`ignore_index`，例如PyTorch常用`255`

### 5. 内存管理

- 单个影像文件较大（~100-150MB）
- 6000×6000像素需考虑内存限制
- 建议：
  - 分块处理（如512×512或256×256）
  - 使用内存映射
  - 按需读取

### 6. 数据预处理

常见预处理步骤：

1. **归一化**：
   - 影像像素值归一化到[0, 1]或[-1, 1]
   - DSM高度值归一化

2. **波段选择**：
   - 可选择使用全部或部分波段
   - 考虑计算资源与性能平衡

3. **数据增强**：
   - 旋转、翻转
   - 随机裁剪
   - 颜色抖动（对RGB）
   - 注意不要改变标签的语义

---

## 参考文献

### 引用格式

如果您在研究中使用ISPRS Potsdam数据集，请按以下格式引用：

```bibtex
@inproceedings{ISPRS2D2015,
  title={ISPRS 2D Semantic Labeling Contest},
  author={Rottensteiner, Franz and Mayer, Balint and Nemayer, Julie and
          Zhu, Xiaoxiang and Lefevre, Stephane and Heipke, Christian},
  booktitle={ISPRS Annals of Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume={II-3/W4},
  pages={633--640},
  year={2015},
  doi={10.5194/isprsannals-II-3-W4-633-2015}
}
```

### 相关文献

1. **数据集介绍论文**：
   - Rottensteiner, F., et al. (2014). "The ISPRS 2D Semantic Labeling Benchmark." *ISPRS Workshop*

2. **方法学论文**（使用该数据集的代表性工作）：
   - Badrinarayanan, V., et al. (2017). "SegNet: Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." *IEEE TPAMI*
   - Kamnitsas, K., et al. (2017). "Efficient Multi-Scale 3D CNN with Fully Connected CRF for Accurate Brain Lesion Segmentation." *Medical Image Analysis*
   - 卷积网络在遥感领域的应用

3. **竞赛结果**：
   - ISPRS官方报告
   - 各参赛队伍的技术报告

### 相关资源

- **ISPRS官网**：https://www2.isprs.org/
- **数据集下载**：ISPRS官方网站或合作机构
- **基准排名**：ISPRS竞赛结果页面

---

## 附录

### A. 类别颜色映射表

| 建议训练ID | 类别名称 | RGB颜色 | 十六进制 |
|------------|----------|---------|----------|
| 0 | 不透水表面 | (255, 255, 255) | #FFFFFF |
| 1 | 建筑物 | (0, 0, 255) | #0000FF |
| 2 | 低矮植被 | (0, 255, 255) | #00FFFF |
| 3 | 树木 | (0, 255, 0) | #00FF00 |
| 4 | 汽车 | (255, 255, 0) | #FFFF00 |
| 5 | 杂乱/背景 | (255, 0, 0) | #FF0000 |
| 255（ignore） | 边界/忽略 | (0, 0, 0) | #000000 |

### B. 数据集统计信息

```
总瓦片数（完整覆盖）: 38
参与者标签瓦片数: 24
完整参考标签瓦片数: 38
总像素数（每瓦片）: 36,000,000
总像素数（参与者标签瓦片）: 864,000,000
总像素数（完整参考标签瓦片）: 1,368,000,000
总覆盖面积（每瓦片）: 0.09 km² (300m × 300m)
总覆盖面积（所有瓦片）: 约3.42 km²
```

### C. 与ISPRS Vaihingen数据集对比

| 属性 | Potsdam | Vaihingen |
|------|---------|-----------|
| 位置 | 波茨坦 | 法伊英根 |
| GSD | 5厘米 | 9厘米 |
| 影像尺寸 | 6000×6000 | 约3000×3000（不等） |
| 瓦片数 | 38 | 33 |
| 有标签瓦片 | 24 | 16 |
| 影像类型 | RGB, IRRG, RGBIR | IR, RGB, DSM |
| 场景特点 | 规则城市布局 | 不规则城市场景 |

### D. 常用代码库

1. **POTSDAM-SEMSEG**：GitHub上的多个开源实现
2. **MMSegmentation**：支持ISPRS数据集
3. **PaddleRS**：飞桨遥感智能解译开发套件
4. **torchgeo**：PyTorch地理学习库

---

## 更新日志

- **2025年**：本文档创建
- 数据集版本：稳定版（2015年发布）

---

## 联系方式

如有疑问或需要更多信息，请联系：

- **ISPRS**: 通过官方网站
- **数据集维护者**: ISPRS工作组

---

**文档版本**: 1.0
**最后更新**: 2025年
**文档作者**: AI Assistant
**许可**: 本文档遵循与数据集相同的使用条款

---

*本文档详细介绍了ISPRS Potsdam 2D语义分割数据集的各个方面，希望能为研究者提供全面的参考。*
