---
title: 深度学习学习记录：从图像张量到 CNN 与 PyTorch
date: 2026-11-11
tags:
  - 深度学习
  - CNN
  - PyTorch
  - CIFAR-10
categories:
  - 技术学习
typora-copy-images-to: assets
---

# 深度学习学习记录：从图像张量到 CNN 与 PyTorch

ACT 使用相机图像和机器人状态预测动作。其中一个基础问题是：摄像头得到的图片，怎样变成神经网络能够处理的数字？网络又怎样从这些数字中提取信息，并通过训练得到有用的输出？

沿着一张“桌面上的苹果”图片，可以把过程分成几步：

```text
图片 → 数字张量 → 提取特征 → 预测结果
                              ↓
                     与正确答案比较，计算损失
                              ↓
                     计算梯度，调整网络参数
```

先用苹果中心位置理解连续数值预测，再使用 CIFAR-10 完成图片分类。两种任务共享训练的基本逻辑，输出含义和损失函数则分别设计。

## 一、Tensor：图片怎样变成数字？

### 1. 从一个像素开始

将图片放大，可以看到一个个小格子，每个小格子就是一个像素。对于常见的 8 位灰度图片，一个像素用一个亮度数值表示：

- `0`：黑色。
- `255`：白色。
- 中间数值：不同程度的灰色。

例如，下面的数字就是一张高 4 像素、宽 4 像素的图片：

```text
  0    0    0    0
  0  255  255    0
  0  255  255    0
  0    0    0    0
```

中间四个像素为白色，外围为黑色。图片上的图案与数字数组一一对应。

在 PyTorch 中，可以这样创建：

```python
import torch

image = torch.tensor([
    [0,   0,   0, 0],
    [0, 255, 255, 0],
    [0, 255, 255, 0],
    [0,   0,   0, 0],
], dtype=torch.float32)
```

`torch.tensor()` 根据数据创建张量；`float32` 表示使用 32 位浮点数存储每个元素。后续乘权重、求和、计算梯度都涉及小数，因此网络通常使用浮点数据。

### 2. 张量、轴和形状分别是什么？

在这里，Tensor（张量）可以理解为按照一定形状组织的数值数组。

| 数据组织方式 | 示例 | shape | 轴数 |
| --- | --- | --- | --- |
| 一个数，标量 | `7` | `()` | 0 |
| 一排数，向量 | `[2, 5, 8]` | `(3,)` | 1 |
| 一个二维表，矩阵 | 4 行 4 列的灰度图 | `(4, 4)` | 2 |
| 三张二维表 | RGB 三通道图片 | `(3, 4, 4)` | 3 |
| 两张 RGB 图片 | 一批图片 | `(2, 3, 4, 4)` | 4 |

**轴表示组织数据的方向，shape 表示每个轴的长度。**

例如 `(3, 4)` 有两个轴，因此是二维张量；第一个轴长度为 3，第二个轴长度为 4。四维张量则可以用“图片编号、通道、行、列”四个轴组织数据。

```python
print(image.shape)    # torch.Size([4, 4])
print(image.ndim)     # 2：轴数
print(image.numel())  # 16：元素数量
```

`(16,)` 与 `(4, 4)` 都有 16 个数，但排列方式不同。对于图像，排列方式还表达了像素之间的空间邻接关系。

### 3. 一张彩色图片为什么有三个通道？

RGB 图片的每个像素由红、绿、蓝三个强度值组成：

```text
红色：(255,   0,   0)
绿色：(  0, 255,   0)
黄色：(255, 255,   0)
白色：(255, 255, 255)
```

因此，一张 RGB 图片可以理解为三张对齐的二维表：R 通道记录红色强度，G 通道记录绿色强度，B 通道记录蓝色强度。

在本例的 PyTorch CNN 中，批量输入使用 NCHW 排列：

$$
X\in\mathbb{R}^{N\times C\times H\times W}
$$

其中：

- $N$：Batch Size，批量大小，即一次处理多少张图片。
- $C$：Channel，通道数；RGB 图片为 3。
- $H$：Height，高度，即行数。
- $W$：Width，宽度，即列数。

![完整展开的图片张量与四个轴](assets/tensor-2-3-4-4.png)

图中用 0～95 演示全部元素的位置；格子底色用于区分通道。

### 4. 形状描述有多大，索引指定取哪里

对于 `batch.shape = (2, 3, 4, 4)`：

```python
batch[0]           # 第一张图片，shape = (3, 4, 4)
batch[0, 2]        # 第一张图片的 B 通道，shape = (4, 4)
batch[0, 2, 1]     # 该通道中行索引为 1 的一行，shape = (4,)
batch[0, 2, 1, 3]  # 该行列索引为 3 的一个数，shape = ()
```

可以形象地理解为：先用前两个索引选中某张图片的某个通道矩阵，再用后两个索引选中矩阵内部的位置。索引从 0 开始。

单个 RGB 像素需要从三个通道的相同位置各取一个数：

```python
pixel = batch[0, :, 1, 2]  # 第一张图片，行 1、列 2 的 RGB
```

其中 `:` 表示保留该轴的全部元素。

## 二、Convolution：为什么要使用卷积层？

### 1. 从对应相乘、再求和开始

先取一个局部图片区域和一组权重：

```text
图片窗口              权重
1  2                  1   0
4  5                  0  -1
```

对应位置相乘，再求和：

$$
1\times1+2\times0+4\times0+5\times(-1)=-4
$$

在 PyTorch 中：

```python
patch = torch.tensor([[1., 2.], [4., 5.]])
weight = torch.tensor([[1., 0.], [0., -1.]])

result = (patch * weight).sum()  # tensor(-4.)
```

这里的 `*` 表示逐元素相乘。矩阵乘法使用 `@`，两者运算规则不同。

### 2. 卷积核就是不断移动的小型特征提取器

将同一组权重在图片上逐格移动，每个位置计算一个结果，再按照空间位置组成输出，这就是本例的卷积计算过程。

![卷积核逐格移动与加权求和](assets/convolution-sliding-window.gif)

对于输入：

```text
1  2  3
4  5  6
7  8  9
```

使用上述 `2×2` 卷积核，每次移动一格，得到：

```text
-4  -4
-4  -4
```

每个输出数都来自不同的局部窗口，四个位置共用同一组权重，这叫 **Weight Sharing（权重共享）**。PyTorch 的 `Conv2d` 按不翻转卷积核的互相关方式计算，深度学习中通常称作卷积。

### 3. 卷积究竟提取什么信息？

考虑一排像素：

```text
输入：0  0  1  1
权重：  -1  1
```

每次取相邻两个像素，计算“右边减左边”，输出为：

```text
0  1  0
```

原始输入表示亮度，输出则反映局部亮度变化。中间的强响应提示这里存在明暗分界。

同理，不同卷积核可以对不同局部图案产生响应。一个卷积核对应一个输出通道；多个卷积核形成多张 Feature Map（特征图）。

```text
原始图片
   ↓
第一组卷积：局部颜色、明暗和纹理关系
   ↓
后续卷积与激活：组合前面的特征，使用更大范围的信息
   ↓
用于类别判断或位置预测的特征表示
```

颜色、边缘、轮廓是便于理解的例子。实际训练通过任务损失调整权重，多个通道共同表达有用信息。

### 4. RGB 输入怎样产生 16 个特征通道？

```python
nn.Conv2d(3, 16, kernel_size=3, padding=1)
```

表示：输入通道为 3，输出通道为 16，卷积核大小为 `3×3`。

每个输出通道都汇总三个输入通道的局部加权结果，再加一个偏置。因此，这层权重形状为：

```text
(16, 3, 3, 3)
 ↑   ↑  ↑  ↑
输出 输入 核高 核宽
```

参数数量为：

$$
16\times3\times3\times3+16=448
$$

前三个输入通道是颜色通道；输出的 16 个通道是学习得到的特征通道。

### 5. 步长和补边怎样改变尺寸？

- `stride`：卷积核每次移动多少格，默认 1。
- `padding`：图片边界外围补多少圈数值，默认补零。

对膨胀系数为 1 的卷积，高度计算为：

$$
H_{out}=\left\lfloor\frac{H_{in}+2P-K}{S}\right\rfloor+1
$$

宽度同理。$P$ 为补边，$K$ 为核大小，$S$ 为步长。

例如 `32×32` 输入、`3×3` 卷积核、补边 1、步长 1：

$$
H_{out}=\frac{32+2-3}{1}+1=32
$$

因此这一层只改变通道数量，保持高、宽。

## 三、ReLU：为什么卷积之后还需要激活函数？

### 1. 负数归零，非负数保留

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

```text
输入：                输出：
-2   3                0   3
 1  -4                1   0
```

```python
x = torch.tensor([[-2., 3.], [1., -4.]])
y = torch.relu(x)
```

ReLU 逐元素处理，输入和输出形状相同。

### 2. 为什么需要非线性？

如果连续两层只有加权和与偏置：

$$
a=2x+1,\qquad y=3a+4
$$

代入后：

$$
y=6x+7
$$

两层仍可以合并为一个仿射关系。加入 ReLU：

$$
y=3\max(0,2x+1)+4
$$

关系变成分段形式：

$$
y=\begin{cases}
4,&x<-0.5\\
6x+7,&x\geq-0.5
\end{cases}
$$

权重运算与 ReLU 交替组合，使网络能够表达复杂的非线性关系。

```text
卷积或全连接 → ReLU → 卷积或全连接 → ReLU
```

单独连续叠加 ReLU，第一次处理后结果已经非负，后面的 ReLU 保持这些数值。

## 四、Pooling：池化为什么能减少计算？

### 1. 每个局部区域保留最大响应

使用 `2×2` 最大池化窗口，步长为 2：

```text
输入：
1  3 │ 2  0
4  2 │ 1  5
─────┼─────
0  1 │ 7  2
3  2 │ 4  6

输出：
4  5
3  7
```

```python
nn.MaxPool2d(kernel_size=2, stride=2)
# 简写：nn.MaxPool2d(2)
```

最大池化没有可训练权重。每个窗口保留局部最强响应，缩小空间尺寸。

### 2. 缩小的是高和宽

```text
输入：(2, 8, 12, 12)
输出：(2, 8,  6,  6)
```

图片数量和通道数量保持不变，每张特征图分别池化。

对于一张 `1024×1024` 特征图，池化后变成 `512×512`，元素数量变为四分之一。最大池化选取局部最大值，会丢失空间细节；它与通过像素平均进行缩图的计算不同。

强响应在同一窗口内小幅移动，最大值可能保持不变。因此池化可以降低对某些局部位移的敏感程度，同时牺牲精确位置信息。

## 五、Flatten 与 Linear：特征怎样变成预测？

### 1. 展平只改变形状

```text
通道 0：       通道 1：
1  2           5  6
3  4           7  8

展平：
[1, 2, 3, 4, 5, 6, 7, 8]
```

![多个通道逐行展平](assets/flatten-channels.gif)

单张样本从 `(2, 2, 2)` 变成 `(8,)`。批量输入则保留第一个轴：

```text
(4, 2, 2, 2) → (4, 8)
```

```python
nn.Flatten(start_dim=1)
```

顺序和元素数量保留，后续全连接层可以根据特征所在的位置使用不同权重。

### 2. 全连接层综合所有输入特征

假设输入为 `[1, 2, 3]`，输出两个数：

$$
x=1\times0.1+2\times0.2+3\times0.3+0.1=1.5
$$

$$
y=1\times(-0.2)+2\times0.1+3\times0.4-0.1=1.1
$$

每个输出有自己的一套权重和偏置，并连接所有输入特征：

```python
nn.Linear(3, 2)
```

该层包含 $3\times2+2=8$ 个参数。一般形式为：

$$
\mathbf y=W\mathbf x+\mathbf b
$$

输出含义由任务和标签定义。两个输出可以用于中心坐标，十个输出可以用于十类图片的类别分数。

### 3. 一张图片经过 CNN 的完整过程

![苹果图片经过卷积、激活、池化、展平和全连接](assets/tinycnn-apple-flow.gif)

动画使用随机初始化参数实际计算，演示各层数据流及数值变化；图中灰度按通道数值范围缩放显示。

```text
输入                (B,  3, 32, 32)
卷积 → ReLU         (B, 16, 32, 32)
池化                (B, 16, 16, 16)
卷积 → ReLU         (B, 32, 16, 16)
池化                (B, 32,  8,  8)
展平                (B, 2048)
全连接 → ReLU       (B, 64)
位置输出            (B, 2)
```

分类时将最后一层改为 `nn.Linear(64, 10)`，得到 `(B, 10)` 的原始类别分数。

## 六、监督学习：程序究竟怎样知道什么是苹果？

### 1. 规则来自手写，还是来自样本？

一种方法是直接写“找红色区域，再检查圆形轮廓”。这类规则对红球、西红柿和青苹果的处理需要额外设计。

CNN 使用带标签样本训练。例如：

| 输入图片 | 标签 |
| --- | --- |
| 红苹果 | 有苹果 |
| 青苹果 | 有苹果 |
| 红球 | 无苹果 |
| 空桌子 | 无苹果 |

网络结构提供计算方式，标签定义正确答案，训练根据预测误差调整参数。

**图片输入网络，标签交给损失函数。** 正确答案通过参数更新影响模型。

### 2. 识别类别与预测位置是两种输出任务

| 任务 | 标签内容 | 输出示例 |
| --- | --- | --- |
| 图片分类 | 类别编号 | 猫、狗等类别的分数 |
| 单目标中心回归 | 中心坐标 | $(x,y)$ |
| 判断存在并定位 | 存在标签和坐标 | 存在分数与 $(x,y)$ |

若每张图片都有一个苹果，可以训练中心位置回归。若同时存在无苹果图片，则可增加存在判断分支，并只对有苹果样本计算位置损失。

### 3. 训练数据与验证数据分别做什么？

```text
训练集：计算损失、更新参数
验证集：检查新样本上的效果、选择模型
测试集：最终评价选定模型
```

CIFAR-10 已经提供类别标签和训练、测试划分，因此可以直接读取图片与对应类别。

## 七、Loss：预测错误怎样变成一个数？

### 1. 用苹果位置理解均方误差

设真实中心为 $(0.70,0.40)$，预测为 $(0.20,0.80)$：

$$
\Delta x=-0.50,\qquad\Delta y=0.40
$$

均方误差为：

$$
\boxed{\mathcal L_{MSE}=\frac{(\hat x-x)^2+(\hat y-y)^2}{2}=0.205}
$$

![预测位置与均方误差的关系](assets/apple-position-loss.gif)

动画通过设定不同预测位置展示损失变化。平方避免正负误差抵消，平均则综合两个坐标。

```python
prediction = torch.tensor([[0.20, 0.80]])
target = torch.tensor([[0.70, 0.40]])
loss = nn.MSELoss()(prediction, target)
```

### 2. 图片分类使用交叉熵

CIFAR-10 有十类，每张图片输出十个原始分数，称为 logits。交叉熵根据正确类别对应的相对分数计算损失。

设第 $j$ 类分数为 $z_j$，正确类别为 $y$：

$$
p_j=\frac{e^{z_j}}{\sum_{k=0}^{9}e^{z_k}},\qquad
\mathcal L_{CE}=-\log p_y
$$

正确类别的概率越高，损失越小。整批图片默认取平均。

```python
def calculate_loss(logits, labels):
    classification_loss = nn.CrossEntropyLoss()
    return classification_loss(logits, labels)
```

接口约定：

```text
logits：(B, 10)，浮点张量
labels：(B,)，int64 类别编号 0～9
loss：()，标量张量
```

`CrossEntropyLoss` 内部以数值稳定方式组合相关运算，直接接收原始分数。先用 `argmax` 选类别会丢失训练需要的分数信息；`.item()` 则把张量转为普通数值，适用于日志输出。

## 八、Gradient：损失怎样改变网络权重？

### 1. 从一个权重开始

令输入 $x=2$，标签 $y=6$，模型为：

$$
\hat y=wx,\qquad\mathcal L=(\hat y-y)^2
$$

当 $w=1$ 时，预测为 2，损失为 16。

| 权重 $w$ | 预测 $2w$ | 平方误差 |
| --- | --- | --- |
| 1 | 2 | 16 |
| 2 | 4 | 4 |
| 3 | 6 | 0 |
| 4 | 8 | 4 |

在 $w=1$ 附近，增大权重可以减小损失。梯度描述损失对参数变化的局部敏感程度。

### 2. 反向传播使用链式法则

$$
\frac{\partial\mathcal L}{\partial\hat y}=2(\hat y-y)=-8
$$

$$
\frac{\partial\hat y}{\partial w}=x=2
$$

所以：

$$
\boxed{\frac{\partial\mathcal L}{\partial w}=-8\times2=-16}
$$

前向计算由权重得到预测、再得到损失；反向传播从损失出发，沿依赖关系计算各参数梯度。存在多条路径时，将相应贡献相加。

### 3. 学习率决定参数更新步幅

基础梯度下降更新：

$$
w_{new}=w_{old}-\eta\frac{\partial\mathcal L}{\partial w}
$$

令学习率 $\eta=0.1$：

$$
w_{new}=1-0.1\times(-16)=2.6
$$

新预测为 5.2，损失变为 0.64。重复更新，权重逐渐接近 3。

| 更新前轮次 | 权重 | 预测 | 损失 | 梯度 |
| --- | --- | --- | --- | --- |
| 1 | 1.0000 | 2.0000 | 16.000000 | -16.0000 |
| 2 | 2.6000 | 5.2000 | 0.640000 | -3.2000 |
| 3 | 2.9200 | 5.8400 | 0.025600 | -0.6400 |
| 4 | 2.9840 | 5.9680 | 0.001024 | -0.1280 |

梯度为负时，减去梯度使参数增加；梯度为正时，参数减小。学习率过大可能越过合适位置，使损失增大。

### 4. PyTorch 怎样执行这三步？

```python
def update_parameters(loss, optimizer):
    optimizer.zero_grad()  # 清除先前保存的梯度
    loss.backward()       # 计算并保存各参数的梯度
    optimizer.step()      # 根据梯度更新参数
```

`backward()` 负责求梯度，`step()` 负责修改参数。PyTorch 默认累加梯度，因此常规单批更新前需要清除旧梯度。

训练一批图片的逻辑：

```python
optimizer.zero_grad()
logits = model(images)
loss = calculate_loss(logits, labels)
loss.backward()
optimizer.step()
```

一批图片称为 Batch；完整遍历一次训练集称为一个 Epoch。训练结束后保存权重，预测新图片时使用学到的权重。

## 九、PyTorch：算法怎样变成程序？

### 1. PyTorch 和 NumPy 有什么关系？

两者都是 Python 数值计算库。NumPy 的主要数组对象是 `ndarray`，PyTorch 的主要数组对象是 `Tensor`。

| 功能 | NumPy | PyTorch |
| --- | --- | --- |
| 多维数组与矩阵运算 | 支持 | 支持 |
| GPU 计算 | NumPy 本身主要使用 CPU | 支持 CPU、GPU |
| 自动求导 | 本身不提供 | `autograd` |
| 神经网络组件 | 本身不提供 | `torch.nn` |
| 参数优化 | 本身不提供神经网络优化器 | `torch.optim` |

```python
import torch
from torch import nn
```

`nn` 是 PyTorch 提供网络组件的模块，`nn.Module`、`nn.Conv2d` 和 `nn.ReLU` 都是其中的类。

### 2. 为什么继承 nn.Module？

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
```

`nn.Module` 提供网络参数与子模块管理能力。`CNN` 继承这些能力，再定义自己的层和前向计算。

| 接口 | 作用 |
| --- | --- |
| `parameters()` | 遍历参数，交给优化器 |
| `.to(device)` | 转移参数和缓冲区到指定设备 |
| `train()` / `eval()` | 设置训练或评估模式 |
| `state_dict()` | 取得参数及持久缓冲区状态，供保存与恢复 |

`super().__init__()` 执行父类初始化，建立模块管理所需的内部结构。

### 3. 层为什么写成 self.features？

```python
self.features = nn.Sequential(...)
```

`self` 指当前网络对象。将网络层保存为对象属性，后续 `forward()` 就能使用同一组层和参数。

同时，`nn.Module` 会自动登记赋给属性的子模块，使 `parameters()`、`.to()` 和 `state_dict()` 能递归找到它们。

```python
# 初始化：保存子模块
self.features = nn.Sequential(...)

# 前向计算：调用子模块，得到本次输入的特征
features = self.features(images)
```

`self.features` 保存网络层；局部变量 `features` 保存本次运算结果。属性名可以改为 `self.extractor` 等，调用处保持一致。

### 4. 为什么 model(images) 会进入 forward？

```python
def forward(self, images):
    features = self.features(images)
    return self.head(features)
```

`model(images)` 会经过 `nn.Module` 的调用机制，执行 `forward()` 及相关钩子。`__init__()` 负责定义有哪些层，`forward()` 负责定义输入如何经过这些层。

### 5. eval 与 no_grad 分别控制什么？

```python
model.eval()
with torch.no_grad():
    logits = model(images)
```

`eval()` 设置模块模式，影响 Dropout、BatchNorm 等依赖模式的层。`no_grad()` 关闭该区域的梯度记录，减少评估时的开销。

本例的卷积、ReLU、池化和全连接在两种模式下计算方式相同，评估时仍显式设置模式，便于后续扩展。

## 十、CIFAR-10：把训练流程接起来

### 1. 图片和标签怎样对应？

CIFAR-10 包含 10 类 `32×32` RGB 图片：50,000 张训练图，10,000 张测试图。类别编号依次为：

```text
0 飞机   1 汽车   2 鸟     3 猫     4 鹿
5 狗     6 青蛙   7 马     8 船     9 卡车
```

![CIFAR-10 各类别的真实训练图片](assets/cifar10-samples.png)

```python
image, label = train_data[0]
```

图片是输入张量，标签是另一个配对保存的类别编号。读取一批样本：

```text
images.shape = (64, 3, 32, 32)
labels.shape = (64,)
```

### 2. 网络定义

```python
import torch
from torch import nn

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, images):
        features = self.features(images)
        return self.head(features)
```

最后输出原始类别分数。预测类别时取最高分所在的编号：

```python
predicted_classes = logits.argmax(dim=1)
```

### 3. 三个核心部分怎样配合？

```python
model = TinyCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
```

该代码块对应核心训练循环；数据加载器、设备和轮数由外层配置。完整练习文件见 [train_cifar10.py](code/train_cifar10.py)。

### 4. 外围函数负责什么？

| 函数 | 职责 |
| --- | --- |
| `build_model()` | 返回网络对象 |
| `calculate_loss()` | 根据分数和标签返回标量损失 |
| `update_parameters()` | 清梯度、反向传播、更新参数 |
| `check_output()` | 检查分数形状、损失形状及数值有效性 |
| `check_interfaces()` | 使用随机输入检查前向与损失接口 |
| `make_loaders()` | 读取数据，划分训练、验证集并组成批次 |
| `run_epoch()` | 遍历一次数据，汇总损失和准确率 |
| `main()` | 管理配置、设备、训练轮次、模型保存和测试 |

练习版从原训练集中划分 45,000 张用于训练，5,000 张用于验证。每轮验证后保存准确率最高的模型，最后用正式测试集评价该模型。

损失按各批样本数加权求平均，准确率按正确预测数除以总样本数计算。验证阶段关闭梯度记录并跳过参数更新。

### 5. 运行步骤与环境

本机环境：

```text
Conda 环境：lerobot
PyTorch：2.7.1+cu128
Torchvision：0.22.1+cu128
GPU：NVIDIA GeForce RTX 5070 Laptop GPU
```

在笔记目录下执行：

```bash
conda activate lerobot
python code/train_cifar10.py --check
python code/train_cifar10.py --epochs 5
```

笔记配套脚本读取当前工作区 `work/datasets` 中的 CIFAR-10 缓存。迁移目录时，修改脚本顶部的 `DATA` 指向实际数据目录。

接口检查的实际输出：

```text
接口通过：图片 (4, 3, 32, 32) → 分数 (4, 10)
损失形状 ()，数值 2.2737
```

损失数值对应当次随机初始化和随机输入；接口检查关注输出尺寸及损失类型。单批更新检查确认了梯度计算和参数修改正常。

## 十一、与官方 CIFAR-10 例程对比：配置会带来什么影响？

官方例程见 [Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)，配套保存的源码见 [official_cifar10_tutorial.py](code/official_cifar10_tutorial.py)。

### 1. 网络结构

官方卷积部分：

```python
self.conv1 = nn.Conv2d(3, 6, 5)
self.pool = nn.MaxPool2d(2, 2)
self.conv2 = nn.Conv2d(6, 16, 5)
```

```text
官方：32 → 卷积 28 → 池化 14 → 卷积 10 → 池化 5
      展平特征：16 × 5 × 5 = 400

练习：32 → 卷积 32 → 池化 16 → 卷积 16 → 池化 8
      展平特征：32 × 8 × 8 = 2048
```

官方全连接为 `400→120→84→10`，练习版为 `2048→64→10`。根据两个实际网络对象统计：

| 网络 | 可训练参数数量 |
| --- | ---: |
| 官方例程 | 62,006 |
| 练习版 | 136,874 |

更多通道和空间位置使练习版保留更多特征，同时增加参数和计算开销。识别表现需要结合训练配置与测试结果判断。

### 2. 图片归一化

官方将浮点像素从 `0～1` 转换为 `−1～1`：

```python
Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
```

$$
x'=\frac{x-0.5}{0.5}
$$

练习版使用 `ToTensor()` 保留 `0～1`。数值分布会影响中间激活和梯度，训练与预测需要使用相同预处理。

### 3. 优化器、批量与轮数

| 配置 | 官方例程 | 练习版默认值 |
| --- | --- | --- |
| 损失函数 | CrossEntropyLoss | CrossEntropyLoss |
| 优化器 | SGD，动量 0.9 | Adam |
| 学习率 | 0.001 | 0.001 |
| 批量大小 | 4 | 64 |
| 训练轮数 | 2 | 5 |
| 模型选择 | 保存训练末尾模型 | 保存验证准确率最佳模型 |

带动量的 SGD 综合当前梯度和历史方向；Adam 使用梯度的一阶、二阶矩估计调整更新。相同学习率在两种优化器下会产生不同更新。

批量大小影响单次更新综合的样本数量，以及每轮更新次数。45,000 张训练图、批量 64 时，每轮约 704 次更新。

官方采用 `F.relu()`，练习采用 `nn.ReLU()`，对应相同的 ReLU 运算。拆分函数与使用 `Sequential` 主要改变组织方式。



