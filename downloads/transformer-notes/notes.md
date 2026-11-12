---
title: Transformer 学习笔记
date: 2026-11-12
tags:
  - 深度学习
  - Transformer
  - PyTorch
  - ACT
categories:
  - 技术学习
typora-copy-images-to: assets
---

# 从 Attention 到 **Transformer**

> ### ***"Attention is all you need."***

## 一、Introduction：Transformer 解决什么问题？

一条序列中的信息通常相互关联。例如“桌上的苹果在夹爪左边”，判断相对位置需要同时处理苹果和夹爪的信息。机器翻译也需要结合前后文，才能确定一个词在句子中的含义。

Transformer 的核心做法是：**让每个位置根据输入，计算自己与其他位置的关联，再按关联权重汇总信息。**

经典论文为 Vaswani 等人在 2017 年发表的 *Attention Is All You Need*。论文提出基于注意力的序列转换网络，在机器翻译任务中验证了结构的效果，并讨论了并行计算的优势。[论文入口](https://arxiv.org/abs/1706.03762)

### 1. 原始 Transformer 的总体结构

原论文采用 Encoder–Decoder 结构。Encoder 将输入序列转成上下文相关的特征，Decoder 根据这些特征和已有输出预测下一个 token。

```text
源序列 → 向量表示 + 位置信息 → Encoder ─────────┐
                                             ↓
已有目标序列 → 向量表示 + 位置信息 → Decoder → 输出层
```

原版模型将编码器层和解码器层分别堆叠 6 次。一层的结构在后文逐步展开。

## 二、Token 与 Embedding：序列怎样变成张量？

### 1. Token：序列中的基本处理单位

Token 是序列中的一个基本单位，具体含义由数据组织方式决定。

```text
输入：[2, 8, 1, 4]
       ↑  ↑  ↑  ↑
       一共有 4 个 token
```

文本可以按字、词或词片段切分；图像可以按图像块组织；ACT 中 CNN 特征图上的一个空间位置也可以对应一个视觉 token。

Token 数量与特征维度分别描述两个轴：前者表示有多少个位置，后者表示每个位置用多少个数表示。

### 2. Embedding：可训练的向量查找表

```python
token_embedding = nn.Embedding(10, 8)
```

这行代码创建一张形状为 `(10, 8)` 的参数表：10 个可用编号，每个编号对应一个 8 维向量。输入编号 2，就取出第 2 号行；编号从 0 开始。

```text
编号       对应的向量
0      [ 0.2, -0.1,  0.5, ... ]
1      [ 0.7,  0.3, -0.2, ... ]
2      [-0.4,  0.8,  0.1, ... ]
…                   …
9      [ 0.1,  0.4, -0.6, ... ]
```

表中的数通常随机初始化，之后根据最终任务的损失更新。每一维的含义由训练形成，无须提前指定。

```python
import torch
from torch import nn

tokens = torch.tensor([[2, 8, 1, 4], [7, 3, 9, 2]])
token_embedding = nn.Embedding(10, 8)
features = token_embedding(tokens)

print(tokens.shape)    # torch.Size([2, 4])
print(features.shape)  # torch.Size([2, 4, 8])
```

![Token 编号经过 Embedding 变成特征向量](assets/01-token-embedding.png)

这里的三个轴分别是：

$$
\boxed{[B,N,D]=[\text{批量大小},\text{序列长度},\text{特征维度}]}
$$

`Embedding(10, 8)` 中的 10 表示编号种类数。序列长度由输入决定。

### 3. 数字编号与数值大小

在这个例子中，数字作为离散编号查表。Embedding 本身没有内置“8 比 2 大”的规则，比较关系需要网络通过任务训练学到。机器人关节角度等连续数值可以经过线性层投影成特征向量。

## 三、Positional Encoding：怎样表达位置？

同一个编号在同一次前向计算中查到的向量相同。但 `[2,8,1,4]` 和 `[4,8,1,2]` 的顺序不同，比较首尾数字时，结果也可能不同。

普通自注意力需要额外的位置线索。常用做法是将 token 向量与位置向量相加：

$$
\boxed{x_i=e_i+p_i}
$$

其中，$e_i$ 是第 $i$ 个 token 的向量，$p_i$ 是该位置的向量。

例如：

```text
数字向量：[-0.4, 0.8,  0.1]
位置向量：[ 0.1, 0.0, -0.2]
相加结果：[-0.3, 0.8, -0.1]
```

相加后特征维度保持不变。

### 1. 可学习的位置嵌入

为每个位置建立一行可训练参数，使用位置编号查询：

```python
position_embedding = nn.Embedding(4, 8)
positions = torch.arange(4)                 # [0, 1, 2, 3]
position_features = position_embedding(positions)  # [4, 8]
x = features + position_features           # [2, 4, 8]
```

这里用到广播：`[4,8]` 的位置向量被加到批量中的每条序列上。

### 2. 原论文的正弦、余弦位置编码

原论文采用以下固定公式生成位置编码，并实验比较了可学习位置嵌入：

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$pos$ 表示位置，$i$ 表示成对特征维度的编号。不同维度使用不同频率，让每个位置获得一组位置特征。本笔记配套代码使用可学习的位置嵌入。

## 四、Self-Attention：每个位置怎样汇总其他位置的信息？

### 1. 先理解权重的作用

处理位置 0 时，假设对四个位置算出的权重为：

$$
\alpha_0=[0.1,0.2,0.1,0.6]
$$

那么位置 0 的注意力输出是：

$$
z_0=0.1v_0+0.2v_1+0.1v_2+0.6v_3
$$

$v_j$ 是位置 $j$ 提供的 Value 向量。位置 3 的权重最大，因此它提供的信息在这次汇总中占比最大。

每个位置分别计算自己的权重。因此注意力权重是一张 $N\times N$ 的表：**行表示接收信息的位置，列表示提供信息的位置。**

```text
                来源0  来源1  来源2  来源3
更新位置0        0.1    0.2    0.1    0.6
更新位置1        0.3    0.4    0.2    0.1
更新位置2        0.2    0.2    0.3    0.3
更新位置3        0.5    0.1    0.1    0.3
```

每一行的权重和为 1。权重根据当前输入计算。

### 2. Q、K、V 分别是什么？

| 名称 | 全称 | 作用 |
| --- | --- | --- |
| Q | Query | 发起匹配，确定当前位置需要怎样的信息 |
| K | Key | 与 Query 匹配，用来计算分数 |
| V | Value | 在加权汇总时提供实际内容 |

**Q 和 K 决定各取多少，V 提供实际汇总的信息。** 自注意力中的 Q、K、V 都来自同一条序列。

### 3. Q、K、V 的数值从哪里来？

输入向量分别经过三个可训练的线性变换：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

公式先省略偏置。假设一个位置的输入为 $x=[1,2,3]$，并且：

$$
W_Q=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\qquad
W_K=\begin{bmatrix}0&1\\1&0\\1&-1\end{bmatrix}
$$

则：

$$
q=xW_Q=[1\times1+2\times0+3\times1,\;1\times0+2\times1+3\times1]=[4,5]
$$

$$
k=xW_K=[5,-2]
$$

同一个输入得到不同的 Q、K，是因为两组权重不同。训练时，损失通过反向传播更新 $W_Q,W_K,W_V$；Q、K、V 是这些参数根据当前输入算出的中间结果。

同一层、同一个头中，各位置共用同一组投影参数。这与 CNN 在不同位置复用同一个卷积核的参数共享思想相通。

```python
self.q_layer = nn.Linear(d_model, d_model)
self.k_layer = nn.Linear(d_model, d_model)
self.v_layer = nn.Linear(d_model, d_model)

q = self.q_layer(x)
k = self.k_layer(x)
v = self.v_layer(x)
```

PyTorch 的 `Linear` 按 `x @ weight.T + bias` 存储和计算权重；上述数学公式将投影矩阵写成方便右乘的形式。

### 4. 点积打分、缩放和 Softmax

以 $q_0=[1,0]$ 为例，四个 K 分别为：

$$
k_0=[1,0],\ k_1=[0,1],\ k_2=[1,1],\ k_3=[2,0]
$$

点积是对应元素相乘再相加，所以匹配分数为：

$$
[q_0k_0^T,q_0k_1^T,q_0k_2^T,q_0k_3^T]=[1,0,1,2]
$$

除以 $\sqrt{d_k}$，这里 $d_k=2$：

$$
s\approx[0.707,0,0.707,1.414]
$$

这个缩放控制点积的尺度，避免维度较大时分数幅度增大，使 Softmax 过于饱和。

再用 Softmax 将分数转成权重：

$$
\alpha_j=\frac{e^{s_j}}{\sum_l e^{s_l}}
$$

得到约 $[0.199,0.098,0.199,0.403]$。如果 V 为 $[[1,0],[0,2],[2,0],[0,4]]$，使用上述保留三位小数的权重加权求和，得到约 $[0.597,1.808]$。

![单个位置的自注意力计算过程](assets/02-self-attention.png)

### 5. 用矩阵一次处理所有位置

把各位置的 Q、K、V 按行放在矩阵中，得到完整公式：

$$
\boxed{\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V}
$$

```python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(head_dim)
weights = torch.softmax(scores, dim=-1)
context = weights @ v
```

对于一个头：

```text
Q       [N, dₖ]
Kᵀ      [dₖ, N]
QKᵀ     [N, N]     每个位置对所有位置的打分
weights [N, N]     每行分别做 Softmax
V       [N, dᵥ]
输出     [N, dᵥ]
```

`@` 表示矩阵乘法；`transpose(-2,-1)` 交换最后两个轴；`dim=-1` 表示沿最后一维，也就是信息来源位置，计算 Softmax。

## 五、Multi-Head Attention：为什么使用多个头？

一个头学习一组信息匹配和汇总方式。多个头使用不同投影参数，并行处理相同输入，再将结果组合起来：

$$
head_i=\operatorname{Attention}(XW_i^Q,XW_i^K,XW_i^V)
$$

$$
\operatorname{MHA}(X)=\operatorname{Concat}(head_1,\ldots,head_h)W_O
$$

例如一个头较多关注末尾位置，另一个头较多关注当前位置。每个头的分工通过训练形成。

![两个注意力头的并行计算和拼接](assets/03-multihead.png)

### 1. 多头的张量形状

代码采用 $D=8,h=2$，所以每个头的维度为 $d_k=4$：

```text
线性投影后的 Q：[B, N, 8]
        ↓ reshape
[B, N, 2, 4]
        ↓ transpose(1, 2)
[B, 2, N, 4]
```

交换轴后，每个批次、每个头可以独立进行矩阵乘法：

```text
注意力权重：[B, 2, N, N]
加权输出：  [B, 2, N, 4]
        ↓ transpose(1, 2)
[B, N, 2, 4]
        ↓ reshape，拼接头的特征
[B, N, 8]
        ↓ 输出线性层
[B, N, 8]
```

这里先通过线性层生成新的特征，再拆分到不同头；每个头对应不同的可训练投影。

## 六、Residual Connection 与 LayerNorm：怎样连接多层网络？

### 1. 残差连接：把输入加回模块输出

$$
Y=X+F(X)
$$

例如输入是 $[1,2,3]$，模块输出是 $[0.2,-0.5,0.8]$，相加得到 $[1.2,1.5,3.8]$。

输入通过一条直接路径参与输出，模块可以学习在原有表示上增加怎样的调整。这条路径也有助于梯度传播。直接相加没有可训练参数，两个张量的形状应保持一致。

### 2. LayerNorm：调整每个位置的特征分布

对于形状为 `[B,N,D]` 的特征，`nn.LayerNorm(D)` 对每个位置的 D 个特征分别计算均值和方差：

$$
\mu=\frac1D\sum_{i=1}^D x_i,\qquad
\sigma^2=\frac1D\sum_{i=1}^D(x_i-\mu)^2
$$

$$
\boxed{y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i}
$$

以 $[1,2,3]$ 为例，均值为 2，标准差约为 0.816。减均值再除以标准差，得到约 $[-1.225,0,1.225]$。

最后使用可学习的 $\gamma$ 缩放、$\beta$ 偏移。同一层中的各位置共用这些参数。$\epsilon$ 是用于数值稳定的小常数。

```python
norm = nn.LayerNorm(8)
y = norm(x)  # x 和 y 的形状均为 [B, N, 8]
```

经过可学习的缩放和偏移后，最终结果的均值、方差也会相应改变。

## 七、Feed-Forward Network：Transformer 中的 MLP

FFN 对每个位置的向量进行非线性加工：

$$
\operatorname{FFN}(x)=\operatorname{ReLU}(xW_1+b_1)W_2+b_2
$$

```python
self.ffn = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
)
```

```text
[B, N, 8] → Linear → [B, N, 16]
                     ↓ ReLU
[B, N, 8] ← Linear ← [B, N, 16]
```

这里先扩大维度，再映射回原维度。中间的非线性激活让两层线性变换能够表达更复杂的关系。

各位置分别经过同一个 FFN，共用参数。注意力负责跨位置交换信息，FFN 对交换后的特征进一步加工。CNN 中 `Flatten → Linear → ReLU → Linear` 的预测头，也包含同类 MLP 结构。

## 八、Encoder：将各个模块拼起来

按照原论文的 Post-LayerNorm 排列，一层编码器为：

$$
H=\operatorname{LayerNorm}(X+\operatorname{MHA}(X))
$$

$$
Y=\operatorname{LayerNorm}(H+\operatorname{FFN}(H))
$$

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model=8, num_heads=2, hidden_dim=16):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_output, weights = self.attention(x)
        x = self.norm1(x + attention_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x, weights
```

第二次残差加回的是 FFN 的输入，也就是第一次 LayerNorm 后的特征。多个编码器层连续连接，序列经过多轮信息交换和特征加工。

![Encoder 与 Decoder 单层结构对照](assets/04-encoder-decoder.png)

### 1. 输入输出为什么形状相同？

一个编码器层的输入输出都可以是 `[B,N,D]`。位置数量和特征维度相同，但向量中的数已经改变：输出特征结合了其他位置的信息。

### 2. 怎样用于分类？

编码器输出的是一串特征。可以取一个汇总位置，或沿序列轴平均，再接分类头：

```python
encoded, weights = encoder(x)  # [B, N, D]
summary = encoded.mean(dim=1)  # [B, D]，固定长度序列的平均汇总
logits = classifier(summary)  # [B, 类别数]
```

若输入存在补齐位置，注意力和平均汇总时都要屏蔽这些位置。

## 九、Decoder：结合已有输出和 Encoder 特征

Decoder 在自注意力和 FFN 之间增加了交叉注意力：

```text
目标序列特征
    ↓
带因果掩码的自注意力
    ↓
残差连接 + LayerNorm
    ↓
交叉注意力 ← Encoder 输出
    ↓
残差连接 + LayerNorm
    ↓
FFN
    ↓
残差连接 + LayerNorm
```

### 1. 因果掩码：预测时可以看到哪些位置？

原版翻译模型采用自回归生成，每次根据已经生成的内容预测下一个 token。训练时将目标序列错开一位：

```text
Decoder 输入：开始符  小猫  坐在  桌上
预测目标：     小猫   坐在  桌上  结束符
```

输入位置“坐在”需要预测“桌上”，所以不能访问它后面的“桌上”输入。因果掩码将这些禁止访问的注意力分数设为负无穷，Softmax 后对应权重变成 0。

![因果掩码及目标序列右移](assets/05-causal-mask.png)

```python
seq_len = scores.size(-1)
blocked = torch.triu(
    torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device),
    diagonal=1,
)
scores = scores.masked_fill(blocked, float('-inf'))
weights = torch.softmax(scores, dim=-1)
```

矩阵中主对角线上方的位置被屏蔽。训练时可以对整条右移后的序列并行计算，掩码负责限制各位置能够使用的信息；自回归推理时，则逐步追加新生成的 token。

### 2. 交叉注意力：读取 Encoder 的输出

交叉注意力沿用同样的注意力公式，但 Q 与 K、V 的来源不同：

```text
Decoder 当前特征 → 线性变换 → Q
Encoder 输出     → 线性变换 → K、V
```

假设目标长度为 M、源序列长度为 N，则一个头的权重矩阵形状为 `[M,N]`：每个目标位置决定从哪些源位置获取信息。

```python
q = q_layer(decoder_features)
k = k_layer(encoder_output)
v = v_layer(encoder_output)
```

翻译时，这一步让正在生成的内容与原句特征联系起来。Decoder 最后通过线性层输出词表中各 token 的分数，再转换成概率进行预测。

## 十、PyTorch 实现：完整的单层 Encoder

配套脚本使用两条长度为 4 的数字序列，特征维度为 8、注意力头数为 2、FFN 中间维度为 16。结构采用 Post-LayerNorm，代码展开了多头注意力的各步计算，便于观察张量形状。

[打开配套代码](code/encoder.py)

```python
"""
形状约定：B = 批量大小，N = 序列长度，D = 特征维度。
"""

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=8, num_heads=2):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("特征维度必须能被注意力头数整除")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 每层同时计算所有头的 Q/K/V；稍后拆成多个头。
        # nn.Linear 包含可训练的权重和偏置。
        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape  # [B, N, D]

        q = self.q_layer(x)  # [B, N, D]
        k = self.k_layer(x)
        v = self.v_layer(x)

        # 例如 D=8、头数=2：每个头处理 4 维。
        # [B, N, 8] → [B, N, 2, 4] → [B, 2, N, 4]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # @ 是矩阵乘法。transpose(-2, -1) 交换 K 的最后两个轴。
        # 每个头中：[N, head_dim] @ [head_dim, N] → [N, N]
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # 最后一维对应信息来源位置：每一行权重之和为 1。
        weights = torch.softmax(scores, dim=-1)  # [B, 头数, N, N]

        # 各位置按自己的权重汇总 V。
        context = weights @ v  # [B, 头数, N, head_dim]

        # 把同一个位置各个头的结果放在一起，再拼接。
        context = context.transpose(1, 2)  # [B, N, 头数, head_dim]
        context = context.reshape(batch_size, seq_len, d_model)
        output = self.output_layer(context)  # [B, N, D]
        return output, weights


class EncoderLayer(nn.Module):
    def __init__(self, d_model=8, num_heads=2, hidden_dim=16):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),  # 8 → 16
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),  # 16 → 8
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_output, weights = self.attention(x)
        x = self.norm1(x + attention_output)  # 第一次残差连接与归一化

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  # 第二次加回的是 FFN 的输入
        return x, weights


def main():
    torch.manual_seed(42)

    # 两条序列，每条有四个数字。数字在这里作为查表编号。
    tokens = torch.tensor([[2, 8, 1, 4], [7, 3, 9, 2]])  # [2, 4]
    batch_size, seq_len = tokens.shape
    d_model = 8

    token_embedding = nn.Embedding(10, d_model)  # 数字 0～9
    position_embedding = nn.Embedding(seq_len, d_model)
    encoder = EncoderLayer(d_model=d_model, num_heads=2, hidden_dim=16)

    positions = torch.arange(seq_len)  # [0, 1, 2, 3]
    token_features = token_embedding(tokens)  # [2, 4, 8]
    position_features = position_embedding(positions)  # [4, 8]
    # 广播：同一组位置向量加到批量中的每条序列上。
    x = token_features + position_features  # [2, 4, 8]

    output, weights = encoder(x)

    print("原始数字：\n", tokens)
    print("Encoder 输入形状：", tuple(x.shape))
    print("Encoder 输出形状：", tuple(output.shape))
    print("注意力权重形状：", tuple(weights.shape))
    print("第 1 条序列、第 1 个头的权重矩阵：\n", weights[0, 0].detach())
    print("每行权重之和：", weights[0, 0].sum(dim=-1).detach())

    # 这里只演示前向计算。输出是特征，分类任务再接分类头。
    # 训练时对最终任务计算损失，反向传播会更新上述网络参数。


if __name__ == "__main__":
    main()
```

### 1. 运行方法与输出

在安装了 PyTorch 的 Python 环境中，从笔记所在目录执行：

```bash
python code/encoder.py
```

固定随机种子为 42，本地运行得到：

```text
原始数字：
 tensor([[2, 8, 1, 4],
        [7, 3, 9, 2]])
Encoder 输入形状： (2, 4, 8)
Encoder 输出形状： (2, 4, 8)
注意力权重形状： (2, 2, 4, 4)
第 1 条序列、第 1 个头的权重矩阵：
 tensor([[0.2354, 0.0670, 0.2012, 0.4964],
        [0.0802, 0.0493, 0.2399, 0.6306],
        [0.5691, 0.1616, 0.1375, 0.1318],
        [0.1727, 0.2036, 0.2875, 0.3362]])
每行权重之和： tensor([1.0000, 1.0000, 1.0000, 1.0000])
```

输出中的 `(2,2,4,4)` 分别表示：2 条序列、2 个头、4 个接收位置、4 个来源位置。每行权重和为 1，与 Softmax 的计算对应。网络参数处于随机初始化状态，这组输出用于观察前向计算和张量形状。

### 2. 训练时哪些数会变化？

可训练参数包括数字嵌入表、位置嵌入表、Q/K/V 投影、注意力输出投影、FFN 的线性层，以及 LayerNorm 的缩放和偏移参数。Q、K、V 和注意力权重是每次根据输入计算的中间结果。

训练一个完整任务时，将这些模块注册在同一个 `nn.Module` 中，为输出接上任务头，再将 `model.parameters()` 交给优化器：

```python
logits = model(tokens)
loss = loss_fn(logits, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

最终任务的损失沿整条计算路径反向传播，改变特征表示、注意力匹配和输出预测。
