"""单层 Transformer Encoder：多头自注意力 → Add & Norm → FFN → Add & Norm。

运行：python encoder.py
形状约定：B = 批量大小，N = 序列长度，D = 特征维度。
采用原论文的 Post-LayerNorm 排列，使用两头自注意力。
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
