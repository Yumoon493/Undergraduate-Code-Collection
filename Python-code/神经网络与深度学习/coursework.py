import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSHAttention(nn.Module):
    """LSH-based Sparse Attention Module with debugging"""

    def __init__(self, dim, num_heads=8, bucket_size=64, dropout=0.1, debug=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.bucket_size = bucket_size
        self.head_dim = dim // num_heads
        self.debug = debug

        # Projection layers
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

        # LSH parameters
        self.lsh_proj = nn.Parameter(torch.randn(dim, bucket_size))
        self.dropout = nn.Dropout(dropout)

    def _lsh_bucketing(self, x):
        proj = torch.matmul(x, self.lsh_proj)
        proj = F.normalize(proj, dim=-1)
        hashes = torch.argmax(proj, dim=-1)
        if self.debug:
            print(f"LSH Bucketing - Unique hashes: {torch.unique(hashes).numel()}")
        return hashes

    def _sort_by_bucket(self, x, hashes):
        sorted_hashes, indices = torch.sort(hashes, dim=-1)
        sorted_x = torch.gather(
            x, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        )
        if self.debug:
            print(f"Sort by bucket - First 5 hashes: {sorted_hashes[0, :5]}")
        return sorted_x, sorted_hashes, indices

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape

        if self.debug:
            print(f"Input shapes: Q={query.shape}, K={key.shape}, V={value.shape}")

        # Linear projections
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Compute LSH buckets
        Q_hashes = self._lsh_bucketing(Q)
        K_hashes = self._lsh_bucketing(K)

        # Sort by buckets
        Q_sorted, Q_hashes_sorted, Q_indices = self._sort_by_bucket(Q, Q_hashes)
        K_sorted, K_hashes_sorted, K_indices = self._sort_by_bucket(K, K_hashes)
        V_sorted = torch.gather(
            V, dim=1, index=K_indices.unsqueeze(-1).expand(-1, -1, V.size(-1))
        )

        # Split into heads
        Q_heads = Q_sorted.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K_heads = K_sorted.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V_heads = V_sorted.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if self.debug:
            print(f"Attn scores shape: {attn_scores.shape}")
            print(f"Attn scores stats: min={attn_scores.min().item()}, max={attn_scores.max().item()}")

        # Create bucket-aware mask
        bucket_mask = (Q_hashes_sorted.unsqueeze(-1) == K_hashes_sorted.unsqueeze(-2))
        bucket_mask = bucket_mask.unsqueeze(1)

        # Apply masks
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_mask = torch.logical_and(mask, bucket_mask)
        else:
            attn_mask = bucket_mask

        attn_scores = attn_scores.masked_fill(~attn_mask, -1e9)

        # Softmax and attention output
        attn_probs = F.softmax(attn_scores, dim=-1)
        if self.debug:
            print(f"Attn probs sum: {attn_probs.sum(dim=-1).mean().item():.4f}")
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, V_heads)

        # Merge heads and restore original order
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.dim)
        attn_output = self.out(attn_output)

        if self.debug:
            print(f"Output shape: {attn_output.shape}")
            print(f"Output stats: mean={attn_output.mean().item()}, std={attn_output.std().item()}")

        return attn_output


# 测试代码
if __name__ == "__main__":
    # 初始化模块（开启调试模式）
    attn = LSHAttention(
        dim=64,
        num_heads=4,
        bucket_size=16,
        dropout=0.1,
        debug=True
    )

    # 创建测试输入（小维度便于查看）
    batch_size, seq_len, dim = 2, 32, 64
    query = torch.randn(batch_size, seq_len, dim)
    key = torch.randn(batch_size, seq_len, dim)
    value = torch.randn(batch_size, seq_len, dim)

    # 前向传播
    output = attn(query, key, value)

    # 打印输出摘要
    print("\n=== LSH Attention Test Complete ===")
    print(f"Final output shape: {output.shape}")
    print(f"First token output: {output[0, 0, :5].tolist()}")