import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrow

plt.figure(figsize=(10, 8), dpi=300)
ax = plt.gca()

# 绘制Transformer块
blocks = [
    ("Input Embedding", 0.1, 0.8, 0.15, 0.08),
    ("Positional Encoding", 0.1, 0.65, 0.15, 0.08),
    ("Multi-Head Attention", 0.3, 0.75, 0.18, 0.1),
    ("Add & Norm", 0.5, 0.75, 0.12, 0.08),
    ("Feed Forward", 0.3, 0.6, 0.15, 0.08),
    ("Add & Norm", 0.5, 0.6, 0.12, 0.08),
    ("Output", 0.7, 0.7, 0.12, 0.08)
]

for text, x, y, w, h in blocks:
    ax.add_patch(Rectangle((x, y), w, h, fill=None, edgecolor='blue', lw=1.5))
    plt.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9)

# 绘制注意力公式
plt.text(0.35, 0.9, r'$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$',
         fontsize=14, bbox=dict(facecolor='#FFF59D', alpha=0.8))

# 绘制复杂度标注
ax.add_patch(Rectangle((0.28, 0.72), 0.22, 0.13, fill=None, edgecolor='red', linestyle='--', lw=2))
plt.text(0.39, 0.85, r'$O(n^2)$ Complexity Bottleneck', color='red',
         fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))

# 绘制连接线
arrow_params = {'head_width': 0.015, 'head_length': 0.02, 'fc': 'k', 'ec': 'k'}
plt.arrow(0.25, 0.8, 0.05, -0.05, **arrow_params)
plt.arrow(0.25, 0.65, 0.05, 0.1, **arrow_params)
plt.arrow(0.48, 0.75, 0.02, 0, **arrow_params)
plt.arrow(0.48, 0.6, 0.02, 0, **arrow_params)
plt.arrow(0.3, 0.64, 0, -0.04, **arrow_params)
plt.arrow(0.62, 0.7, 0.08, 0, **arrow_params)

plt.title("Fig. 1: DeepSeek Attention Architecture with Computational Bottleneck", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig('core_principle.png', dpi=300, bbox_inches='tight')