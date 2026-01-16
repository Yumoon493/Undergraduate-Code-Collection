import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.figure(figsize=(12, 8), dpi=300)
ax = plt.gca()

# 模块定义
modules = [
    ("Input Sequence\n(10k tokens)", 0.1, 0.5, 0.18, 0.12),
    ("LSH Projection\n(b=64 buckets)", 0.3, 0.7, 0.2, 0.12),
    ("Bucket Sorting", 0.3, 0.3, 0.15, 0.1),
    ("Sparse Attention\nCalculation", 0.5, 0.5, 0.2, 0.15),
    ("Output Context\n(2.8x faster)", 0.7, 0.5, 0.18, 0.12)
]

# 绘制模块
colors = ['#E3F2FD', '#FFF8E1', '#E8F5E9', '#F3E5F5', '#FFEBEE']
for i, (text, x, y, w, h) in enumerate(modules):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=colors[i], edgecolor='#37474F', lw=1.5))
    plt.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10)

# 绘制箭头
def draw_curved_arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', mutation_scale=20,
        color='#455A64', lw=1.5
    ))

draw_curved_arrow(0.28, 0.5, 0.3, 0.7)  # Input to LSH
draw_curved_arrow(0.28, 0.5, 0.3, 0.3)  # Input to Sorting
draw_curved_arrow(0.3+0.2, 0.7, 0.5, 0.57)  # LSH to Attention
draw_curved_arrow(0.3+0.15, 0.3, 0.5, 0.43)  # Sorting to Attention
draw_curved_arrow(0.5+0.2, 0.5, 0.7, 0.5)  # Attention to Output

# 性能对比（简化的mathtext兼容版本）
plt.text(0.5, 0.2,
         r'$\bf{Performance\ Improvement}$' + '\n\n' +
         r'$\rm{Original\ \ }$' + r'$12.7\ \rm{s}$' + '\n' +
         r'$\rm{Optimized\ }$' + r'$4.5\ \rm{s}$' + '\n\n' +
         r'$O(n^2) \rightarrow O(n \log n)$',
         fontsize=14, bbox=dict(facecolor='#FFF59D', alpha=0.8),
         ha='center')

plt.title("Fig. 2: LSH Attention Optimization Workflow", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig('optimization.png', dpi=300, bbox_inches='tight')