import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow

plt.figure(figsize=(14, 8), dpi=300)
ax = plt.gca()

# 系统组件
components = [
    ("PDF Contract\nParser", 0.1, 0.5, 0.15, 0.1),
    ("Tokenization &\nPreprocessing", 0.25, 0.5, 0.18, 0.1),
    ("Optimized\nDeepSeek Encoder", 0.45, 0.5, 0.2, 0.15),
    ("Clause\nRecognition", 0.65, 0.7, 0.15, 0.1),
    ("Risk\nDetection", 0.65, 0.5, 0.15, 0.1),
    ("Summary\nGeneration", 0.65, 0.3, 0.15, 0.1),
    ("Legal Analysis\nDashboard", 0.85, 0.5, 0.18, 0.12)
]

# 绘制组件
colors = ['#E1BEE7', '#BBDEFB', '#C8E6C9', '#FFECB3', '#FFCCBC', '#D7CCC8', '#F5B7D1']
for i, (text, x, y, w, h) in enumerate(components):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=colors[i], edgecolor='#37474F', lw=1.5, alpha=0.9))
    plt.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, weight='bold')

# 绘制连接线
arrow_params = {'head_width': 0.02, 'head_length': 0.03, 'fc': '#5D4037', 'ec': '#5D4037'}
plt.arrow(0.1+0.15, 0.5, 0.1, 0, **arrow_params)
plt.arrow(0.25+0.18, 0.5, 0.15, 0, **arrow_params)
plt.arrow(0.45+0.2, 0.5, 0.15, 0.2, **arrow_params)
plt.arrow(0.45+0.2, 0.5, 0.15, 0, **arrow_params)
plt.arrow(0.45+0.2, 0.5, 0.15, -0.2, **arrow_params)
plt.arrow(0.65+0.15, 0.7, 0.15, -0.2, **arrow_params)
plt.arrow(0.65+0.15, 0.5, 0.15, 0, **arrow_params)
plt.arrow(0.65+0.15, 0.3, 0.15, 0.2, **arrow_params)

# 突出DeepSeek模块
ax.add_patch(Rectangle((0.44, 0.49), 0.22, 0.17, fill=None, edgecolor='#D32F2F', lw=3, linestyle='-'))
plt.text(0.45, 0.67, "Optimized DeepSeek Encoder\n(LSH Attention Enabled)",
         color='#D32F2F', fontsize=12, weight='bold')

# API调用示例（修改后使用\mathtt）
plt.text(0.45, 0.3,
         r'$\mathtt{legal\_analysis = DeepSeekLegalProcessor(}$' + '\n' +
         r'$\quad\mathtt{contract\_text, }$' + '\n' +
         r'$\quad\mathtt{enable\_lsh=True}$' + '\n' +
         r'$\mathtt{)}$',
         fontsize=11, bbox=dict(facecolor='#E0F7FA', alpha=0.7),
         ha='left')

plt.title("Fig. 3: Legal Document Processing System Architecture", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig('application.png', dpi=300, bbox_inches='tight')