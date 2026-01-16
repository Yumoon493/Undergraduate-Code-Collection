import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 定义标签转换函数
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[i] for i in labels]

# 定义图像显示函数
def show_images(images, labels, num_rows, num_cols, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(labels[i])
    plt.tight_layout()
    plt.show()

# 选择前 18 张图像和标签
images = train_images[:18]
labels = train_labels[:18]

# 显示图像
show_images(images, get_fashion_mnist_labels(labels), num_rows=2, num_cols=9)