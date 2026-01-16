import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# No sensitive information, data preprocessing is carried out
# 1. Data Cleaning: Check data integrity (no missing values)
print("训练集形状:", train_images.shape, "标签数量:", len(train_labels))
print("测试集形状:", test_images.shape, "标签数量:", len(test_labels))
# Output: Training set shape: (60,000, 28, 28) Number of labels: 60,000;
# Test set shape: (10,000, 28, 28) Label quantity: 10,000

# 2. Data Standardization: Pixel Value Normalization (0-1)
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# 3. Flatten the image (for feature extraction)
train_flat = train_images.reshape((-1, 28*28))
test_flat = test_images.reshape((-1, 28*28))


# Image feature extraction, calculate the mean and variance of pixels in each category
class_stats = []
for cls in range(10):
    cls_images = train_flat[train_labels == cls]
    mean = np.mean(cls_images, axis=0).reshape(28, 28)
    std = np.std(cls_images, axis=0).reshape(28, 28)
    class_stats.append((mean, std))

# Save the statistical results of features
np.savez('fashionmnist_stats.npz', means=[s[0] for s in class_stats], stds=[s[1] for s in class_stats])

# Principal Component Analysis (PCA) Dimension reduction
# Standardized Data (PCA Requirements)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_flat)

# Extract the first two principal components
pca = PCA(n_components=2)
pca_features = pca.fit_transform(train_scaled)

# Principal component explains variance
explained_variance = pca.explained_variance_ratio_
print(f"主成分1解释方差: {explained_variance[0]:.3f}，主成分2解释方差: {explained_variance[1]:.3f}")
# Output: Principal Component 1 explains variance: 0.111, Principal component 2 explains variance: 0.049


# Data Visualization
plt.figure(figsize=(10, 5))
plt.hist(train_labels, bins=10, edgecolor='black', alpha=0.7, color='skyblue')
plt.xticks(range(10), class_names, rotation=45, ha='right')
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Training Set Class Distribution')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()

#25 Sample Image Display
plt.figure(figsize=(12, 12))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(train_images[i], cmap='binary')
    plt.title(class_names[train_labels[i]])
    plt.axis('off')
plt.savefig('sample_images.png')
plt.show()

# Category Average image
plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    mean_img = class_stats[i][0]
    plt.imshow(mean_img, cmap='binary')
    plt.title(f'{class_names[i]}\nMean Image')
    plt.axis('off')
plt.savefig('mean_images.png')
plt.show()


#PCA dimensionality reduction scatter plot
plt.figure(figsize=(10, 7))
for cls in range(10):
    indices = np.where(train_labels == cls)
    plt.scatter(pca_features[indices, 0], pca_features[indices, 1],
                alpha=0.6, label=class_names[cls])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization of FashionMNIST')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.savefig('pca_scatter.png')
plt.show()

