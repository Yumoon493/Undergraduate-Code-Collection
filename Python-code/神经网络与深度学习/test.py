import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# 文件路径
file_path = r"D:\HuaweiMoveData\Users\24901\Desktop\pic.png"

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"文件不存在: {file_path}")
else:
    print(f"文件存在: {file_path}")

    # 读取图像
    image = cv2.imread(file_path)
    if image is None:
        print(f"无法读取文件: {file_path}")
    else:
        # 将图像从 BGR 转换为 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 定义填充大小
        top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)

        # 常数填充
        constant = cv2.copyMakeBorder(image, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=(0, 0, 255))
        plt.imshow(constant)
        plt.title('Constant Padding')
        plt.show()

        # 镜像填充
        reflect = cv2.copyMakeBorder(image, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
        plt.imshow(reflect)
        plt.title('Mirror Padding')
        plt.show()

        # 显示对比图
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(constant)
        plt.title('Constant Padding')

        plt.subplot(1, 2, 2)
        plt.imshow(reflect)
        plt.title('Mirror Padding')

        plt.show()