import cv2
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread(r"D:\HuaweiMoveData\Users\24901\Desktop\pic.png")#不可以出现中文
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像从BGR转换为RGB

top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
constant = cv2.copyMakeBorder(image, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=(0, 0, 255))
plt.imshow(constant)
plt.title('Constant Padding')
plt.show()

reflect = cv2.copyMakeBorder(image, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
plt.imshow(reflect)
plt.title('Mirror Padding')
plt.show()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(constant)
plt.title('Constant Padding')

plt.subplot(1, 2, 2)
plt.imshow(reflect)
plt.title('Mirror Padding')

plt.show()