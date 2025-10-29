import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image = cv2.imread(r"D:\ZSS_pytorch_classification\13.jpg")

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用中值滤波去噪
gray = cv2.medianBlur(gray, 5)

# 使用全局阈值进行二值化
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 对二值图像进行形态学操作以去除噪声
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# 确定背景区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 使用距离变换找到前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# 找到未知区域（即背景减去前景的部分）
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 标记前景
ret, markers = cv2.connectedComponents(sure_fg)

# 添加1，确保背景是1而不是0
markers = markers + 1

# 将未知区域标记为0
markers[unknown == 255] = 0

# 应用分水岭算法
markers = cv2.watershed(image, markers)

# 分割边界标记为红色
image[markers == -1] = [0, 0, 255]

# 显示结果
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('分割结果')
plt.subplot(122), plt.imshow(markers, cmap='gray'), plt.title('Markers')
plt.show()