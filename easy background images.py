import cv2
import numpy as np

def overlay_mask_on_original(original_path, mask_path, output_path):
    # 读取原始图像和掩膜
    original_image = cv2.imread(original_path)
    mask = cv2.imread(mask_path)

    # 将红色区域当作掩膜（阈值可根据实际情况调整）
    red_mask = (mask[:, :, 2] > 150) & (mask[:, :, 1] < 100) & (mask[:, :, 0] < 100)

    # 在原始图像中将背景设置为零
    original_image[~red_mask] = 0

    # 保存结果
    cv2.imwrite(output_path, original_image)

# 示例用法
overlay_mask_on_original("original_image.jpg", "mask_image.jpg", "output_image.jpg")