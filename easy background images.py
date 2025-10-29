import cv2
import numpy as np

def overlay_mask_on_original(original_path, mask_path, output_path):
    original_image = cv2.imread(original_path)
    mask = cv2.imread(mask_path)

    red_mask = (mask[:, :, 2] > 150) & (mask[:, :, 1] < 100) & (mask[:, :, 0] < 100)

    original_image[~red_mask] = 0

    cv2.imwrite(output_path, original_image)

overlay_mask_on_original("original_image.jpg", "mask_image.jpg", "output_image.jpg")
