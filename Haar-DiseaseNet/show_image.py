import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义数据增强的转换操作
data_transform = transforms.Compose([
    #transforms.RandomResizedCrop(224),

    transforms.RandomRotation(degrees=50),  # 旋转角度范围为 -30 到 30 度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),#亮度、对比度（颜色抖动）
    #transforms.RandomHorizontalFlip(),#随机水平翻转

    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 显示图像的函数
def show_image(tensor, title=""):
    # 反标准化，将图像值从 (-1, 1) 范围转回到 (0, 1)
    tensor = tensor * 0.5 + 0.5
    np_image = tensor.numpy().transpose(1, 2, 0)
    plt.imshow(np_image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# 加载图像
image_paths = ["42.png", "765.png", "1599.png","3885.png","4052.png"]  # 替换为你实际的图像路径
images = [Image.open(img_path) for img_path in image_paths]

# 对图像进行数据增强并显示
for i, img in enumerate(images):
    print(f"Original Image {i+1}:")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    for j in range(3):  # 对每张图像应用三次数据增强
        transformed_img = data_transform(img)
        print(f"Transformed Image {i+1}, Version {j+1}:")
        show_image(transformed_img)