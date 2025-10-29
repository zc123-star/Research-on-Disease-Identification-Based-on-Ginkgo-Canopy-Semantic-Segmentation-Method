import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# 确保避免OpenMP相关错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义数据增强的转换操作
data_transform = transforms.Compose([
    transforms.RandomRotation(degrees=50),  # 旋转角度范围为 -50 到 50 度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 亮度、对比度、饱和度、色调抖动
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 输入和输出文件夹路径
input_folder = r"D:\Deep-learning-for-image-processing-master - 副本\data_set\flower_data\train\sunscald"  # 替换为你实际的输入文件夹路径
output_folder = r"D:\Deep-learning-for-image-processing-master - 副本\data_set\flower_data\train\sunscald(1)"  # 替换为你实际的输出文件夹路径

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有图像文件名
image_paths = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# 对每张图像进行数据增强并保存
for img_name in image_paths:
    img_path = os.path.join(input_folder, img_name)
    img = Image.open(img_path)

    for i in range(2):  # 对每张图像应用三次数据增强
        transformed_img = data_transform(img)

        # 反标准化并转换为PIL图像
        transformed_img = transformed_img * 0.5 + 0.5
        transformed_img = transforms.ToPILImage()(transformed_img)

        # 构造增强后的图像文件名并保存
        output_img_name = f"{os.path.splitext(img_name)[0]}_augmented_{i+1}.png"
        output_img_path = os.path.join(output_folder, output_img_name)
        transformed_img.save(output_img_path)

        print(f"Saved: {output_img_path}")

print("数据增强和保存操作已完成。")