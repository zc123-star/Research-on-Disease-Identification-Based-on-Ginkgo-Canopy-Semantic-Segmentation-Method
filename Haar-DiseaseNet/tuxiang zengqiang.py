import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

data_transform = transforms.Compose([
    transforms.RandomRotation(degrees=50),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

input_folder = r"D:\Deep-learning-for-image-processing-master - 副本\data_set\flower_data\train\sunscald"  
output_folder = r"D:\Deep-learning-for-image-processing-master - 副本\data_set\flower_data\train\sunscald(1)"  

os.makedirs(output_folder, exist_ok=True)

image_paths = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

for img_name in image_paths:
    img_path = os.path.join(input_folder, img_name)
    img = Image.open(img_path)

    for i in range(2): 
        transformed_img = data_transform(img)

        transformed_img = transformed_img * 0.5 + 0.5
        transformed_img = transforms.ToPILImage()(transformed_img)

        output_img_name = f"{os.path.splitext(img_name)[0]}_augmented_{i+1}.png"
        output_img_path = os.path.join(output_folder, output_img_name)
        transformed_img.save(output_img_path)

        print(f"Saved: {output_img_path}")

