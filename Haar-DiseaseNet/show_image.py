import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

data_transform = transforms.Compose([
    #transforms.RandomResizedCrop(224),

    transforms.RandomRotation(degrees=50), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def show_image(tensor, title=""):
    tensor = tensor * 0.5 + 0.5
    np_image = tensor.numpy().transpose(1, 2, 0)
    plt.imshow(np_image)
    plt.title(title)
    plt.axis('off')
    plt.show()

image_paths = ["42.png", "765.png", "1599.png","3885.png","4052.png"]  
images = [Image.open(img_path) for img_path in image_paths]

for i, img in enumerate(images):
    print(f"Original Image {i+1}:")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    for j in range(3):  
        transformed_img = data_transform(img)
        print(f"Transformed Image {i+1}, Version {j+1}:")

        show_image(transformed_img)
