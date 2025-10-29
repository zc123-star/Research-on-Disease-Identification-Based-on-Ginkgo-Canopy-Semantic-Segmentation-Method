import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import AlexNet
from utils import GradCAM, show_cam_on_image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # Initialize your model architecture (e.g., VGG16)
    model = AlexNet( num_classes=5, init_weights=True)

    # Load the pre-trained weights into the model
    model_weights_path = "5class_AlexNet.pth"  # Replace with your weight file path
    model.load_state_dict(
        torch.load(model_weights_path, map_location=torch.device('cpu')))  # Map location for CPU usage

    # Switch to evaluation mode
    model.eval()

    # Define the target layers for GradCAM
    target_layers = [model.features[-1]]  # Use the last layer of features for GradCAM

    # Data transformations (Normalization is crucial as it should match what the model was trained with)
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224 (assuming this size was used during training)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])

    # Load image
    img_path = "4yangguangzhuoshao.jpg"
    assert os.path.exists(img_path), f"File: '{img_path}' does not exist."
    img = Image.open(img_path).convert('RGB')

    # Apply transformations
    img_tensor = data_transform(img)

    # Expand batch dimension (from [C, H, W] to [N, C, H, W])
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # Set target category (the class index you're interested in, e.g., 1 for some specific class)
    target_category =4 # Adjust based on your target class index

    # Compute GradCAM
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    # The grayscale_cam has a shape of [1, 224, 224], squeeze it to [224, 224]
    grayscale_cam = grayscale_cam[0, :]

    # Resize the grayscale_cam to the original image size
    original_size = img.size  # (width, height)
    grayscale_cam_resized = np.array(Image.fromarray(grayscale_cam).resize(original_size, Image.BILINEAR))

    # Visualize the CAM on the original image
    visualization = show_cam_on_image(np.array(img) / 255.0, grayscale_cam_resized, use_rgb=True)

    # Show the visualization
    plt.imshow(visualization)
    plt.axis('off')
    plt.show()

    # Save the visualization image


if __name__ == '__main__':
    main()