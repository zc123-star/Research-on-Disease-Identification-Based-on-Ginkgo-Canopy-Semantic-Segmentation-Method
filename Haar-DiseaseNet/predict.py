import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import Haar-DiseaseNet
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image
img = Image.open("../predict/healthy_/1.png")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)


# read class_indict
try:
    json_file = open('./class_indices.json','r')   
    class_indict = json.load(json_file)        

except Exception as e:
    print(e)
    exit(-1)


# create model
model = AlexNet(num_classes=5)

# load model weights
model_weights_path = "5class_Haar-DiseaseNet.pth"
model.load_state_dict(torch.load(model_weights_path))         

model.eval()
with torch.no_grad():
    #predicrt class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output,dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)],predict[predict_cla].item())
plt.show()


