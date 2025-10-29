import os
import sys
import json
import pandas as pd
import openpyxl

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from model import Haar-DiseaseNet
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  
image_path = data_root + "/data_set/flower_data/"  

train_dataset = datasets.ImageFolder(root=os.path.join(image_path+ "train"), transform=data_transform["train"])
train_num = len(train_dataset)

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
validate_dataset = datasets.ImageFolder(root=os.path.join(image_path+ "val"),  transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=0)

flower_list = train_dataset.class_to_idx
sorted_flower_list = dict(sorted(flower_list.items(), key=lambda item: item[1]))
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
model_name = "AlexNet"
net = AlexNet( num_classes=5, init_weights=True)
model = net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

def train_model(model, device, train_loader,optimizer, epoch):
    train_loss = 0
    Best_acc = 0
    Correct = 0
    model.train()

    for step, data in enumerate(train_loader, start=0):  
        images, labels = data  
        optimizer.zero_grad()  
        time_start = time.perf_counter()
        outputs = model(images.to(device))  
        pre_y = torch.max(outputs, dim=1)[1]
        Correct += (pre_y == labels.to(device)).sum().item()
        loss = loss_function(outputs, labels.to(device))  
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            train_loss += loss.item()
            print('Train Epoch : {} \t train Loss : {:.6f} \t Time:{}'.format(epoch, loss.item(), time_start))
    train_accurate = Correct / train_num
    if train_accurate > Best_acc:
        Best_acc = train_accurate
        torch.save(model.state_dict(), save_path)
    print('Train_acc:{:.3f}'.format(100 * Correct / len(train_loader.dataset)))
    Acc = 100 * Correct / len(train_loader.dataset)

    return train_loss, Acc


save_path = './5class_AlexNet.pth'




def test_model(model,device,validate_loader):
    model.eval()
    correct = 0
    best_acc = 0
    test_loss =0
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            outputs = model(val_images.to(device))
            test_loss += loss_function(outputs, val_labels).item()
            predict_y = torch.max(outputs, dim=1)[1]  
            #print(val_labels)
            #print(predict_y)
            correct += (predict_y == val_labels.to(device)).sum().item()

        val_accurate = correct / val_num

        if val_accurate > best_acc:
             best_acc = val_accurate
             torch.save(model.state_dict(), save_path)


        test_loss /= len(validate_loader.dataset)
        print('Val_loss : {: .4f}, Accuracy : {: .3f}%\n'.format(test_loss,100*correct / len(validate_loader.dataset)))

        acc = 100 * correct / len(validate_loader.dataset)

        return  test_loss, acc



list = []
Train_loss_list = []
Train_Acc_list = []
Valid_loss_list = []
Valid_Accuracy_list = []
for epoch in range(0, 150):
    train_loss, Acc = train_model(model, device, train_loader, optimizer, epoch)
    Train_loss_list.append(train_loss)
    Train_Acc_list.append(Acc)

    test_loss, acc = test_model(model, device, validate_loader,)
    Valid_loss_list.append(test_loss)
    Valid_Accuracy_list.append(acc)
    list.append(test_loss)

min_num = list[0]
min_index = 0
for iii in range(len(list)):
    if list[iii] < min_num:
        min_num = list[iii]
        min_index = iii

minloss = min_num
model.eval()

accuracy = test_model(model, device, validate_loader)

plt.rcParams['font.sans-serif'] = ['SimHei']       
plt.rcParams['axes.unicode_minus'] = False          
x1 = range(0, 150)
y1 = Train_loss_list
y3 = Valid_Accuracy_list
y2 = Valid_loss_list
y4 = Train_Acc_list

#保存数据
pdata = pd.DataFrame({'number': x1, 'Train_loss': y1, 'Valid_loss': y2,'Valid_Acc':y3, 'Train_Acc': y4})
pdata.to_excel('./150epoch_Haar-DiseaseNet.xlsx', index=True, float_format='%.4f')
df1 = pd.read_excel('./150epoch_Haar-DiseaseNet.xlsx')

