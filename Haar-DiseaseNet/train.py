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
from model import AlexNet
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

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # 获取图像的路径
image_path = data_root + "/data_set/flower_data/"  # flower data set path ## 获取图像路径

#导入训练集
train_dataset = datasets.ImageFolder(root=os.path.join(image_path+ "train"), transform=data_transform["train"])
train_num = len(train_dataset)

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
#导入验证集
validate_dataset = datasets.ImageFolder(root=os.path.join(image_path+ "val"),  transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=0)

#转换分类的字典的键的顺序
flower_list = train_dataset.class_to_idx
sorted_flower_list = dict(sorted(flower_list.items(), key=lambda item: item[1]))
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
#开始训练
model_name = "AlexNet"
net = AlexNet( num_classes=5, init_weights=True)
model = net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

#定义训练
def train_model(model, device, train_loader,optimizer, epoch):
    train_loss = 0
    Best_acc = 0
    Correct = 0
    model.train()

    for step, data in enumerate(train_loader, start=0):  ##遍历训练集，step从0 开始计算
        images, labels = data  ##获取训练集的图像和标签
        optimizer.zero_grad()  ##清除历史梯度
        time_start = time.perf_counter()
        outputs = model(images.to(device))  ##正向传播
        pre_y = torch.max(outputs, dim=1)[1]
        Correct += (pre_y == labels.to(device)).sum().item()
        loss = loss_function(outputs, labels.to(device))  ##计算损失
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


save_path = './5class_AlexNet.pth'#保存训练权重路径



##定义测试方法
def test_model(model,device,validate_loader):
    model.eval()
    correct = 0
    best_acc = 0#最优准确率
    test_loss =0#测试损失
    with torch.no_grad():#不更新历史梯度 参数进行更新，防止计算过大
        for val_data in validate_loader:#遍历验证集
            val_images, val_labels = val_data#划分验证集  图片  标签
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            outputs = model(val_images.to(device))
            test_loss += loss_function(outputs, val_labels).item()#计算验证集的损失，用于后续评估
            predict_y = torch.max(outputs, dim=1)[1]  #预测和标签进行对比 以output中值最大位置对应的索引（标签）作为预测输出
            #print(val_labels)
            #print(predict_y)
            correct += (predict_y == val_labels.to(device)).sum().item()#累计验证集中，预测正确样本个数

        val_accurate = correct / val_num#预测正确个数/样本总数=测试集准确率

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
print('model%s' % min_index)
print('验证集最高准确率')
print('{}%'.format(Valid_Accuracy_list[min_index]))

model.eval()

accuracy = test_model(model, device, validate_loader)
print('测试集准确率')
print('{}%'.format(accuracy[1]))

#绘图定义
plt.rcParams['font.sans-serif'] = ['SimHei']       ##显示中文标签
plt.rcParams['axes.unicode_minus'] = False          ##这两行需要手动设置
x1 = range(0, 150)
y1 = Train_loss_list
y3 = Valid_Accuracy_list
y2 = Valid_loss_list
y4 = Train_Acc_list

#保存数据
pdata = pd.DataFrame({'number': x1, 'Train_loss': y1, 'Valid_loss': y2,'Valid_Acc':y3, 'Train_Acc': y4})
pdata.to_excel('./150epoch_AlexNet.xlsx', index=True, float_format='%.4f')
df1 = pd.read_excel('./150epoch_AlexNet.xlsx')
# #绘图
# plt.subplot(411)
# plt.legend('Train_Loss')
# plt.plot(x1, y1, 'red', label="Train_Loss")
# plt.legend(loc = 'best')
# #plt.xlabel('轮数')
# plt.ylabel('训练集损失')
#
# plt.subplot(412)
# plt.legend('Val_Loss')
# plt.plot(x1, y2, 'yellow', label="Val_Loss")
# plt.legend(loc = 'best')
# # plt.xlabel('轮数')
# plt.ylabel('验证集损失')
#
# plt.subplot(413)
# plt.plot(x1, y4, 'cyan', label="Train_Acc")
# plt.legend(loc = 'best')
# # plt.xlabel('轮数')
# plt.ylabel('训练集准确率')
#
# plt.subplot(414)
# plt.plot(x1, y3, 'blue', label="Valid_Acc")
# plt.legend(loc = 'best')
# plt.xlabel('轮数')
# plt.ylabel('验证集准确率')

plt.show()
