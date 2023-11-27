import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from torchvision import models
import torch.utils.data as Data
from torch import nn
from d2l import torch as d2l

root = '/home/ubutnu/hardDisk/DeepLearning/imagenet'
def get_imagenet(root, transform=None,train = True):
    if train:
        root = os.path.join(root, 'train')
    else:
        root = os.path.join(root, 'val')
    return datasets.ImageFolder(root = root,
                transform=transform)


def resnet():
    net= torchvision.models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return net
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def init_weight(m):
     if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

def right_num(y_hat, y):  #@save
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n
        
    def add(self,*args):
        self.data=[a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def my_train(net,train_iter,test_iter,num_epochs,lr,device):
    net.apply(init_weight)
    net.to(device)
    
    optimizer = torch.optim.SGD(net.parameters() ,lr=lr)
    lr_period, lr_decay = 4, 0.9
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 60, 90], gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=60,eta_min=0.1)
    loss = nn.CrossEntropyLoss()
    num_batches = len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(3)
        #scheduler.step()
        for i,(X,y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat,y)       
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        scheduler.step()
        
        # test_acc = evaluate_accuracy_gpu(net, test_iter)
        #animator.add(epoch + 1, (None, None, test_acc))
    return train_l,train_acc,test_acc

train_dataset = get_imagenet(root,
            transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))

lr, num_epochs,batch_size= 0.0002, 50, 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = resnet()
#print(net)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size,num_workers=8,shuffle=False)
#valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=batch_size,num_workers=8)

train_l,train_acc,test_acc=my_train(net,train_loader,train_loader,num_epochs,lr,device)