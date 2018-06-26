from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms,datasets,models
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch import nn
from torch.autograd import Variable
import torch
import os
import time
import os
import shutil
import random
import numbers
import torchvision.transforms.functional as F
import torch.nn.functional as NF

epochs=10
crop_size=(320,480)
class_nums=21
batch_sizes=32

classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]
label_index = np.zeros(256 ** 3, dtype='int64')
for i in range(len(colormap)):
    temp = colormap[i]
    label_index[temp[0] * 256 * 256 + temp[1] * 256 + temp[2]] = i

def img_crop(img,crop_size,cor=(0,0),img_or_label=True):
    h = img.size[1]
    w=img.size[0]
    h_crop=crop_size[0]
    w_crop=crop_size[1]
    if img_or_label==True:
        cor = (random.randint(0, abs(h - h_crop)), random.randint(0, abs(w - w_crop)))
    if h>=h_crop and w>=w_crop:
        h_random = cor[0]
        w_random = cor[1]
        img = img.crop((w_random, h_random, w_random + w_crop, h_random + h_crop))
    else:
        img_new = np.zeros((h_crop, w_crop, 3), dtype='uint8')
        img_old = np.array(img, dtype='uint8')
        if h<h_crop and w<w_crop:
            img_new[:h, :w, :] = img_old[:, :, :]
            img = img_new
        elif h<h_crop:
            w_random = cor[1]
            img_new[:h, :, :] = img_old[:, w_random:w_random+w_crop, :]
            img = img_new
        else:
            h_random = cor[0]
            img_new[:, :w, :] = img_old[h_random:h_random+h_crop, :, :]
            img = img_new
    if img_or_label==True:
        return img, cor
    else:
        return img

def label_aug(label):
    label = np.array(label, dtype='int64')
    label = label[:,:,0]*256*256+label[:,:,1]*256+label[:,:,2]
    return torch.from_numpy(label_index[label])

def img_aug(img):
    aug=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    return aug(img)

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

class MyDataSet(Dataset):
    def __init__(self,train,transform=None):
        root='VOCdevkit/VOC2012/ImageSets/Segmentation/'
        self.train=train
        self.transform=transform
        self.lines=[]
        if self.train==True:
            f = open(root+'train.txt', 'r')
            for line in f.readlines():
                self.lines.append(line.split('\n')[0])
        else:
            f = open(root+'val.txt','r')
            for line in f.readlines():
                self.lines.append(line.split('\n')[0])

    def __getitem__(self, item):
        title=self.lines[item]
        img=Image.open('VOCdevkit/VOC2012/JPEGImages/'+title+'.jpg')
        label = Image.open('VOCdevkit/VOC2012/SegmentationClass/' + title + '.png').convert('RGB')

        img,cor=img_crop(img,crop_size)
        label = img_crop(label, crop_size, cor, False)

        if self.transform is not None:
            img = self.transform(img)

        label=label_aug(label)

        return img,label

    def __len__(self):
        return len(self.lines)

pretrain=models.resnet34(pretrained=True)
class fcn(nn.Module):
    def __init__(self,class_num):
        super(fcn, self).__init__()
        self.block1=nn.Sequential(*list(pretrain.children())[:-2])
        self.block2=nn.Conv2d(512,class_num,1)
        self.block3=nn.ConvTranspose2d(class_num,class_num,64,32,16,bias=False)
        self.block3.weight.data=bilinear_kernel(class_num,class_num,64)

    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        return x

## vgg16
'''
pretrain=models.vgg16(pretrained=True)
net_list=list(pretrain.children())[0]
class fcn(nn.Module):
    def __init__(self,class_num):
        super(fcn, self).__init__()
        self.block1 = nn.Sequential(net_list[0:17])
        self.block2 = nn.Sequential(net_list[17:24])
        self.block3 = nn.Sequential(net_list[24:31])

        self.transfrom1 = nn.Conv2d(256,class_num,1)
        self.transfrom2 = nn.Conv2d(512, class_num, 1)
        self.transfrom3 = nn.Conv2d(512, class_num, 1)

        self.bilinear1_2 = nn.ConvTranspose2d(class_num, class_num, 4, 2, 1, bias=False)
        self.bilinear1_2.weight.data = bilinear_kernel(class_num, class_num, 4)
        self.bilinear2_2 = nn.ConvTranspose2d(class_num, class_num, 4, 2, 1, bias=False)
        self.bilinear2_2.weight.data = bilinear_kernel(class_num, class_num, 4)
        self.bilinear3_8 = nn.ConvTranspose2d(class_num, class_num, 16, 8, 4, bias=False)
        self.bilinear3_8.weight.data = bilinear_kernel(class_num, class_num, 16)

    def forward(self,x):
        x = self.block1(x)
        s1=self.transfrom1(x)
        x = self.block2(x)
        s2=self.transfrom2(x)
        x = self.block3(x)
        s3=self.transfrom3(x)
        s3=self.bilinear1_2(s3)
        s2=s2+s3
        s2=self.bilinear2_2(s2)
        s1=s1+s2
        s1=self.bilinear3_8(s1)
        return s1
'''

train_set=MyDataSet(train=True,transform=img_aug)
train_data=DataLoader(train_set,batch_size=batch_sizes,shuffle=True,num_workers=6)
test_set=MyDataSet(train=False,transform=img_aug)
test_data=DataLoader(test_set,batch_size=batch_sizes,shuffle=True)

net=fcn(class_nums).cuda()
optimizer=torch.optim.SGD(net.parameters(),lr=0.1,weight_decay=0.0001)
criterion=nn.NLLLoss()
# '''
for e in range(epochs):
    acc_train=0
    loss_train=0
    net.train()
    for img,label in train_data:
        img=Variable(img).cuda()
        label=Variable(label).cuda()
        out=net(img)
        predit=out.data.max(1)[1]
        scores=NF.log_softmax(out,dim=1)
        loss=criterion(scores,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc_train+=(predit.data==label.data).sum()
        loss_train+=loss.data
    # print 'okkkk'
    print 'Epoch:{},acc_train:{},loss_train:{}'.format(e,int(acc_train)*1.0/(320*480*len(train_set)),loss_train/len(train_set))
torch.save(net.state_dict(),'fcn_crop_index_myself.pth')
# '''

##
# def to_three(x):
#     img_three = np.zeros((320, 480, 3), dtype='uint8')
#     for i in range(320):
#         for j in range(480):
#             img_three[i, j, :] = colormap[x[i, j]]
#     return img_three
# t=8
# _,display=plt.subplots(t,3)
# # net.load_state_dict(torch.load('fcn_chuchao.pth'))
# net.load_state_dict(torch.load('fcn_crop_index_myself.pth'))
# net.eval()
# for i in range(t):
#     img, label = test_set[i]
#     title=test_set.lines[i]
#     img=img.reshape(1,3,320,480)
#     img = Variable(img).cuda()
#     predit = net(img)
#     predit = predit.cpu()
#     predit = predit.data.max(1)[1]
#     predit = predit.data.squeeze().numpy()
#     label=label.numpy()
#     predit = to_three(predit)
#     label = to_three(label)
#     # temp = Image.fromarray(predit)
#     # temp.save('en'+str(i)+'.png', quality=100)
#     display[i,0].imshow(Image.open('VOCdevkit/VOC2012/JPEGImages/'+title+'.jpg'))
#     display[i,1].imshow(label)
#     display[i,2].imshow(predit)
# plt.show()
