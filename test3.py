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
from myutils import train
import os
import shutil
import random
import numbers
import torchvision.transforms.functional as F
import torch.nn.functional as NF

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

classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']

colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

cm2lbl = np.zeros(256**3)
for i,cm in enumerate(colormap):
    cm2lbl[cm[0]*256*256+cm[1]*256+cm[2]] = i

def image2label(img):
    data = np.array(img, dtype='int32')
    idx = data[:, :, 0] * 256 * 256 + data[:, :, 1] * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')

def im_aug(x):
    aug=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    return aug(x)

class MyDataSet(Dataset):
    def __init__(self,root,train=True,transforms=None):
        self.root=root
        self.train=train
        self.transforms=transforms
        if self.train==True:
            f = open(root + 'ImageSets/Segmentation/train.txt', 'r')
        else:
            f = open(root + 'ImageSets/Segmentation/val.txt', 'r')
        self.lines=[]
        for line in f.readlines():
            title = line.split('\n')[0]
            self.lines.append(title)

    def __getitem__(self, item):
        img=Image.open(self.root+'JPEGImages/'+self.lines[item]+'.jpg')
        target=Image.open(self.root+'SegmentationClass/'+self.lines[item]+'.png').convert('RGB')
        img,cor=img_crop(img,(320,480))
        target=img_crop(target,(320,480),cor,img_or_label=False)
        if self.transforms is not None:
            img=self.transforms(img)
        target=torch.from_numpy(image2label(target))
        return img,target

    def __len__(self):
        return len(self.lines)

path_root='VOCdevkit/VOC2012/'
train_set=MyDataSet(path_root,train=True,transforms=im_aug)
train_data=DataLoader(train_set,shuffle=True,batch_size=32,num_workers=6)
eval_set=MyDataSet(path_root,train=False,transforms=im_aug)
eval_data=DataLoader(eval_set,shuffle=False,batch_size=32,num_workers=6)

pretrain=models.resnet34(pretrained=True)

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

class fcn(nn.Module):
    def __init__(self,class_nums):
        super(fcn, self).__init__()
        self.block1=nn.Sequential(*list(pretrain.children())[:-2])
        self.block2=nn.Sequential(nn.Conv2d(512,class_nums,1))
        self.block3 = nn.ConvTranspose2d(class_nums, class_nums, 64, stride=32, padding=16, bias=False)
        self.block3.weight.data = bilinear_kernel(class_nums, class_nums, 64)

    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x = self.block3(x)
        return x

net=fcn(21).cuda()

optimizer=torch.optim.SGD(net.parameters(),lr=0.1,weight_decay=0.0001)
criterion=nn.NLLLoss()

epoch=30

# '''
for e in range(epoch):
    train_acc=0
    train_loss=0
    net.train()
    for im,label in train_data:
        im=Variable(im).cuda()
        label=Variable(label).cuda()
        print 'label',label.type()
        print 'image',im.type()
        out=net(im)
        print 'out',out.type()
        predit = out.data.max(1)[1]
        score = NF.log_softmax(out, dim=1)
        print 'scores',score.type()
        loss=criterion(score,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc+=(predit.data==label.data).sum()
        train_loss+=loss.data

    eval_acc=0
    net.eval()
    for im,label in eval_data:
        im=Variable(im).cuda()
        label=Variable(label).cuda()
        out=net(im)
        predit=out.data.max(1)[1]
        eval_acc+=(predit.data==label.data).sum()

    print 'echo:{},train_acc:{},train_loss:{},eval:{}'.format(e,int(train_acc)*1.0/(320*480*len(train_set)),train_loss/len(train_data),int(eval_acc)*1.0/(320*480*len(eval_set)))

torch.save(net.state_dict(),'fcn_chuchao_crop_myself.pth')
# '''

## display by multitle
# net.load_state_dict(torch.load('fcn_chuchao.pth'))
# img=Image.open('/home/gshdong/deeplab_demo/VOCdevkit/VOC2012/JPEGImages/2007_002212.jpg')#2007_002088 #2007_002212 #2007_001825 #2007_004768
# img,cor=MyRandomCrop((320,480))(img)
# img=im_aug(img).reshape(1,3,320,480)
# img=Variable(img).cuda()
# net.eval()
# predit=net(img)
# predit=predit.cpu()
# predit=predit.data.max(1)[1]
# predit=predit.data.squeeze().numpy()
# eight=0
# for i in range(320):
#     for j in range(480):
#         if predit[i,j]==8:
#             eight += 1
# print eight*1.0/(320*480)
# img_three=np.zeros((320,480,3),dtype='uint8')
# for i in range(320):
#     for j in range(480):
#         img_three[i,j,:]=colormap[predit[i,j]]
# plt.imshow(img_three)
# plt.show()

## display by plt
# def label_index(x):
#     img_three = np.zeros((320, 480, 3), dtype='uint8')
#     for i in range(320):
#         for j in range(480):
#             img_three[i, j, :] = colormap[x[i, j]]
#     return img_three
# t=8
# _,display=plt.subplots(t,3)
# # net.load_state_dict(torch.load('fcn_chuchao.pth'))
# net.load_state_dict(torch.load('fcn_chuchao_crop_myself.pth'))
# net.eval()
# for i in range(t):
#     img,label=eval_set[i+50]
#     title=eval_set.lines[i+50]
#     img=img.reshape(1,3,320,480)
#     img = Variable(img).cuda()
#     predit = net(img)
#     predit = predit.cpu()
#     predit = predit.data.max(1)[1]
#     predit = predit.data.squeeze().numpy()
#     label=label.numpy()
#     predit = label_index(predit)
#     label = label_index(label)
#     display[i,0].imshow(Image.open('/home/gshdong/deeplab_demo/VOCdevkit/VOC2012/JPEGImages/'+title+'.jpg'))
#     display[i,1].imshow(label)
#     display[i,2].imshow(predit)
# plt.show()





