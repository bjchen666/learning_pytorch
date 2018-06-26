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

epochs=5
crop_size=(320,480)
class_nums=21
batch_sizes=16
learn_rate=0.01

np.seterr(invalid='ignore')

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

def each_nums(label_t,label_p,class_nums):
    each_num=np.bincount(class_nums*label_t+label_p,minlength=class_nums**2).reshape(class_nums,class_nums)
    return each_num

def iou_calculate(label_t,label_p,class_nums):
    result=np.zeros((class_nums,class_nums))
    for i,j in zip(label_t,label_p):
        result+=each_nums(i,j,class_nums)
    iou=np.diag(result)/(result.sum(0)+result.sum(1)-np.diag(result))
    return np.nanmean(iou)

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

pretrain=models.vgg16(pretrained=True)
net_list=list(pretrain.children())[0]
class fcn(nn.Module):
    def __init__(self,class_num):
        super(fcn, self).__init__()
        self.block1 = nn.Sequential(net_list[0:17])
        self.block2 = nn.Sequential(net_list[17:24])
        self.block3 = nn.Sequential(net_list[24:31])

        self.transfrom = nn.Conv2d(512, class_num, 1)

        self.bilinear = nn.ConvTranspose2d(class_num, class_num, 64, 32, 16, bias=False)
        self.bilinear.weight.data = bilinear_kernel(class_num, class_num, 64)

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.transfrom(x)
        x = self.bilinear(x)
        return x

net=fcn(class_nums).cuda()

train_set=MyDataSet(train=True,transform=img_aug)
train_data=DataLoader(train_set,batch_size=batch_sizes,shuffle=True,num_workers=6)
test_set=MyDataSet(train=False,transform=img_aug)
test_data=DataLoader(test_set,batch_size=batch_sizes,shuffle=True)

epochs_add=0

# if os.path.exists('checkpoint.pth.tar'):
#     checkpoint=torch.load('checkpoint.pth.tar')
#     net.load_state_dict(checkpoint['model_state_dict'])
#     epochs_add=checkpoint['epochs']+1

optimizer=torch.optim.SGD(net.parameters(),lr=learn_rate,weight_decay=0.0001)
criterion=nn.NLLLoss()

# '''
for e in range(epochs_add,epochs):
    acc_train=0
    miou_train=0
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
        miou=[iou_calculate(i,j,class_nums) for i,j in zip(predit.cpu().numpy(),label.cpu().numpy())]
        miou_train+=np.array(miou).mean()

    eval_acc = 0
    miou_test=0
    net.eval()
    for im, label in test_data:
        im = Variable(im).cuda()
        label = Variable(label).cuda()
        out = net(im)
        predit = out.data.max(1)[1]
        eval_acc += (predit.data == label.data).sum()
        miou = [iou_calculate(i, j, class_nums) for i, j in zip(predit.cpu().numpy(), label.cpu().numpy())]
        miou_test += np.array(miou).mean()

    acc_train = int(acc_train) * 1.0 / (320 * 480 * len(train_set))
    eval_acc=int(eval_acc)*1.0/(320*480*len(test_set))
    miou_train=miou_train/len(train_data)
    miou_test=miou_test/len(test_data)

    print 'Epoch:{},acc_train:{},miou_train:{},loss_train:{},eval_acc:{},miou_test:{}'.format(e,acc_train,miou_train,loss_train/len(train_set),eval_acc,miou_test)

    # torch.save({
    #     'epochs':e,
    #     'model_state_dict':net.state_dict(),
    #     'acc_train':acc_train,
    #     'acc_test':eval_acc,
    #     'miou_train':miou_train,
    #     'miou_test': miou_test
    # },'checkpoint.pth.tar')

# torch.save(net.state_dict(),'fcn_vgg16_fuse_crop_index_myself.pth')
# '''

#
# def to_three(x):
#     img_three = np.zeros((320, 480, 3), dtype='uint8')
#     for i in range(320):
#         for j in range(480):
#             img_three[i, j, :] = colormap[x[i, j]]
#     return img_three
# t=8
# _,display=plt.subplots(t,3)
# net.load_state_dict(torch.load('fcn_vgg16_fuse_crop_index_myself.pth'))
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
