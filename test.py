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

class MyRandomCrop():
    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
    def __call__(self, img):
        if self.padding > 0:
            img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w),(i,j)
    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class FixedCrop(object):
    def __init__(self, i, j, h, w, padding=0):
        self.i = i
        self.j = j
        self.h = h
        self.w = w
        self.padding = padding

    def __call__(self, img):
        if self.padding > 0:
            img = F.pad(img, self.padding)

        return F.crop(img, self.i, self.j, self.h, self.w)
im=Image.open('VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg')
im,cor = MyRandomCrop((320,480))(im)
im2=Image.open('VOCdevkit/VOC2012/SegmentationClass/2007_000033.png').convert('RGB')
im2=FixedCrop(cor[0],cor[1],320,480)(im2)
im3=Image.open('VOCdevkit/VOC2012/JPEGImages/2007_000480.jpg')
im3,cor = MyRandomCrop((320,480))(im3)
im4=Image.open('VOCdevkit/VOC2012/SegmentationClass/2007_000480.png').convert('RGB')
im4=FixedCrop(cor[0],cor[1],320,480)(im4)
# _,test=plt.subplots(2,2)
# test[0][0].imshow(im)
# test[0][1].imshow(im2)
# test[1][0].imshow(im3)
# test[1][1].imshow(im4)
#plt.show()

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
        lines_temp=f.readlines()
        for line in lines_temp:
            title = line.split('\n')[0]
            img_size=Image.open(self.root+'JPEGImages/'+title+'.jpg').size
            if img_size[0]<480 or img_size[1]<320:
                pass
            else:
                self.lines.append(title)

    def __getitem__(self, item):
        img=Image.open(self.root+'JPEGImages/'+self.lines[item]+'.jpg')
        target=Image.open(self.root+'SegmentationClass/'+self.lines[item]+'.png').convert('RGB')
        img,cor=MyRandomCrop((320,480))(img)
        target=FixedCrop(cor[0],cor[1],320,480)(target)
        if self.transforms is not None:
            img=self.transforms(img)
        target=torch.from_numpy(image2label(target))
        return img,target

    def __len__(self):
        return len(self.lines)

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

path_root='VOCdevkit/VOC2012/'
train_set=MyDataSet(path_root,train=True,transforms=im_aug)
train_data=DataLoader(train_set,shuffle=True,batch_size=32,num_workers=6)
eval_set=MyDataSet(path_root,train=False,transforms=im_aug)
eval_data=DataLoader(eval_set,shuffle=False,batch_size=32,num_workers=6)
pretrain=models.resnet34(pretrained=True)

class fcn(nn.Module):
    def __init__(self,class_nums):
        super(fcn, self).__init__()
        self.block1=nn.Sequential(*list(pretrain.children())[:-2])
        self.block2=nn.Sequential(nn.Conv2d(512,class_nums,1))
        self.block3=nn.Sequential(nn.ConvTranspose2d(class_nums,class_nums,16,8,4,bias=False))
        self.block4=nn.Sequential(nn.ConvTranspose2d(class_nums,class_nums,4,2,1,bias=False))
        self.block5=nn.Sequential(nn.ConvTranspose2d(class_nums,class_nums,4,2,1,bias=False))
        #self.block3.weight.data=bilin

    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        return x

net=fcn(21).cuda()
# print net

optimizer=torch.optim.SGD(net.parameters(),lr=0.5,weight_decay=0.0001)
criterion=nn.NLLLoss()

epoch=60

# def acc_nums(img,label):
#     nums=0
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             for k in range(img.shape[2]):
#                 if img[i,j,k]==label[i,j,k]:
#                     nums+=1
#     return nums

for e in range(epoch):
    train_acc=0
    train_loss=0
    train_acc_cls=0
    train_mean_iu=0
    train_fwavacc=0
    train_acc2=0
    train_loss2=0
    net.train()
    for im,label in train_data:
        im=Variable(im).cuda()
        label=Variable(label).cuda()
        out=net(im)
        out = NF.log_softmax(out, dim=1)
        predit=out.data.max(1)[1]
        loss=criterion(out,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss
        train_acc2+=(predit.data==label.data).sum()
        train_loss2+=loss.data

        label_pred = out.max(1)[1].data.cpu().numpy()
        label_true = label.data.cpu().numpy()
        for lbt, lbp in zip(label_true, label_pred):
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, 21)
            train_acc += acc
            train_acc_cls += acc_cls
            train_mean_iu += mean_iu
            train_fwavacc += fwavacc

    print 'Epoch: {}, Train Loss: {}, Train Acc: {}, Train Mean IU: {}'.format(e, train_loss / len(train_data), train_acc / len(train_set), train_mean_iu / len(train_set))
    print '**train_acc:{}'.format(int(train_acc2) * 1.0 / (320 * 480 * len(train_set)))
    # eval_acc=0
    # net.eval()
    # for im,label in eval_data:
    #     im=Variable(im).cuda()
    #     label=Variable(label).cuda()
    #     out=net(im)
    #     out = NF.log_softmax(out, dim=1)
    #     predit=out.data.max(1)[1]
    #     eval_acc+=(predit.data==label.data).sum()

    #print 'echo:{},train_acc:{},train_loss:{},eval:{}'.format(e,int(train_acc)*1.0/(320*480*len(train_set)),train_loss/len(train_set),int(eval_acc)*1.0/(320*480*len(eval_set)))

#torch.save(net.state_dict(),'fcn_chuchao.pth')