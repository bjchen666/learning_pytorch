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
        print 'a'
    else:
        img_new = np.zeros((h_crop, w_crop, 3), dtype='uint8')
        img_old = np.array(img, dtype='uint8')
        if h<h_crop and w<w_crop:
            img_new[:h, :w, :] = img_old[:, :, :]
            img = img_new
            print 'b'
        elif h<h_crop:
            w_random = cor[1]
            img_new[:h, :, :] = img_old[:, w_random:w_random+w_crop, :]
            img = img_new
            print 'c'
        else:
            h_random = cor[0]
            img_new[:, :w, :] = img_old[h_random:h_random+h_crop, :, :]
            img = img_new
            print 'd'
    if img_or_label==True:
        return img, cor
    else:
        return img


crop=(300,300)
img=Image.open('/home/gshdong/deeplab_demo/VOCdevkit/VOC2012/JPEGImages/2007_002212.jpg')
label=Image.open('/home/gshdong/deeplab_demo/VOCdevkit/VOC2012/SegmentationClass/2007_002212.png').convert('RGB')
result=img_crop(img,crop)
img=result[0]
lab=img_crop(label,crop,result[1],False)
_,test=plt.subplots(2,2)
test[0,0].imshow(img)
test[0,1].imshow(lab)
plt.show()
