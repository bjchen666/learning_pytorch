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

epochs=30
crop_size=(320,480)
class_nums=21
batch_sizes=8
learn_rate=0.01

str='Time:'+time.ctime()+'\n'+'Epochs:'+str(epochs)+'  '+'Batch_sizes:'+str(batch_sizes)+'  '+'Learn_rate:'+str(learn_rate)+'\n'
with open('result_vgg.csv', 'a') as f:
    f.write(str + '\n')

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
    h_random = cor[0]
    w_random = cor[1]
    img = img.crop((w_random, h_random, w_random + w_crop, h_random + h_crop))
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
        else:
            f = open(root+'val.txt','r')
        lines_temp = f.readlines()
        for line in lines_temp:
            title = line.split('\n')[0]
            img_size = Image.open('VOCdevkit/VOC2012/JPEGImages/' + title + '.jpg').size
            if img_size[0] < 480 or img_size[1] < 320:
                pass
            else:
                self.lines.append(title)

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

# pretrain=models.vgg16(pretrained=True)
# net_list=list(pretrain.children())[0]
class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn, self).__init__()
        vgg = models.vgg16(pretrained=True)
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features3 = nn.Sequential(*features[: 17])
        self.features4 = nn.Sequential(*features[17: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        self.upscore2.weight.data.copy_(bilinear_kernel(num_classes, num_classes, 4))
        self.upscore_pool4.weight.data.copy_(bilinear_kernel(num_classes, num_classes, 4))
        self.upscore8.weight.data.copy_(bilinear_kernel(num_classes, num_classes, 16))

    def forward(self, x):
        x_size = x.size()
        pool3 = self.features3(x)
        pool4 = self.features4(pool3)
        pool5 = self.features5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(0.01 * pool4)
        upscore_pool4 = self.upscore_pool4(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
                                           + upscore2)

        score_pool3 = self.score_pool3(0.0001 * pool3)
        upscore8 = self.upscore8(score_pool3[:, :, 9: (9 + upscore_pool4.size()[2]), 9: (9 + upscore_pool4.size()[3])]
                                 + upscore_pool4)
        return upscore8[:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous()

net=fcn(class_nums).cuda()

train_set=MyDataSet(train=True,transform=img_aug)
train_data=DataLoader(train_set,batch_size=batch_sizes,shuffle=True,num_workers=6)
test_set=MyDataSet(train=False,transform=img_aug)
test_data=DataLoader(test_set,batch_size=batch_sizes,shuffle=True,num_workers=6)

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
    # loss_train=0
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
        # loss_train+=loss.data
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
    # str='Epoch:{},acc_train:{},miou_train:{},loss_train:{},eval_acc:{},miou_test:{}'.format(e,acc_train,miou_train,loss_train/len(train_set),eval_acc,miou_test)
    str = 'Epoch:{},acc_train:{},miou_train:{},eval_acc:{},miou_test:{}'.format(e, acc_train, miou_train, eval_acc,miou_test)
    print str
    with open('result_vgg.csv','a') as f:
        f.write(str+'\n')

    # torch.save({
    #     'epochs':e,
    #     'model_state_dict':net.state_dict(),
    #     'acc_train':acc_train,
    #     'acc_test':eval_acc,
    #     'miou_train':miou_train,
    #     'miou_test': miou_test
    # },'checkpoint.pth.tar')

    torch.save(net.state_dict(),'fcn_vgg16_fuse_ok_crop_index_myself.pth')
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
# net.load_state_dict(torch.load('fcn_vgg16_fuse_ok_crop_index_myself.pth'))
# net.eval()
# for i in range(t):
#     img, label = test_set[i+30]
#     title=test_set.lines[i+30]
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
