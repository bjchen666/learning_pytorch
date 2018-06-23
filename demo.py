import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,models
from PIL import Image
from torch.utils import trainer
import time
from myutils import train
import os

pretrain_model=models.resnet18(pretrained=True)
dim_in=pretrain_model.fc.in_features
pretrain_model.fc=nn.Linear(dim_in,10)
net=pretrain_model.cuda()

def data_tf(x):
    x = x.resize((224, 224), 2)
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x

train_set = datasets.CIFAR10('./data2', train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
test_set = datasets.CIFAR10('./data2', train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1,weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

def num_correct(out, label):
    out_index = out.max(1)[1]
    num = 0
    for i in range(len(out)):
        if out_index.data[i] == label.data[i]:
            num += 1
    return num

def learnrate(optim,lr):
    for i in optim.param_groups:
        i['lr']=lr

epoches=40

for e in range(epoches):
    if e==20:
        learnrate(optimizer,0.01)
    train_loss = 0
    train_acc = 0
    temp = 0
    start=time.time()
    net.train()
    for imdata,label in train_data:
        # temp += 1
        # print 'Total', temp, '/', len(train_data)
        imdata=Variable(imdata).cuda()
        label=Variable(label).cuda()
        print 'imdata',imdata.shape
        print 'imdata',imdata
        print 'label',label
        predit=net(imdata)
        print 'predit',predit
        loss=criterion(predit,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss
        train_acc += num_correct(predit, label) * 1.0 / imdata.shape[0]
    eval_loss = 0
    eval_acc = 0
    net.eval()
    for imdata, label in test_data:
        imdata = Variable(imdata).cuda()
        label = Variable(label).cuda()
        predit = net(imdata)
        loss = criterion(predit, label)
        eval_loss += loss.data
        eval_acc += num_correct(predit, label) * 1.0 / imdata.shape[0]
    end = time.time()
    print 'epoch:{} train_loss:{:.6f} train_acc:{:.6f} test_loss:{:.6f} test_acc:{:.6f} time:{}'.format(e, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data), eval_acc / len(test_data),end-start)

## about numpy
'''
list=[[1,2,3],[4,5,6],[7,8,9]]
print np.array(list)
print np.empty((6,3))
print np.zeros((6,3))
print np.ones((6,3))
print np.arange(6)
print (np.arange(6))**2
print np.arange(15,30,2)
print np.linspace(0,10,11)
a=np.array([[1,1],[0,1]])
b=np.array([[2,0],[3,4]])
print a*b
print np.dot(a,b)
print np.random.random((6,3))
print a.sum()
print b.max()
print b.min()
print np.arange(15)[3:8]
c=np.arange(15).reshape(5,3)
print c[1,:]
print c[1]
print c.T
'''

## fashion-mnist by myself design network
'''
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16,0.001),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32,0.001),
            nn.ReLU(),
            nn.MaxPool2d(2,2) #48
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64,0.001),
            nn.ReLU(),
            nn.MaxPool2d(2,2) #24
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.001),
            nn.ReLU(),
            nn.MaxPool2d(3,3) #8
        )
        self.fc=nn.Sequential(
            nn.Linear(8*8*128,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=x.view(x.shape[0],-1)
        return self.fc(x)

net=MyNet().cuda()

def img_aug(x):
    aug=transforms.Compose(
        [
            transforms.Resize((96,96)),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ]
    )
    return aug(x)

class MYDATASET(Dataset):
    def __init__(self,root,transform=None,train=True):
        self.root=root
        self.transform=transform
        f=open(root+'train.txt','r')
        self.lines=f.readlines()

    def __getitem__(self, item):
        line=self.lines[item].split('//')[1].split(' ')
        img=Image.open(self.root+line[0]).convert('RGB')
        if self.transform is not None:
            img=self.transform(img)
        target=int(line[1].split('\n')[0][7])
        return img,target

    def __len__(self):
        return len(self.lines)

batch_size=64
learn_rate=0.1
echo=20

optimizer=torch.optim.SGD(net.parameters(),lr=learn_rate,weight_decay=0.0001)
criterion=nn.CrossEntropyLoss()

train_set=MYDATASET('fashion-mnist/',transform=img_aug)
train_data=DataLoader(train_set,shuffle=True,batch_size=batch_size,num_workers=6)
test_set=MYDATASET('fashion-mnist/',transform=img_aug)
test_data=DataLoader(test_set,batch_size=batch_size,num_workers=6)

for e in range(echo):
    acc_num=0
    net.train()
    for img,label in train_data:
        img=Variable(img).cuda()
        label=Variable(label).cuda()
        predit=net(img)
        loss = criterion(predit, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predit=predit.data.max(1)[1]
        acc_num+=(predit==label).sum()
    acc_val_num = 0
    net.eval()
    for img, label in test_data:
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        predit = net(img)
        predit = predit.data.max(1)[1]
        acc_val_num += (predit == label).sum()
    print 'Echo:{},train_acc:{},eval_acc:{}'.format(e,float(acc_num.data)/len(train_set),float(acc_val_num.data)/len(test_set))
    #print 'Echo:{},train_acc:{}'.format(e, float(acc_num.data) / len(train_set))

torch.save(net.state_dict(),'fashion-mnist.pth')

# net.load_state_dict(torch.load('fashion-mnist.pth'))
# path='fashion-mnist/test/'
# title_list=os.listdir(path)
# data_num=len(title_list)
# net.eval()
# for i in range(data_num):
#     im=Image.open(path+title_list[i]).convert('RGB')
#     im=img_aug(im).reshape(1,3,96,96)
#     im=Variable(im).cuda()
#     predit=net(im)
#     predit=predit.data.max(1)[1]
#     index=int(predit)
#     if index==0:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/0')
#     elif index==1:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/1')
#     elif index==2:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/2')
#     elif index==3:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/3')
#     elif index==4:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/4')
#     elif index==5:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/5')
#     elif index==6:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/6')
#     elif index==7:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/7')
#     elif index==8:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/8')
#     elif index==9:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/9')
'''

## fashion-mnist by pretrain
'''
pre_model=models.resnet18(pretrained=True)
dim_in=pre_model.fc.in_features
pre_model.fc=nn.Linear(dim_in,10)
net=pre_model.cuda()

def img_aug(x):
    aug=transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ]
    )
    return aug(x)

class MYDATASET(Dataset):
    def __init__(self,root,transform=None,train=True):
        self.root=root
        self.transform=transform
        f=open(root+'train.txt','r')
        self.lines=f.readlines()

    def __getitem__(self, item):
        line=self.lines[item].split('//')[1].split(' ')
        img=Image.open(self.root+line[0]).convert('RGB')
        if self.transform is not None:
            img=self.transform(img)
        target=int(line[1].split('\n')[0][7])
        return img,target

    def __len__(self):
        return len(self.lines)

batch_size=64
learn_rate=0.01
echo=6

optimizer=torch.optim.SGD(net.parameters(),lr=learn_rate,weight_decay=0.0001)
criterion=nn.CrossEntropyLoss()

train_set=MYDATASET('fashion-mnist/',transform=img_aug)
train_data=DataLoader(train_set,shuffle=True,batch_size=batch_size)
test_set=MYDATASET('fashion-mnist/',transform=img_aug)
test_data=DataLoader(test_set,batch_size=batch_size)

for e in range(echo):
    acc_num=0
    net.train()
    for img,label in train_data:
        img=Variable(img).cuda()
        label=Variable(label).cuda()
        predit=net(img)
        loss = criterion(predit, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predit=predit.data.max(1)[1]
        acc_num+=(predit==label).sum()
    acc_val_num = 0
    net.eval()
    for img, label in test_data:
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        predit = net(img)
        predit = predit.data.max(1)[1]
        acc_val_num += (predit == label).sum()
    print 'Echo:{},train_acc:{},eval_acc:{}'.format(e,float(acc_num.data)/len(train_set),float(acc_val_num.data)/len(test_set))
    #print 'Echo:{},train_acc:{}'.format(e, float(acc_num.data) / len(train_set))

torch.save(net.state_dict(),'fashion-mnist.pth')

# pre_model=models.resnet18(pretrained=True)
# dim_in=pre_model.fc.in_features
# pre_model.fc=nn.Linear(dim_in,10)
# pre_model.load_state_dict(torch.load('fashion-mnist.pth'))
# net=pre_model.cuda()
# path='fashion-mnist/test/'
# title_list=os.listdir(path)
# data_num=len(title_list)
# net.eval()
# for i in range(data_num):
#     im=Image.open(path+title_list[i]).convert('RGB')
#     im=img_aug(im).reshape(1,3,224,224)
#     im=Variable(im).cuda()
#     predit=net(im)
#     predit=predit.data.max(1)[1]
#     index=int(predit)
#     if index==0:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/0')
#     elif index==1:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/1')
#     elif index==2:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/2')
#     elif index==3:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/3')
#     elif index==4:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/4')
#     elif index==5:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/5')
#     elif index==6:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/6')
#     elif index==7:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/7')
#     elif index==8:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/8')
#     elif index==9:
#         shutil.copy(path+title_list[i],'fashion-mnist/result/9')
'''

## wan luo shang bie ren xie de ji cheng Dataset shi yong DataLoader
'''
# -----------------ready the dataset--------------------------
root="E:/fashion_mnist/"
def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

train_data=MyDataset(txt=root+'train.txt', transform=transforms.ToTensor())
test_data=MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)
'''

## dog vs cat by ji cheng Dataset chong xie Dataset lai shi yong DataLoader
'''
class MYDATASET(Dataset):
    def __init__(self,root,train=True,transform=None,download=False):
        self.root=root
        self.train=train
        self.transform=transform
        self.title_list = os.listdir(root)

    def __getitem__(self, item):
        img = Image.open(self.root + self.title_list[item])
        class_label = self.title_list[item].split('.')[0]
        if class_label == 'dog':
            target = 0
        else:
            target = 1

        if self.transform is not None:
            img=self.transform(img)
        return img,target

    def __len__(self):
        return len(self.title_list)

def img_aug(x):
    aug=transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ]
    )
    return aug(x)

batch_size=64
epoches=8
learn_rate=0.01

pre_model=models.resnet18(pretrained=True)
dim_in=pre_model.fc.in_features
pre_model.fc=nn.Linear(dim_in,2)
net=pre_model.cuda()

optimer=torch.optim.SGD(net.parameters(),lr=learn_rate,weight_decay=0.0001)
criterion=nn.CrossEntropyLoss()

train_set=MYDATASET('dogorcat/train/dog/',transform=img_aug)
train_data=DataLoader(train_set,shuffle=True,batch_size=batch_size)
test_set=MYDATASET('dogorcat/val/dog/',transform=img_aug)
test_data=DataLoader(test_set,shuffle=True,batch_size=batch_size)

def train_acc(x,y):
    acc=0
    j=0
    for i in x.data.max(1)[1]:
        if i==y[j]:
            acc+=1
        j+=1
    return acc

for e in range(epoches):
    acc_num=0
    net.train()
    for im,lab in train_data:
        im=Variable(im).cuda()
        lab=Variable(lab).cuda()
        predit=net(im)
        loss = criterion(predit, lab)
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        acc_num += train_acc(predit, lab)

    acc_val_num = 0
    net.eval()
    for im,lab in test_data:
        im=Variable(im).cuda()
        lab=Variable(lab).cuda()
        predit=net(im)
        acc_val_num += train_acc(predit, lab)

    print 'Echo:{},train_acc:{},eval_acc:{}'.format(e, acc_num * 1.0 / len(train_set),acc_val_num*1.0/len(test_set))
'''

## dog vs cat by mei you shi yong Dataset he DataLoader
'''
# title_list=os.listdir('kaggle/train')
# for i in range(len(title_list)):
#     dogorcat=title_list[i][0:3]
#     if dogorcat=='dog':
#         shutil.copy('kaggle/train/'+str(title_list[i]),'dogorcat/train/dog')
#     else:
#         shutil.copy('kaggle/train/'+str(title_list[i]),'dogorcat/train/cat')
# dog_list=os.listdir('dogorcat/train/dog')
# cat_list=os.listdir('dogorcat/train/cat')
# dog_len=len(dog_list)
# cat_len=len(cat_list)
# for i in range(int(dog_len*0.9),dog_len):
#     shutil.move('dogorcat/train/dog/' + str(dog_list[i]), 'dogorcat/val/dog')
# for i in range(int(cat_len*0.9),cat_len):
#     shutil.move('dogorcat/train/cat/' + str(cat_list[i]), 'dogorcat/val/cat')

# title_list=os.listdir('dogorcat/train/cat')
# for title in title_list:
#     shutil.move('dogorcat/train/cat/'+title, 'dogorcat/train/dog')

# title_list=os.listdir('dogorcat/val/cat')
# for title in title_list:
#     shutil.move('dogorcat/val/cat/'+title, 'dogorcat/val/dog')

def im_aug(x):
    aug=transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ]
    )
    return aug(x)

batch_size=64
epoches=8
learn_rate=0.01

pre_model=models.resnet18(pretrained=True)
dim_in=pre_model.fc.in_features
pre_model.fc=nn.Linear(dim_in,2)
net=pre_model.cuda()

optimer=torch.optim.SGD(net.parameters(),lr=learn_rate,weight_decay=0.0001)
criterion=nn.CrossEntropyLoss()

path='dogorcat/train/dog/'
all_list=os.listdir(path)
random.shuffle(all_list)
all_num=len(all_list)
all_data=torch.Tensor(all_num,3,224,224)
all_label=torch.LongTensor(all_num)
for i in range(all_num):
    im=all_list[i]
    if im[0:3]=='dog':
        all_label[i]=0
    else:
        all_label[i] = 1
    im=Image.open(path+im)
    im = im_aug(im)
    all_data[i] = im

path='dogorcat/val/dog/'
all_list=os.listdir(path)
random.shuffle(all_list)
all_val_num=len(all_list)
all_val_data=torch.Tensor(all_val_num,3,224,224)
all_val_label=torch.LongTensor(all_val_num)
for i in range(all_val_num):
    im=all_list[i]
    if im[0:3]=='dog':
        all_val_label[i]=0
    else:
        all_val_label[i] = 1
    im=Image.open(path+im)
    im = im_aug(im)
    all_val_data[i] = im

def train_acc(x,y):
    acc=0
    j=0
    for i in x.data.max(1)[1]:
        if i==y[j]:
            acc+=1
        j+=1
    return acc

for e in range(epoches):
    i=0
    j=0
    acc_num=0
    net.train()
    while i<all_num:
        if i+batch_size>all_num:
            j=all_num
        else:
            j=i+batch_size
        image=Variable(all_data[i:j]).cuda()
        lab=Variable(all_label[i:j]).cuda()
        predit= net(image)
        loss=criterion(predit,lab)
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        acc_num+=train_acc(predit,lab)
        i=i+batch_size

    i = 0
    j = 0
    acc_val_num = 0
    net.eval()
    while i < all_val_num:
        if i + batch_size > all_val_num:
            j = all_val_num
        else:
            j = i + batch_size
        image = Variable(all_val_data[i:j]).cuda()
        lab = Variable(all_val_label[i:j]).cuda()
        predit = net(image)
        acc_val_num += train_acc(predit, lab)
        i = i + batch_size

    print 'Echo:{},train_acc:{},eval_acc:{}'.format(e, acc_num * 1.0 / all_num,acc_val_num*1.0/all_val_num)

torch.save(net.state_dict(),'dog_vs_cat.pth')

## shi ji tu pian  che shi
# pre_model=models.resnet18(pretrained=True)
# dim_in=pre_model.fc.in_features
# pre_model.fc=nn.Linear(dim_in,2)
# pre_model.load_state_dict(torch.load('dog_vs_cat.pth'))
# net=pre_model.cuda()
# 
# def im_aug(x):
#     aug=transforms.Compose(
#         [
#             transforms.Resize((224,224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
#         ]
#     )
#     return aug(x)
# 
# path='dogorcat/test2/'
# title_list=os.listdir(path)
# all_num=len(title_list)#-11200
# data_predit=range(all_num)
# all_data=torch.Tensor(all_num,3,224,224)
# for i in range(all_num):
#     im = title_list[i]
#     im = Image.open(path + im)
#     im = im_aug(im)
#     all_data[i] = im
# 
# net.eval()
# for i in range(all_num):
#     image = Variable(all_data[i:i+1]).cuda()
#     predit = net(image)
#     data_predit[i]=predit.data.max(1)[1][0]
# 
# for i in range(all_num):
#     if '{}'.format(data_predit[i])=='0':
#         data_label='dog'
#         shutil.copy(path+title_list[i],'dogorcat/result2/dog/')
#     else:
#         data_label='cat'
#         shutil.copy(path+title_list[i],'dogorcat/result2/cat/')
'''

## shi yong ImageNet yv xun lian resnet18 at cifar
'''
pretrain_model=models.resnet18(pretrained=True)
dim_in=pretrain_model.fc.in_features
pretrain_model.fc=nn.Linear(dim_in,10)
net=pretrain_model.cuda()

def data_tf(x):
    x = x.resize((224, 224), 2)
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x

train_set = datasets.CIFAR10('./data2', train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = datasets.CIFAR10('./data2', train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1,weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

def num_correct(out, label):
    out_index = out.max(1)[1]
    num = 0
    for i in range(len(out)):
        if out_index.data[i] == label.data[i]:
            num += 1
    return num

def learnrate(optim,lr):
    for i in optim.param_groups:
        i['lr']=lr

epoches=40

for e in range(epoches):
    if e==20:
        learnrate(optimizer,0.01)
    train_loss = 0
    train_acc = 0
    temp = 0
    start=time.time()
    net.train()
    for imdata,label in train_data:
        # temp += 1
        # print 'Total', temp, '/', len(train_data)
        imdata=Variable(imdata).cuda()
        label=Variable(label).cuda()
        predit=net(imdata)
        loss=criterion(predit,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss
        train_acc += num_correct(predit, label) * 1.0 / imdata.shape[0]
    eval_loss = 0
    eval_acc = 0
    net.eval()
    for imdata, label in test_data:
        imdata = Variable(imdata).cuda()
        label = Variable(label).cuda()
        predit = net(imdata)
        loss = criterion(predit, label)
        eval_loss += loss.data
        eval_acc += num_correct(predit, label) * 1.0 / imdata.shape[0]
    end = time.time()
    print 'epoch:{} train_loss:{:.6f} train_acc:{:.6f} test_loss:{:.6f} test_acc:{:.6f} time:{}'.format(e, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data), eval_acc / len(test_data),end-start)

torch.save(net.state_dict(),'resnet18_pre_cifar.pth')
'''

## shi yong yi jing xun lian hao de cifar mo xing jin xing zhen shi tu pian de shi bie
'''
# pretrain_model=models.resnet18(pretrained=True)
# dim_in=pretrain_model.fc.in_features
# pretrain_model.fc=nn.Linear(dim_in,10)
# net=pretrain_model
# net.load_state_dict(torch.load('resnet18_pre_cifar.pth'))
# #print pretrain_model
# im=Image.open('dog5.jpg')
# im=transforms.Resize((224,224))(im)
# im=transforms.ToTensor()(im)
# im=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])(im)
# im=im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
# im=Variable(im)
# net.eval()
# predit=net(im)
# label=predit.max(1)[1][0]
# label_list=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
# print 'Class:',label_list[label]

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1=nn.Sequential(
            conv3x3(in_channel, out_channel, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.conv2=nn.Sequential(
            conv3x3(out_channel, out_channel),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)

        self.relu3=nn.ReLU()

    def forward(self, x):
        out=self.conv1(x)
        out=self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = self.relu3(x + out)
        return out

class resnet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)

        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            residual_block(64, 64),
            residual_block(64, 64)
        )

        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128)
        )

        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256)
        )

        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512),
            nn.AvgPool2d(3)
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

im=Image.open('horse2.jpg')
im=transforms.Resize(96)(im)
im=transforms.ToTensor()(im)
im=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])(im)
im=im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
im=Variable(im)
net=resnet(3,10)
net.load_state_dict(torch.load('resnet_cifar.pth'))
net.eval()
predit=net(im)
label=predit.max(1)[1][0]
label_list=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
print 'Class:',label_list[label]
'''

## MNIST by juan ji sheng jing wang luo
'''
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,16,3,1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(16,32,3,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(32,64,3,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer4=nn.Sequential(
            nn.Conv2d(64,128,3,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc=nn.Sequential(
            nn.Linear(128*4*4,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

def im_aug(x):
    data_aug=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])   
        ]
    )
    return data_aug(x)
'''

## zhi ji de shu jv jing guo yi lun qian xian chuang bo
'''
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        print 'enen:', x.shape
        x = self.layer2(x)
        print 'enen:', x.shape
        x = self.layer3(x)
        print 'enen:', x.shape
        x = self.layer4(x)
        print 'enen:', x.shape
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        print 'enen:', x.shape
        return x

data = torch.Tensor(1, 1, 28, 28)
net = MyNet()
print net(data)
'''

## jian shu jv ji li de tu pian xian shi chu lai
'''
# train_data=datasets.CIFAR10('./data2',train=True)
# row=5
# col=5
# label=np.empty((row,col))
# _,test=plt.subplots(row,col)
# for i in range(row):
#     for j in range(col):
#         test[i][j].imshow(np.array(train_data[i*row+j][0]))
#         test[i][j].get_xaxis().set_visible(False)
#         test[i][j].get_yaxis().set_visible(False)
#         label[i][j]=train_data[i*row+j][1]
# print label
# plt.show()


# data=datasets.MNIST('./data',train=True)
# pil=np.array(data[7][0],dtype='float32')
# plt.imshow(pil,cmap='gray')
# plt.show()

train_set = datasets.CIFAR10('./data2', train=True,download=True)
pil=np.array(train_set[7][0],dtype='float32')
plt.imshow(pil)
plt.show()
'''

## shu jv zhen qiang
'''
# im=Image.open('cat.png')
# # im=transforms.Resize(80)(im)
# # im=transforms.Resize((100,50))(im)
# # im=transforms.RandomCrop((150,120))(im)
# # im=transforms.CenterCrop(100)(im)
# # im=transforms.RandomHorizontalFlip()(im)
# # im=transforms.RandomVerticalFlip()(im)
# # im=transforms.RandomRotation(40)(im)
# im=transforms.ColorJitter(brightness=0)(im)
# im=transforms.ColorJitter(hue=0.5)(im)
# plt.imshow(im)
# plt.show()

im=Image.open('cat.png')
im_aug=transforms.Compose([transforms.Resize(120),
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(120),
                           transforms.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5)])
row=3
col=3
_,test=plt.subplots(row,col)
for i in range(row):
    for j in range(col):
        test[i][j].imshow(im_aug(im))
        test[i][j].get_xaxis().set_visible(False)
        test[i][j].get_yaxis().set_visible(False)
plt.show()
'''

## ResNet with weight decay and lr shuai jiang
'''
def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1=nn.Sequential(
            conv3x3(in_channel, out_channel, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.conv2=nn.Sequential(
            conv3x3(out_channel, out_channel),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)

        self.relu3=nn.ReLU()

    def forward(self, x):
        out=self.conv1(x)
        out=self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = self.relu3(x + out)
        return out

class resnet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)

        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            residual_block(64, 64),
            residual_block(64, 64)
        )

        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128)
        )

        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256)
        )

        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512),
            nn.AvgPool2d(3)
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

def data_tf(x):
    x = x.resize((96, 96), 2)
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x

train_set = datasets.CIFAR10('./data2', train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_set = datasets.CIFAR10('./data2', train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

net = resnet(3, 10).cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1,weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

def num_correct(out, label):
    out_index = out.max(1)[1]
    num = 0
    for i in range(len(out)):
        if out_index.data[i] == label.data[i]:
            num += 1
    return num

def learnrate(optim,lr):
    for i in optim.param_groups:
        i['lr']=lr

epoches=40

for e in range(epoches):
    if e==20:
        learnrate(optimizer,0.01)
    train_loss = 0
    train_acc = 0
    temp = 0
    start=time.time()
    net.train()
    for imdata,label in train_data:
        # temp += 1
        # print 'Total', temp, '/', len(train_data)
        imdata=Variable(imdata).cuda()
        label=Variable(label).cuda()
        predit=net(imdata)
        loss=criterion(predit,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss
        train_acc += num_correct(predit, label) * 1.0 / imdata.shape[0]
    eval_loss = 0
    eval_acc = 0
    net.eval()
    for imdata, label in test_data:
        imdata = Variable(imdata).cuda()
        label = Variable(label).cuda()
        predit = net(imdata)
        loss = criterion(predit, label)
        eval_loss += loss.data
        eval_acc += num_correct(predit, label) * 1.0 / imdata.shape[0]
    end = time.time()
    print 'epoch:{} train_loss:{:.6f} train_acc:{:.6f} test_loss:{:.6f} test_acc:{:.6f} time:{}'.format(e, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data), eval_acc / len(test_data),end-start)
'''

## GoogLeNet at CIFAR
'''
def block(indata,outdata,kernel,stride=1,padding=0):
    conv=nn.Sequential(nn.Conv2d(indata,outdata,kernel,stride,padding),nn.BatchNorm2d(outdata,0.001),nn.ReLU())
    return conv

class testNet(nn.Module):
    def __init__(self,indata,out1_1,out2_1,out2_3,out3_1,out3_5,out4_1):
        super(testNet, self).__init__()
        self.layer1=nn.Sequential(block(indata,out1_1,1))
        self.layer2=nn.Sequential(block(indata,out2_1,1),block(out2_1,out2_3,3,padding=1))
        self.layer3=nn.Sequential(block(indata,out3_1,1),block(out3_1,out3_5,5,padding=2))
        self.layer4=nn.Sequential(nn.MaxPool2d(kernel_size=3,stride=1,padding=1),block(indata,out4_1,1))

    def forward(self,x):
        conv1=self.layer1(x)
        conv2 = self.layer2(x)
        conv3 = self.layer3(x)
        conv4 = self.layer4(x)
        result=torch.cat((conv1,conv2,conv3,conv4),dim=1)
        return result

class googlenet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(googlenet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Sequential(
            block(in_channel,64,7,2,3),
            nn.MaxPool2d(3, 2)
        )

        self.block2 = nn.Sequential(
            block(64, 64,1),
            block(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(3, 2)
        )

        self.block3 = nn.Sequential(
            testNet(192, 64, 96, 128, 16, 32, 32),
            testNet(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2)
        )

        self.block4 = nn.Sequential(
            testNet(480, 192, 96, 208, 16, 48, 64),
            testNet(512, 160, 112, 224, 24, 64, 64),
            testNet(512, 128, 128, 256, 24, 64, 64),
            testNet(512, 112, 144, 288, 32, 64, 64),
            testNet(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2)
        )

        self.block5 = nn.Sequential(
            testNet(832, 256, 160, 320, 32, 128, 128),
            testNet(832, 384, 182, 384, 48, 128, 128),
            nn.AvgPool2d(2,2)
        )

        self.classifier = nn.Sequential(nn.Linear(1024, num_classes))

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

def data_tf(x):
    x = x.resize((96, 96), 2)
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x

train_set = datasets.CIFAR10('./data2', train=True, transform=data_tf)
test_set = datasets.CIFAR10('./data2', train=False, transform=data_tf)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_data = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

net = googlenet(3, 10).cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

def num_correct(out, label):
    out_index = out.max(1)[1]
    num = 0
    for i in range(len(out)):
        if out_index.data[i] == label.data[i]:
            num += 1
    return num

epoches=20

for e in range(epoches):
    train_loss = 0
    train_acc = 0
    temp = 0
    start=time.time()
    net.train()
    for imdata,label in train_data:
        # temp += 1
        # print 'Total', temp, '/', len(train_data)
        imdata=Variable(imdata).cuda()
        label=Variable(label).cuda()
        predit=net(imdata)
        loss=criterion(predit,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss
        train_acc += num_correct(predit, label) * 1.0 / imdata.shape[0]
    eval_loss = 0
    eval_acc = 0
    net.eval()
    for imdata, label in test_data:
        imdata = Variable(imdata).cuda()
        label = Variable(label).cuda()
        predit = net(imdata)
        loss = criterion(predit, label)
        eval_loss += loss.data
        eval_acc += num_correct(predit, label) * 1.0 / imdata.shape[0]
    end = time.time()
    print 'epoch:{} train_loss:{:.6f} train_acc:{:.6f} test_loss:{:.6f} test_acc:{:.6f} time:{}'.format(e, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data), eval_acc / len(test_data),end-start)
'''

## VGG at CIFAR
'''
def block(nums,inputs,outputs):
    net=[nn.Conv2d(inputs,outputs,kernel_size=3,padding=1),nn.ReLU()]
    for i in range(nums-1):
        net.append(nn.Conv2d(outputs,outputs,kernel_size=3,padding=1))
        net.append(nn.ReLU())
    net.append(nn.MaxPool2d(2,2))
    return nn.Sequential(*net)

def test(nums_list,inout_list):
    net=[]
    for a,b in zip(nums_list,inout_list):
        net.append(block(a,b[0],b[1]))
    return net

net=nn.Sequential(*test((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512))))

class VggNet(nn.Module):
    def __init__(self):
        super(VggNet,self).__init__()
        self.feature=net
        self.classifier=nn.Sequential(nn.Linear(512,100),nn.ReLU(),nn.Linear(100,10))
    def forward(self,x):
        x=self.feature(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x

#print VggNet()

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x

train_set = datasets.CIFAR10('./data2', train=True, transform=data_tf,download=True)
test_set = datasets.CIFAR10('./data2', train=False, transform=data_tf,download=True)
train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_data = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

vggnet=VggNet().cuda()

optimer=torch.optim.SGD(vggnet.parameters(),lr=0.1)
criterion=nn.CrossEntropyLoss()

def num_correct(out, label):
    out_index = out.max(1)[1]
    num = 0
    for i in range(len(out)):
        if out_index.data[i] == label.data[i]:
            num += 1
    return num

epoches=20

for e in range(epoches):
    train_loss = 0
    train_acc = 0
    temp = 0
    start=time.time()
    vggnet.train()
    for imdata,label in train_data:
        # temp += 1
        # print 'Total', temp, '/', len(train_data)
        imdata=Variable(imdata).cuda()
        label=Variable(label).cuda()
        predit=vggnet(imdata)
        loss=criterion(predit,label)
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        train_loss+=loss
        train_acc += num_correct(predit, label) * 1.0 / imdata.shape[0]
    eval_loss = 0
    eval_acc = 0
    vggnet.eval()
    for imdata, label in test_data:
        imdata = Variable(imdata).cuda()
        label = Variable(label).cuda()
        predit = vggnet(imdata)
        loss = criterion(predit, label)
        eval_loss += loss.data
        eval_acc += num_correct(predit, label) * 1.0 / imdata.shape[0]
    end = time.time()
    print 'epoch:{} train_loss:{:.6f} train_acc:{:.6f} test_loss:{:.6f} test_acc:{:.6f} time:{}'.format(e, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data), eval_acc / len(test_data),end-start)
'''

## zhuan hua wei hui du tu xian
'''
im=Image.open('cat.jpg').convert('L')
im=np.array(im)
plt.imshow(im,cmap='gray')
plt.show()
'''

## juan ji de yi ge san tong dao zhen me bian chen yi tong dao by nn.Conv2d and myself
'''
## 3-->1
im=torch.Tensor(1,3,28,28)
conv=nn.Conv2d(3,1,3,bias=False)
result=conv(im)
print result.data.squeeze()
q=conv.weight
im=im.reshape(3,28,28)
w=im.shape[1]
h=im.shape[2]
kernel=q.squeeze()
k=3
cell=np.empty((3,h-k+1,w-k+1))
for d in range(3):
    for i in range(k - 1, h):
        for j in range(k - 1, w):
            cell[d,i - k + 1, j - k + 1] = (im[d,i - k + 1:i + 1, j - k + 1:j + 1] * kernel[d]).sum()
one=cell[0]
two=cell[1]
three=cell[2]
total=one+two+three
print total

## 3-->5
im=torch.Tensor(1,3,28,28)
conv=nn.Conv2d(3,5,3,bias=False)
result=conv(im)
q=conv.weight

w=im.shape[2]
h=im.shape[3]
kernel=q
k=3
cell=np.empty((5,3,h-k+1,w-k+1))
cell2=np.zeros((1,5,26,26))
for n in range(5):
    for d in range(3):
        for i in range(k - 1, h):
            for j in range(k - 1, w):
                cell[n,d, i - k + 1, j - k + 1] = (im[0,d, i - k + 1:i + 1, j - k + 1:j + 1] * kernel[n,d]).sum()
for i in range(5):
    for j in range(3):
        cell2[0,i]+=cell[i,j]
print 'result',result
print 'myself',cell2
'''

## juan ji cao zuo by myself
'''
im=Image.open('cat.jpg').convert('L')
im=np.array(im,dtype='float32')
w=im.shape[1]
h=im.shape[0]
kernel=[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
kernel=np.array(kernel)
k=kernel.shape[0]
cell=np.empty((h-k+1,w-k+1))
for i in range(k-1,h):
    for j in range(k-1,w):
        cell[i-k+1,j-k+1]=(im[i-k+1:i+1,j-k+1:j+1]*kernel).sum()
plt.imshow(cell,cmap='gray')
plt.show()
'''

## juan ji cao zuo by nn.Conv2d
'''
# im=Image.open('cat.jpg').convert('L')
# im=np.array(im,dtype='float32')
# im=torch.from_numpy(im.reshape(1,1,im.shape[0],im.shape[1]))
# kernel=np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32').reshape(1,1,3,3)
# kernel=torch.from_numpy(kernel)
# conv=nn.Conv2d(1,1,3,bias=False)
# conv.weight.data=kernel
# result=conv(Variable(im))
# result=result.data.squeeze().numpy()
# plt.imshow(result,cmap='gray')
# plt.show()

im=Image.open('cat.jpg').convert('L')
im=np.array(im,dtype='float32')
im=im.reshape(1,1,im.shape[0],im.shape[1])
im=torch.from_numpy(im)
kernel=[[-1, -1, -1], [-1, 8, -1], [-1, -1, -2]]
kernel=np.array(kernel,dtype='float32')
kernel=kernel.reshape(1,1,3,3)
kernel=torch.from_numpy(kernel)
conv=nn.Conv2d(1,1,3)
conv.weight.data=kernel
result=conv(im)
im=result.data.numpy().reshape(result.shape[2],result.shape[3])
plt.imshow(im,cmap='gray')
plt.show()
'''

## zui da ci hua
'''
im=Image.open('cat.jpg').convert('L')
im=np.array(im,dtype='float32')
print 'before:',im.shape[0],im.shape[1]
im=torch.from_numpy(im.reshape(1,1,im.shape[0],im.shape[1]))
pool=nn.MaxPool2d(2,2)
result=pool(Variable(im))
result=result.data.squeeze().numpy()
print 'after:',result.shape[0],result.shape[1]
plt.imshow(result,cmap='gray')
plt.show()
'''

## MNIST shi bie by Module
'''
batch_sizes=64
epoches=40
learn_rate=0.1

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x

train_set = datasets.MNIST('./data', train=True, transform=data_tf, download=True)
test_set = datasets.MNIST('./data', train=False, transform=data_tf, download=True)

train_data = DataLoader(train_set, batch_size=batch_sizes, shuffle=True)
test_data = DataLoader(test_set, batch_size=batch_sizes*2, shuffle=False)

class Hello_Net(nn.Module):
    def __init__(self,in_dim,hidden_1,hidden_2,hidden_3,out_dim):
        super(Hello_Net,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(in_dim,hidden_1),nn.ReLU())
        self.layer2=nn.Sequential(nn.Linear(hidden_1,hidden_2),nn.ReLU())
        self.layer3=nn.Sequential(nn.Linear(hidden_2,hidden_3),nn.ReLU())
        self.layer4=nn.Sequential(nn.Linear(hidden_3,out_dim),nn.ReLU())

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        return x

model=Hello_Net(784,400,200,100,10).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

def num_correct(out, label):
    out_index = out.max(1)[1]
    num = 0
    for i in range(len(out)):
        if out_index.data[i] == label.data[i]:
            num += 1
    return num

for e in range(epoches):
    train_loss = 0
    train_acc = 0
    temp = 0
    model.train()
    for im, label in train_data:
        #temp+=1
        #print 'Total',temp,'/',len(train_data)
        im = Variable(im).cuda()
        label = Variable(label).cuda()
        out = model(im)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        train_acc += num_correct(out, label) * 1.0 / im.shape[0]

    eval_loss = 0
    eval_acc = 0
    model.eval()
    for im, label in test_data:
        im = Variable(im).cuda()
        label = Variable(label).cuda()
        out = model(im)
        loss = criterion(out, label)
        eval_loss += loss.data
        eval_acc += num_correct(out, label) * 1.0 / im.shape[0]
    print 'epoch:{} train_loss:{:.6f} train_acc:{:.6f} test_loss:{:.6f} test_acc:{:.6f}'.format(e, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data), eval_acc / len(test_data))

torch.save(model, './model_module.pth')
'''

## MNIST by Sequential
'''
batch_sizes=64
epoches=40
learn_rate=0.1

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x

train_set = datasets.MNIST('./data', train=True, transform=data_tf, download=True)
test_set = datasets.MNIST('./data', train=False, transform=data_tf, download=True)

train_data = DataLoader(train_set, batch_size=batch_sizes, shuffle=True)
test_data = DataLoader(test_set, batch_size=batch_sizes, shuffle=False)

net = nn.Sequential(
    nn.Linear(784, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)
#net=torch.load('model.pth')
net=net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate)

def num_correct(out,label):
    out_index=out.max(1)[1]
    num = 0
    for i in range(len(out)):
        if out_index.data[i]==label.data[i]:
            num+=1
    return num

for e in range(epoches):
    train_loss = 0
    train_acc = 0
    num_corrects=0
    temp=0
    net.train()
    for im, label in train_data:
        #temp+=1
        #print 'Total',temp,'/',len(train_data)
        im = Variable(im).cuda()
        label = Variable(label).cuda()
        out = net(im)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        train_acc+=num_correct(out, label)*1.0/im.shape[0]

    eval_loss = 0
    eval_acc = 0
    net.eval()
    for im, label in test_data:
        im = Variable(im).cuda()
        label = Variable(label).cuda()
        out = net(im)
        loss = criterion(out, label)
        eval_loss += loss.data
        eval_acc += num_correct(out, label) * 1.0 / im.shape[0]

    print 'epoch:{} train_loss:{:.6f} train_acc:{:.6f} test_loss:{:.6f} test_acc:{:.6f}'.format(e,train_loss/len(train_data),train_acc/len(train_data),eval_loss/len(test_data),eval_acc/len(test_data))

torch.save(net,'./model.pth')
'''

##logistic hui gui, er fen lei, pytorch function
'''
xa_train=[]
ya_train=[]
xb_train=[]
yb_train=[]
f=open('data.txt','r+')
data=[(float(str.split(',')[0]),float(str.split(',')[1]),int(str.split(',')[2][0])) for str in f.readlines()]
data=np.array(data)
data[:,0]=data[:,0]/max(data[:,0])
data[:,1]=data[:,1]/max(data[:,1])

for i in range(len(data)):
    if data[i][2]==1:
        xa_train.append(data[i][0])
        ya_train.append(data[i][1])
    else:
        xb_train.append(data[i][0])
        yb_train.append(data[i][1])

x_train=torch.from_numpy(np.array(data)[:,0:2]).float()
y_train=torch.from_numpy(np.array(data)[:,2]).unsqueeze(1).float()

w=nn.Parameter(torch.randn(2,1))
b=nn.Parameter(torch.zeros(1))
x_train=Variable(x_train)
y_train=Variable(y_train)

optimer=torch.optim.SGD([w,b],lr=1)  #0.1 xiao guo ye bu cuo,dan shi 0.01 xiao guo bian de fei chang cha
criterion=nn.BCEWithLogitsLoss()

def test_model(x):
    return F.sigmoid(torch.mm(x, w)+b)

num_echo=10000

for echo in range(num_echo):
    yyy = test_model(x_train)
    loss=criterion(yyy,y_train)
    optimer.zero_grad()
    loss.backward()
    optimer.step()
print loss

x=np.arange(0.2,1,0.01)
w=w.data.numpy()
b=b.data.numpy()
y=(-w[0][0]*x-b[0])/w[1][0]
plt.plot(x,y,'y-',label='Class:Logits')
plt.plot(xa_train,ya_train,'b.',label='Class:1')
plt.plot(xb_train,yb_train,'r.',label='Class:0')
plt.legend()
plt.show()
'''

##logistic hui gui, er fen lei, myself
'''
xa_train=[]
ya_train=[]
xb_train=[]
yb_train=[]
f=open('data.txt','r+')
data=[(float(str.split(',')[0]),float(str.split(',')[1]),int(str.split(',')[2][0])) for str in f.readlines()]
data=np.array(data)
data[:,0]=data[:,0]/max(data[:,0])
data[:,1]=data[:,1]/max(data[:,1])

for i in range(len(data)):
    if data[i][2]==1:
        xa_train.append(data[i][0])
        ya_train.append(data[i][1])
    else:
        xb_train.append(data[i][0])
        yb_train.append(data[i][1])

x_train=torch.from_numpy(np.array(data)[:,0:2]).float()
y_train=torch.from_numpy(np.array(data)[:,2]).unsqueeze(1).float()

w=Variable(torch.randn(2,1),requires_grad=True)
b=Variable(torch.zeros(1),requires_grad=True)
x_train=Variable(x_train)
y_train=Variable(y_train)

def test_model(x):
    return F.sigmoid(torch.mm(x, w)+b)

def test_loss(indata,outdata):
    logits = (-(outdata * indata.log() + (1 - outdata) * (1 - indata).log())).mean()
    return logits

num_echo=10000
rate_learn=1  #0.1 xiao guo ye bu cuo,dan shi 0.01 xiao guo bian de fei chang cha

for echo in range(num_echo):
    yyy = test_model(x_train)
    loss=test_loss(yyy,y_train)
    loss.backward()
    w.data = w.data - rate_learn * w.grad.data
    b.data = b.data - rate_learn * b.grad.data
    w.grad.zero_()
    b.grad.zero_()
print loss

x=np.arange(0.2,1,0.01)
w=w.data.numpy()
b=b.data.numpy()
y=(-w[0][0]*x-b[0])/w[1][0]
plt.plot(x,y,'y-',label='Class:Logits')
plt.plot(xa_train,ya_train,'b.',label='Class:1')
plt.plot(xb_train,yb_train,'r.',label='Class:0')
plt.legend()
plt.show()
'''

##duo wei xiang xing mo xing
'''
w_target=np.array([0.5,3,2.4])
b_target=np.array([0.9])

x=np.arange(-3,3.1,0.1)
y=b_target+w_target[0]*x+w_target[1]*x**2+w_target[2]*x**3

w=Variable(torch.randn(3,1),requires_grad=True)
b=Variable(torch.zeros(1),requires_grad=True)

x_train=Variable(torch.from_numpy(np.stack([x**i for i in range(1,4)],1)).float())
y_train=Variable(torch.from_numpy(y).float().unsqueeze(1))

def test_model(x):
    return torch.mm(x, w) + b
def test_loss(indata,outdata):
    return torch.mean((indata-outdata)**2)

num_echo=100
rate_learn=0.001

for echo in range(num_echo):
    yyy = test_model(x_train)
    loss=test_loss(yyy,y_train)
    loss.backward()
    w.data=w.data-rate_learn*w.grad
    b.data=b.data-rate_learn*b.grad
    w.grad.data.zero_()
    b.grad.data.zero_()
    print loss

plt.plot(x,y,'b-',label='old')
plt.plot(x,yyy.data.numpy(),'r-',label='new')
plt.xlabel('x')
plt.xlabel('y')
plt.show()
'''

##yi wei xiang xing mo xing
'''
x_train=np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],[9.779],[6.182],[7.59],[2.167],[7.042],[10.791],[5.313],[7.997],[3.1]], dtype=np.float32)
y_train=np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],[3.366],[2.596],[2.53],[1.221],[2.827],[3.465],[1.65],[2.904],[1.3]],dtype=np.float32)

x_train=torch.from_numpy(x_train)
y_train=torch.from_numpy(y_train)

w=Variable(torch.randn(1),requires_grad=True)
b=Variable(torch.zeros(1),requires_grad=True)

x_train=Variable(x_train)
y_train=Variable(y_train)

def test_model(x):
    return w*x+b

def test_loss(indata,outdata):
    return torch.mean((indata-outdata)**2)

num_echo=100
rate_learn=0.001

for echo in range(num_echo):
    yyy = test_model(x_train)
    loss=test_loss(yyy,y_train)
    loss.backward()
    w.data=w.data-rate_learn*w.grad
    b.data=b.data-rate_learn*b.grad
    w.grad = torch.zeros(1)
    b.grad = torch.zeros(1)
    print loss

plt.plot(x_train.data.numpy(),y_train.data.numpy(),'b+',label='old')
plt.plot(x_train.data.numpy(),yyy.data.numpy(),color='red',label='new')
plt.legend()
plt.show()
'''

##Python hui tu gong jv matplotlib
'''
a=np.array([[1,2],[3,4],[5,6],[7,8]])
x=np.linspace(-10,10,150)
y=np.sin(x)
plt.plot(x,y,'b+',label='y=sin(x)')
plt.plot(x,y+1,'r.',label='y=sin(x)+1')

#plt.subplot(221)
#plt.subplot(222)
#plt.subplot(212)

plt.title('cool')
plt.xlabel('xx')
plt.ylabel('yy')
plt.legend()  #plt.legend(loc=1)
#plt.savefig('dong.jpg')
plt.show()
'''

##ji shuan ti du
'''
x=Variable(torch.randn(3),requires_grad=True)
y=x*2
print y
y.backward(torch.FloatTensor([1,0.8,0.01]))
print x.grad
'''

##ji shuan ti du
'''
x=Variable(torch.Tensor([1]),requires_grad=True)
w=Variable(torch.Tensor([2]),requires_grad=True)
b=Variable(torch.Tensor([3]),requires_grad=True)
y=w*x+b
y.backward()
print x.grad
print w.grad
print b.grad
'''

##ji chu
'''
a=torch.LongTensor([[1,2],[3,4],[5,6],[7,8]])
print 'a',a
b=torch.IntTensor(6,3)
print 'b',b
c=torch.randn(6,3)
print 'c',c
d=torch.zeros((6,3))
print 'd',d
print torch.Tensor([3])
'''