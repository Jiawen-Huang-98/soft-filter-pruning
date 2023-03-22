# -*- encoding:utf-8 -*-
"""
@作者：Javen-Huang
@文件名：代码解析.py
@时间：2022/2/24  15:18
@文档说明:
"""
import math
import sys, os
import torch
from torch import sigmoid

import models

from torchvision import transforms
from pruning_train import validate
import torch.autograd as autograd
from torchvision import datasets

a = torch.rand(2,2,2,2)
print(a)
b = a.view(-1)
print(b)
x = b[:-1]
print(x)
y = b[:-2]
print(y)
# c = a.view(a.size()[1],-1)
# print(c)
# d = a.view(a.size()[0],-1)
# print(d)






# 使用线性退火方式
# epochs = 100
# decay_rate_init = 1
#
# for epoch in range(1, epochs+1):
#     # temp = temp*(1 - 2*epoch/epochs)
#     decay_rate = float('%.4f' % (decay_rate_init*(1 - epoch/epochs)))
#     print(decay_rate)



#  使用Sigmoid进行非线性退火
# epochs = 100
#
# temp_0 = 10
# decay_rate = 1
# for epoch in range(1,epochs+1):
#     temp = temp_0*(1 - 2*epoch/epochs)
#     temp_tensor = torch.tensor(temp)
#     # temp = temp_0*(1 - 2*epoch/epochs)
#     # m = sigmoid()
#     decay_rate = sigmoid(temp_tensor)
#     print('temp_tensor:',temp_tensor)
#     print('decay_rate:',decay_rate)



# a = torch.FloatTensor(10)
# print('a:',a)
# for i in range(10):
#
#     # a =a - torch.FloatTensor(0.1)
#     print(a)




# decay = 1
# test = 1
# for i in range(10):
#     decay = float('%.4f' % (decay - 0.1))
#     test = float('%.6f' % (decay))
#     print("decay:",decay)
#     print("test",test)
#     print(1-1)







# print(int(300/10))
# if 1 % 30 == 0 :
#     print(True)





# m = nn.Sigmoid()
# input = torch.randn(3)
# output = m(input)
# print(output)

# a = torch.randn(3)
# b = a.expand(2,3,4,3)
# c = torch.nn.Sigmoid()
# d = c(a)
# e = nn.Sigmoid(a)
# print(a)
# # print(b)
# print(d)
# print(e)

# net = models.__dict__['resnet18'](pretrained = False)
# # net.load_state_dict(r'D:\Xdown资源\Xdown下载任务\resnet18-5c106cde.pth')
# checkpoint = torch.load(r'D:\Xdown资源\Xdown下载任务\resnet18-5c106cde.pth')
# # net_1 = torch.load('D:\python_script\soft-filter-pruning-master\logs\cifar10_resnet110_norm2_0_324_3_rate0.7\checkpoint.pth')
# net.load_state_dict(checkpoint)
# normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
#                                  std = [0.229, 0.224, 0.225])
# valdir = 'D:\Xdown资源\Xdown下载任务\cifar-10-batches-py\data_batch_test'
#
# log = open(os.path.join('D:\Xdown资源\Xdown下载任务\cifar-10-batches-py', 'log_seed_{}.txt'.format(1998)), 'w')
# criterion = nn.CrossEntropyLoss().cuda()
# val_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(valdir, transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize,
#     ])),
#     batch_size = 128, shuffle = False,
#     num_workers = 0, pin_memory = True)
# validate(val_loader, net, criterion, log)
# print(net)
# # print(net_1)





# a = torch.ones(16, 8, 4, 2)
# b = torch.arange(16).resize(16, 1, 1, 1)
# # c = torch.matmul(a, b)
# c = a*b
#
# print('b:', b)
# print('c:', c)
# print('C:', c.size())



# a = autograd.Variable(torch.ones(4, 3, 12, 12))
# con_1 = nn.Conv2d(3, 16, 3, bias = False)
# s = con_1(a)
#
# filter_num = con_1.weight.size()[0]
#
#
# # decay = autograd.Variable(torch.randn(16, 1, 1, 1))
# # decay = autograd.Variable(torch.randn(con_1.parameters()[0].size()[0], 1, 1, 1))
# decay = autograd.Variable(torch.randn(filter_num, 1, 1, 1))
# print('decay.size():',decay.size())
# for item in con_1.parameters():
#     print(item.size())
#     print(item.size()[0])
#     item = item*decay
#     print(item)
# print(s)
# print(s.size())
# print(con_1.parameters())




#
# m = nn.ConvTranspose2d(16, 33, 3, stride=2)
#
# m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# input = autograd.Variable(torch.randn(20, 16, 50, 100))
# output = m(input)


# exact output size can be also specified as an argument
# input = autograd.Variable(torch.randn(1, 16, 12, 12))
# downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
#
# upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)  # 反卷积操作
# h = downsample(input)
#
# print(h.size())
#
# output = upsample(h, output_size=input.size())
# print(output.size())


# m = nn.Conv1d(16, 33, 3, stride=2)
# s = torch.randn(10,10)
# input = autograd.Variable(torch.randn(20, 16, 50))
# output = m(input)
# print(s)
# print(output)

# path = os.listdir(r'D:\python_script\soft-filter-pruning-master\代码测试')
# print(path)
# for i in path:
#     print(i)

# sys.path.append(r'D:\python_script\soft-filter-pruning-master\models')

import test
# import models
import numpy as np
import torch
# import Call_test
# Call_test.A(4)


# a = '--foo 1 --foo 2'.split()
# print(a)
#
# not torch.cuda.is_available()
# print("True")
#
#
# a = torch.ones(9)
# b = a.reshape(3, 3)
# print(a,b,'\n',b.size,'\n',b.size(),'\n')
# print(len(b.size()))

# a = torch.arange(9, dtype= torch.float) - 4
# b = a.reshape((3, 3))
# print(a, b,torch.norm(a))

# c = torch.arange(16, dtype= torch.float).reshape(2,2,2,2)
# d = torch.norm(c,2,1)
# f = d.numpy()
# g = f.argsort()
# print(c,'\n', d, '\n',c.shape, '\n',d.shape, '\n',f, '\n',g)

#
# c = torch.arange(8, dtype= torch.float).reshape(2,2,2)
# d = torch.norm(c,2,1)
# f = d.numpy()
# g = f.argsort()
# print(c,'\n', d, '\n',c.shape, '\n',d.shape, '\n',f, '\n',g)
#
#
# net = models.__dict__['resnet50'](10)  # 设置网络模型


# m = nn.Linear(20, 30)
# input = torch.randn(128, 20)
# output = m(input)
# print(output.size())
# torch.Size([128, 30])


# print(torch.cuda.is_available())