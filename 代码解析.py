# -*- encoding:utf-8 -*-
"""
@作者：Javen-Huang
@文件名：代码解析.py
@时间：2022/2/24  15:18
@文档说明:
"""
import sys
# sys.path.append(r'D:\python_script\soft-filter-pruning-master')
import models
import numpy as np
import torch
import time,datetime
from  utils import AverageMeter,convert_secs2time,time_file_str

import math
import sys, os
import torch
from torch import sigmoid
from utils import AverageMeter
import models

from torchvision import transforms
from pruning_train import validate
import torch.autograd as autograd
from torchvision import datasets as dset

import torch
import torchvision.models as models

# 加载ResNet20模型
model = models.resnet18(pretrained = False)

# 输出模型结构
print(model)

# 将模型转移到GPU上
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)

# 定义损失函数和优化器
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

# 训练模型
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#
#         # 前向传播
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # 反向传播和优化器更新
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # 打印损失和准确率
#         if (i + 1) % 100 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 保存模型
# torch.save(model.state_dict(), 'resnet20.ckpt')


# def main():
#     sys.path.append('./')
#     global teacher_model
#     teacher_model = models.__dict__['resnet20'](10)  # 设置（加载）网络模型
#     teacher_model = torch.nn.DataParallel(teacher_model, device_ids = list(range(1)))  # GPU并行计算
#     check_point = torch.load('pre_model/res_20_checkpoint.pth.tar', map_location = torch.device('cpu'))
#     teacher_model.load_state_dict(check_point['state_dict'])
#     teacher_model.eval()
#     data_path = './data/cifar.python'
#     mean = [x / 255 for x in [125.3, 123.0, 113.9]]
#     std = [x / 255 for x in [63.0, 62.1, 66.7]]
#     test_transform = transforms.Compose(
#             [transforms.ToTensor(),  # 数据格式转换成tensor
#              transforms.Normalize(mean, std)])  # 同正则化
#         # root=cifar-10-batches-py的根目录，train=是训练集，transform=数据的转换操作，download=从网络下载数据，并进行数据初始化操作
#     # train_data = dset.CIFAR10(data_path, train = True, transform = train_transform, download = True)
#     test_data = dset.CIFAR10(data_path, train = False, transform = test_transform, download = True)
#     num_classes = 10
#     dataloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False,
#                                                     num_workers=4, pin_memory=True)
#     criterion = torch.nn.CrossEntropyLoss()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#
#
#     def accuracy(output, target, topk=(1,)):
#         """Computes the precision@k for the specified values of k"""
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)  # 对output的第二哥维度进行排序，筛选出前maxk个值，并返回它们的序号（pred）
#         pred = pred.t()  # 转置
#         # eq返回两个对象对应元素是否相等，expand_as表示将target扩展成与pred相同的维度
#         correct = pred.eq(target.view(1, -1).expand_as(pred))  # view将目标拉成一维向量
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0)  #sum(0)表示对第一维度进行相加
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res
#
#     for i, (input, target) in enumerate(dataloader):
#         with torch.no_grad():
#             input_var = torch.autograd.Variable(input)
#         with torch.no_grad():
#             target_var = torch.autograd.Variable(target)
#
#         # compute output
#         output = teacher_model(input_var)
#         loss = criterion(output, target_var)
#         prec1, prec5 = accuracy(output.data, target, topk = (1, 5))
#         losses.update(loss.item(), input.size(0))
#         top1.update(prec1.item(), input.size(0))
#         top5.update(prec5.item(), input.size(0))
#
#     print('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg))
#
#     def all_zero_nums(self):
#         non_zero = 0
#         zero = 0
#         for index, item in enumerate(self.parameters()):
#             if index in self.mask_index:
#                 # if (index == 0):
#                 a = item.data.view(self.model_length[index])
#                 b = a.cpu().numpy()
#                 non_zero += np.count_nonzero(b)
#                 zero += len(b)-np.count_nonzero(b)
#             if index == max(self.mask_index):
#                 print("number of nonzero weight is %d, zero is %d" % (non_zero, zero))
#     all_zero_nums(teacher_model)
# if __name__ == '__main__':
#     main()



# x = torch.arange(9).reshape(3,3)
# # x = torch.arange(9).view(3,3)
#
# y = x.mul(0)
# print(x)
# print(y)
# print(torch.cat((x, x.mul(0)), 1))



# def time_string():
#   ISOTIMEFORMAT='%Y-%m-%d %X'
#   string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time() + 8*60*60)))
#   return string
#
# now = time_file_str()
# # start_time = str(time.time())
# start_time = time.time()
# epoch_time = AverageMeter()
# a = np.arange(400).reshape(20,20)
# b = np.arange(40,440).reshape(20,20)
# c = a-b
# time.sleep(2)
# epoch_time.update(time.time()-8)
# # need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
# time.sleep(2)
# print("now:",now)
# print(start_time)
# print(epoch_time.val)
# print(time.time())
# print(time.time()-8)


# time_1 = time.time()
# time_2 = datetime.date.today()
# # time_2 = time_2.today()
# hour = int(time_1/3600)
# min
# print(time_1,hour)
# print(time_2,time_string())

# print(max([1,2]))
# print(0%1==0)


# print(np.arange(8).reshape(4,-1))
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


# c = torch.arange(8, dtype= torch.float).reshape(2,2,2)
# d = torch.norm(c,2,1)
# f = d.numpy()
# g = f.argsort()
# print(c,'\n', d, '\n',c.shape, '\n',d.shape, '\n',f, '\n',g)


# net = models.__dict__['resnet50'](10)  # 设置网络模型
# # print(models.__dict__)
#
#
# class people:
#     def __init__(self):
#         self.name = "Tom"
#         self.age = 23
# obj = people()
# print(people.__dict__,'\n',obj.__dict__)




