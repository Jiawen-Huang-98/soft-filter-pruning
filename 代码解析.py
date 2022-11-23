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



x = torch.arange(9).reshape(3,3)
# x = torch.arange(9).view(3,3)

y = x.mul(0)
print(x)
print(y)
print(torch.cat((x, x.mul(0)), 1))



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




