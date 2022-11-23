# -*- encoding:utf-8 -*-
"""
@作者：Javen-Huang
@文件名：重载模型.py
@时间：2022/5/10  16:31
@文档说明:
"""
import models
import torch
from pruning_train import validate

net = models.__dict__['resnet18'](pretrained = False)
# net.load_state_dict(r'D:\Xdown资源\Xdown下载任务\resnet18-5c106cde.pth')
checkpoint = torch.load(r'D:\Xdown资源\Xdown下载任务\resnet18-5c106cde.pth')
# net_1 = torch.load('D:\python_script\soft-filter-pruning-master\logs\cifar10_resnet110_norm2_0_324_3_rate0.7\checkpoint.pth')
net.load_state_dict(checkpoint)


print(net)
# print(net_1)