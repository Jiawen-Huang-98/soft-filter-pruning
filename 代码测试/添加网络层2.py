# -*- encoding:utf-8 -*-
"""
@作者：Javen-Huang
@文件名：添加网络层2.py
@时间：2022/5/9  20:00
@文档说明:
"""
# @Time : 2020/5/9 23:00
# @Author : kingback
# @File : pytorch_practice03.py
# @Software: PyCharm

#PyTorch autograd使定义计算图形和计算梯度变得很容易，但是对于定义复杂的神经网络来说，
# 原始autograd可能太低级了;这就是nn包可以提供帮助的地方。nn包定义了一组模块，
# 您可以将其视为一个神经网络层，它从输入生成输出，并且可能具有一些可训练的权重。

import torch

#神经网络参数定义：
N,D_in,D_out,H=64,1000,10,100
# N：代表64个人
# D_in:代表每个人输入到神经网络1000个数据点
# D_out:代表每个人从神经网络输出10个数据点
# H:代表该神经网络含有100个隐藏层

device=torch.device("cpu")
dtype=torch.float

x=torch.randn(N,D_in,device=device,dtype=dtype)
#定义输入到神经网络之前的数据矩阵，大小为64*1000,这里需要注意的是我们需要指定device为我们刚刚设定好的device
y=torch.randn(N,D_out,device=device,dtype=dtype)
#定义从神经网络输出的的数据矩阵，大小为64*10

#使用nn包将我们的模型定义为一个层序列。神经网络。Sequential是一个包含其他模块的模块，
# 并按顺序应用它们来产生它的输出。每个线性模块使用一个线性函数从输入计算输出，并保存内部张量的权重和偏差。

model=torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out),
)

# nn包还包含流行的损失函数的定义;在这种情况下，我们将使用均方误差(MSE)作为损失函数。
loss_fn=torch.nn.MSELoss(reduction="sum")

learning_rate=1e-04

for it in range(500):
    #前向传递: 通过向模型传递x来计算预测y。模块对象覆盖了操作符，所以你可以像调用函数一样调用它们。当你这样做的时候，
    # 你把一个输入数据的张量传递给模块，它就会产生一个输出数据的张量。
    y_hat=model(x)

    #计算和打印损失。我们传递包含y的预测值和真值的张量，损失函数返回一个包含损失的张量。
    loss=loss_fn(y_hat,y)
    if(it%10==9):
        print(it,loss.item())

    #在进行反向传播之前将梯度置为0
    model.zero_grad()

    #反向传播: 根据模型的所有可学习参数计算损失的梯度。在内部，每个模块的参数都存储在带有requires_grad = True的张量中，
    # 因此这个调用将为模型中的所有可学习参数计算梯度。
    loss.backward()

    # 使用梯度下降更新权重。每个参数都是一个张量，所以我们可以像以前一样得到它的梯度。
    with torch.no_grad():
        for param in model.parameters():
            param-=learning_rate*param.grad
