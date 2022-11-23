# -*- encoding:utf-8 -*-
"""
@作者：Javen-Huang
@文件名：添加忘网络层1.py
@时间：2022/5/9  19:59
@文档说明:
"""
# @Time : 2020/5/9 16:40
# @Author : kingback
# @File : pytorch_practice02.py
# @Software: PyCharm

import torch
import time

##使用pytorch实现两层神经网络，
## 本程序采用自动求偏导，不再人为求，相对来说简单一些

#此部分用于检测程序运行了多长时间
start=time.time()


#选择是使用cpu进行计算还是gpu
device=torch.device("cpu")
# device=torch.device("cuda")

dtype=torch.float
#定义一下数据类型

#神经网络参数定义：
N,D_in,D_out,H=64,1000,10,100
# N：代表64个人
# D_in:代表每个人输入到神经网络1000个数据点
# D_out:代表每个人从神经网络输出10个数据点
# H:代表该神经网络含有100个隐藏层

x=torch.randn(N,D_in,device=device,dtype=dtype)
#定义输入到神经网络之前的数据矩阵，大小为64*1000
#这里需要注意的是我们需要指定device为我们刚刚设定好的device
y=torch.randn(N,D_out,device=device,dtype=dtype)
#定义从神经网络输出的的数据矩阵，大小为64*10
w_1=torch.randn(D_in,H,device=device,dtype=dtype,requires_grad=True)
#大小为1000*100
w_2=torch.randn(H,D_out,device=device,dtype=dtype,requires_grad=True)
#大小为100*10,在这里需要声明一下w1和w2两个参数需要梯度

learning_rate=1e-6
#定义学习率

for it in range(500):
    #forword pass
    h=x.mm(w_1)
    #numpy中的点乘np.dot是数学意义上的向量内积
    #print(h.shape)
    #打印矩阵维度信息
    h_relu=h.clamp(min=0)
    #定义relu 函数，在pytorch中使用.clamp方法
    y_hat=h_relu.mm(w_2)
    #大小为64*10

    #计算损失compute loss
    loss=(y_hat-y).pow(2).sum()
    #估计值与真实值之间差值的平方和再取和,替换numpy的square方法
    print(it,loss.item())
    #此时的loss是只有一个数的tensor，我们可以使用.item()将其值转为数字

    loss.backward()
    # PyTorch给我们提供了autograd的方法做反向传播。如果一个Tensor的requires_grad=True，
    # backward会自动计算loss相对于每个Tensor的gradient。在backward之后，
    # w_1.grad和w_2.grad会包含两个loss相对于两个Tensor的gradient信息。

    with torch.no_grad():
        w_1-=learning_rate*w_1.grad
        w_2-=learning_rate*w_2.grad

        #梯度清零，否则会出现累加效应
        w_1.grad.zero_()
        w_2.grad.zero_()
end = time.time()
print('Running time: %s Seconds' % (end - start))
