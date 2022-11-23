# -*- encoding:utf-8 -*-
"""
@作者：Javen-Huang
@文件名：添加网络层.py
@时间：2022/5/9  19:57
@文档说明:
"""


import torch
import time

##使用pytorch实现之前使用numpy实现的两层神经网络

#此部分用于检测该程序运行多长时间
start=time.time()


device=torch.device("cpu")
#选择是使用cpu进行计算还是gpu
#device=torch.device("cuda")


#神经网络参数定义：
N,D_in,D_out,H=64,1000,10,100
# N：代表64个人
# D_in:代表每个人输入到神经网络1000个数据点
# D_out:代表每个人从神经网络输出10个数据点
# H:代表该神经网络含有100个隐藏层

x=torch.randn(N,D_in,device=device)
#定义输入到神经网络之前的数据矩阵，大小为64*1000
#这里需要注意的是我们需要指定device为我们刚刚设定好的device
y=torch.randn(N,D_out,device=device)
#定义从神经网络输出的的数据矩阵，大小为64*10
w_1=torch.randn(D_in,H,device=device)
#大小为1000*100
w_2=torch.randn(H,D_out,device=device)
#大小为100*10

learning_rate=1e-6

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
    print(it,loss)

    #计算梯度，主要是对(y_hat-y)^2求各项偏导
    y_hat_grad=2*(y_hat-y)
    w_2_grad=h_relu.t().mm(y_hat_grad)
    h_relu_grad=y_hat_grad.mm(w_2.t())
    h_grad=h_relu_grad.clone()
    h_grad[h<0]=0
    w_1_grad=x.t().mm(h_grad)

    #更新w_1和w_2的权值
    w_1=w_1-learning_rate*w_1_grad
    w_2=w_2-learning_rate*w_2_grad
end=time.time()
print('Running time: %s Seconds'%(end-start))
#输出此程序使用时间
