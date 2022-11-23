#  https://github.com/pytorch/vision/blob/master/torchvision/models/__init__.py
import argparse
import os, sys
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
from utils import convert_secs2time, time_string, time_file_str
# from models import print_log
import models   #该模块为自己写的模块，指models文件夹
import random
import numpy as np

model_names = sorted(name for name in models.__dict__   # __dict__是一个字典，其包含了该对象上的全部属性，寻找符合相应条件的字典条目，此处为寻找各个模型
    if name.islower() and not name.startswith("__")     # islower（）函数验证字符串是否有大写字母，如果是会返回True，startswith()字符串是否以什么什么开头
    and callable(models.__dict__[name]))    # callable用于检查一个对象是否能够被调用，如果返回True,对象仍有可能调用失败，如果返回False，那么必定调用失败。

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')   # ArgumengtParser意为参数解析器，该对象将命令行解析成Python所需的全部信息，参数descripition表示帮助文档之前的信息，可以认为是注释
parser.add_argument('data', metavar='DIR',  # add_argument用于添加程序的参数信息，该调用通常指定将获取的字符串如何转换为对象,此处的data为所添加参数的命名或命名列表，metavar表示在使用方法消息中使用的参数值示例
                    help='path to dataset')     # help表示此选项作用的简单描述
parser.add_argument('--save_dir', type=str, default='./', help='Folder to save checkpoints and log.')   # type表示命令行参数应该被转换成的类型，default表示当参数未在命令行出现时使用的值
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,    # choice表示可用的参数的容器（集合）,此处表示存储模型类型（结构）
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')  # dest表示所添加的属性，action命令指定了该如何处理这些参数

args = parser.parse_args(['DIR', r'D:\360downloads'])  # parse_args将参数字符串转换为对象并将其设为命令空间的属性，返回带有成员的命名空间
args.use_cuda = torch.cuda.is_available()   # 检验cuda是否能够被调用

args.prefix = time_file_str()   # 返回字符串格式的时间


def main():
    best_prec1 = 0  # 最高精度重置为0

    if not os.path.isdir(args.save_dir):    # 如果对象是目录，返回True，如果是文件，则返回False
        os.makedirs(args.save_dir)    # 递归目录创建功能与mkdir()相似，但会自动创建最后一级目录所需要的中间目录
    log = open(os.path.join(args.save_dir, '{}.{}.log'.format(args.arch,args.prefix)), 'w')     # 打开目录并以模型名和当前时间进行命名

    #  version information  查看各个模块信息
    print_log("PyThon  version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch version : {}".format(torch.__version__), log)
    print_log("cuDNN   version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Vision  version : {}".format(torchvision.__version__), log)
    #  create model  构建模型
    print_log("=> creating model '{}'".format(args.arch), log)
    model = models.__dict__[args.arch](pretrained=False)    #引入单个模型
    print_log("=> Model : {}".format(model), log)
    print_log("=> parameter : {}".format(args), log)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
      model.features = torch.nn.DataParallel(model.features)    # GPU分布式计算（平行计算）
      model.cuda()  # .cuda()表示将模型移动到GPU上
    else:
      model = torch.nn.DataParallel(model).cuda()

    #  define loss function (criterion) and optimizer   定义交叉熵损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()    # 用于计算损失函数CrossEntripyLOss(x,y),x为真实值，y为预测值

    optimizer = torch.optim.SGD(model.parameters(), args.lr,    # SGD就是optim中的一个算法（优化器）：随机梯度下降算法，lr学习率，momentum动量因子，weigth_decay权重衰减
                                momentum=args.momentum,     # dampening动量的抑制因子，nesterov使用Nesterov动量
                                weight_decay=args.weight_decay,
                                nesterov=True)

    #  optionally resume from a checkpoint   可选择从检查点进行恢复
    if args.resume:     # resume表示最新的检查点的路径
        if os.path.isfile(args.resume):     # 检查args.resume是否是文件类型？
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)    # 使用python的解封设施，意为载入
            args.start_epoch = checkpoint['epoch']  # 赋值
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])     # load_state_dict将预训练的参数权重加载到新的模型中
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)

    cudnn.benchmark = True  # 该设置可以自动寻找最适合当前配置的高效算法，以达到优化运行效率的问题（前提是网络输入的数据维度或类型变化不大）

    #  Data loading code
    traindir = os.path.join(args.data, 'train')     # 连接两个或更多的路径名组件，同时若组件名不包含‘/’，则该函数会自动加上，此处的traindir表示训练集的存储路径
    valdir = os.path.join(args.data, 'val')     # 同上，valdir表示目标值的存储路径
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],    # 正则化参数
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(   # 图片加载器，第一个参数设置是图片（数据集）的根目录
        traindir,
        transforms.Compose([    # compose（）函数将多个图片的操作变换进行串联
            transforms.RandomResizedCrop(224),      # 随机采集一下图像，然后将他们裁剪成统一的大小
            transforms.RandomHorizontalFlip(),      # 将图片依概率P随机垂直翻转（此处的P待输入，默认为50%）
            transforms.ToTensor(),      # 将数据格式转换为tensor
            normalize,      # 将数据进行正则化
        ]))

    train_loader = torch.utils.data.DataLoader(     # DataLoader数据加载器。 结合数据集和采样器，并提供给定数据集的可迭代对象
        train_dataset, batch_size=args.batch_size, shuffle=True,        # batch_size为每次输入数据的行数，shuffle为True时，表示将数据进行打乱
       #  num_workers=args.workers, pin_memory=True, sampler=None)将num_workers置为0，pin_memory为“True“时数据加载器将在返回张量之前复制到CUDA固定内存中
        num_workers = 0, pin_memory = True, sampler = None)     # num_workers表示使用多少个子进程来导入数据

    val_loader = torch.utils.data.DataLoader(   # 此处的数据加载器DataLoader中调用的数据集是一个ImageFolder
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        # num_workers=args.workers, pin_memory=True)将num_workers置为0
        num_workers = 0, pin_memory = True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    filename = os.path.join(args.save_dir, 'checkpoint.{}.{}.pth.tar'.format(args.arch, args.prefix))   # 存储文件名及其路径
    bestname = os.path.join(args.save_dir, 'best.{}.{}.pth.tar'.format(args.arch, args.prefix))     # 存储最好的模型结构并记录

    start_time = time.time()    # 记录开始时间
    epoch_time = AverageMeter()     # 自定义函数：计算并存储平均值和当前值
    for epoch in range(args.start_epoch, args.epochs):  # 从起始时间步进入循环，每个时间步进行的操作如下
        adjust_learning_rate(optimizer, epoch)      # 调整学习率，根据优化器及时间步

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.val * (args.epochs-epoch))       # 通过epoch_time得到目前所花费的时间多少
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)      # 将时，分，秒进行串接
        print_log(' [{:s}] :: {:3d}/{:3d} ----- [{:s}] {:s}'.format(args.arch, epoch, args.epochs, time_string(), need_time), log)

        #  train for one epoch  开始训练一个时间步train（）为自定义函数
        train(train_loader, model, criterion, optimizer, epoch, log)
        #  evaluate on validation set  评估验证集得到正确率--自定义函数
        val_acc_2 = validate(val_loader, model, criterion, log)

        #  remember best prec@1 and save checkpoint   记住最好的 prec@1 并保存检查点
        is_best = val_acc_2 > best_prec1        # 判断是否需要更新准确率
        best_prec1 = max(val_acc_2, best_prec1)
        save_checkpoint({       # 存储检查点信息
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename, bestname)
        #  measure elapsed time   测试经过的时间
        epoch_time.update(time.time() - start_time) # 计算时间差
        start_time = time.time()    # 更新时间信息
    log.close()


def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    #  switch to train mode  切换到训练模式，该模式下会有dropout，即随机切断神经元的连接（50%）
    model.train()

    end = time.time()   # 初始化时间，记录结束时的时间
    for i, (input, target) in enumerate(train_loader):  # 函数enumerate()输入一个序列，返回一个二元数组，该数组包含一个计数值和序列相对位置的值，i为计数值，input
                                                        # 和target分别为输入与输出，输入为一张图片，输出为一个类别的编号
        #  measure data loading time 测试数据加载时间
        data_time.update(time.time() - end)

        # target = target.cuda(async=True)  # 将async修改成non_blocking，将数据载入GPU
        target = target.cuda(non_blocking = True)
        input_var = torch.autograd.Variable(input)  # 将input传播的所有中间值进行记录，下同
        target_var = torch.autograd.Variable(target)

        #  compute output   计算输出
        output = model(input_var)
        loss = criterion(output, target_var)

        #  measure accuracy and record loss 测试正确率和记录损失
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))   #分别计算top-1和top-5的准确率
        losses.update(loss.data[0], input.size(0))  # 更新损失函数及相应的一些指标
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        #  compute gradient and do SGD step  计算梯度同时进行一步梯度下降算法
        optimizer.zero_grad()   #清空梯度，计算损失并返回值，
        loss.backward()     # loss的反向传播
        optimizer.step()    #更新优化器中的所有参数

        #  measure elapsed time 测试经过的时间，此处的end是上次batch的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5), log)


# 验证过程，该函数与训练过程的函数结构类似
def validate(val_loader, model, criterion, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    #  switch to evaluate mode  将模型设置为验证
    model.eval()    #将模型设置为验证模式

    end = time.time()   #重新记录当前时间
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)  将async修改成non_blocking
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input, volatile=True)# autograd只支持标量的求导，此处的volatile是一个布尔值，指示是否用于推断模式（即不保存历史信息）
        target_var = torch.autograd.Variable(target, volatile=True)

        #  compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        #  measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        #  measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5), log)

    print_log(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

    return top1.avg


def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)# 高阶文件操作，将文件filename,复制到名为bestname的文件位置，位置同为dist.

def print_log(print_string, log):   #打印string内容,log文件内容
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()     # flush()将缓冲区的数据进行强行输出
  
  
class AverageMeter(object):
    """Computes and stores the average and current value  用于计算和存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):# 将模型的学习率进行调整，每30个时间步下降至当前的10%
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t() #.t()将pred转置
    correct = pred.eq(target.view(1, -1).expand_as(pred))   #eq()比较元素的相等性,此处的view()同reshape，其中的-1，表明对out进行变形，列数由行数决定,expend_as()将张量扩展成与pred相同的格式

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':  # __name__是系统内置的变量，代表所在的模块名字，若在模块内直接运行时，__name__为__main__,当在其他模块内运行时，__name__为其文件名
    main()
