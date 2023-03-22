from __future__ import division

import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import models
import numpy as np
# import notifyemail as notify

# notify.Reboost(mail_host = 'smtp.163.com', mail_user = 'a429296965@163.com', mail_pass = 'HYYXLSBEYBJUJRCG', default_reciving_list = ['a429296965@163.com'],
#                log_root_path = 'log')
# notify.add_text('初次邮件测试')
# notify.send_log()
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data_path', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
# compress rate
parser.add_argument('--rate', type=float, default=0.9, help='compress rate of model')
parser.add_argument('--layer_begin', type=int, default=1,  help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=1,  help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1,  help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1,  help='compress layer of model')
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dict or not')
parser.add_argument('--decay_rate',type = float,default = 0.5,help = 'the pruned filter decay rate')

args = parser.parse_args(['./data/cifar.python', '--dataset', 'cifar10', '--arch', 'resnet20',
                         '--save_path', './logs/cifar10_resnet20_modeltest_{}'.format(time.strftime('%m-%d %H:%M')),
                         '--batch_size', '256','--evaluate',
                          # '--use_state_dict',
                          '--resume','logs/cifar10_resnet20_modeltest/res_20_checkpoint.pth.tar'])
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

# 设置随机种子
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

def main():
    # Init logger  初始化日志

    # log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    state = {k: v for k, v in args._get_kwargs()}  # 获取args中的关键字参数（关键字参数在函数中是字典类型的）
    print(state)
    print("=> creating model '{}'".format(args.arch))

    print("=> parameter : {}".format(args))
    print("Random Seed: {}".format(args.manualSeed))
    print("python version : {}".format(sys.version.replace('\n', ' ')))
    print("torch  version : {}".format(torch.__version__))
    print("cudnn  version : {}".format(torch.backends.cudnn.version()))
    print("Compress Rate: {}".format(args.rate))
    print("Layer Begin: {}".format(args.layer_begin))
    print("Layer End: {}".format(args.layer_end))
    print("Layer Inter: {}".format(args.layer_inter))
    print("Epoch prune: {}".format(args.epoch_prune))
    # Init dataset 初始化数据集
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)  # assert后为真，则继续向下进行，否则输出后面的报错信息
# 训练（测试）的数据转换操作，进行串联
    train_transform = transforms.Compose(  # 将多个转换器进行串联
        [transforms.RandomHorizontalFlip(),  # 随机水平翻转给定的图片，概率为0.5
         transforms.RandomCrop(32, padding=4),  # 随机选择裁剪的中心点，padding增加的宽度
         transforms.ToTensor(),  # 数据格式转换为tensor
         transforms.Normalize(mean, std)])  # 通过给定的均值方差，将tensor进行正则化
    test_transform = transforms.Compose(
        [transforms.ToTensor(),  # 数据格式转换成tensor
         transforms.Normalize(mean, std)])  # 同正则化

    if args.dataset == 'cifar10':
        # root=cifar-10-batches-py的根目录，train=是训练集，transform=数据的转换操作，download=从网络下载数据，并进行数据初始化操作
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=False)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=False)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.SVHN(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.STL10(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'  # assert后为真，则继续向下进行，否则输出后面的报错信息
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.workers, pin_memory=True)

    print("=> creating model '{}'".format(args.arch))
    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)  # 设置（加载）网络模型
    print("=> network :\n {}".format(net))

    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))  # GPU并行计算



    # define loss function (criterion标准) and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # 损失函数为交叉熵损失函数

    # optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
    #                             weight_decay=state['decay'], nesterov=True)  # 优化器

    if args.use_cuda:  #将数据放置在GPU中
        net.cuda()
        criterion.cuda()

    # recorder = RecorderMeter(args.epochs)  # 计算并记录最小损失函数和它的epoch,先定义成该类，后面进行更新
    # optionally resume from a checkpoint  从断点处继续训练模型（选择性的）
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)  # 从磁盘中读取文件
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']  # 将新的断点处作为新的epoch进行训练
            # if isinstance(net,torch.nn.DataParallel):
            #     net = net.module
            #  model.load_state_dict(torch.load(model_path,map_location=lambda storage, loc: storage).module.state_dict())
            if args.use_state_dict:
                net.module.load_state_dict(checkpoint['state_dict'])  # 将state_dict中的parameters和buffers复制到它的后代中(重载)
            else:
                net = checkpoint['state_dict']
                
            # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载optimizer的状态（重载）
            print("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print("=> do not use any checkpoint for {} model".format(args.arch))

    if args.evaluate:  # 在验证集上进行计算
        time1 = time.time()
        validate(test_loader, net, criterion)
        time2 = time.time()
        print('function took %0.3f ms' % ((time2-time1)*1000.0))
        return



def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode 模型切换到验证模式，该模式下没有dropout
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda(non_blocking = True)
            input = input.cuda()
        with torch.no_grad():  # 在该语句下，所有计算得出的tensor的require_grad设置为False
            input_var = torch.autograd.Variable(input)
        with torch.no_grad():
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    print('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg))

    return top1.avg, losses.avg

# def print(print_string):
#     print("{}".format(print_string))
#     log.write('{}\n'.format(print_string))
#     log.flush()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # 对output的第二哥维度进行排序，筛选出前maxk个值，并返回它们的序号（pred）
    pred = pred.t()  # 转置
    # eq返回两个对象对应元素是否相等，expand_as表示将target扩展成与pred相同的维度
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # view将目标拉成一维向量

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)  #sum(0)表示对第一维度进行相加
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
