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

args = parser.parse_args(['../datasets/CIFAR10/cifar.python', '--dataset', 'cifar10', '--arch', 'resnet20',
                         '--save_path', './logs/cifar10_resnet110_norm2_0_324_3_rate0.7', '--epochs', '200', '--schedule', '1', '60', '120', '160',
                         '--gammas', '10', '0.2', '0.2', '0.2', '--learning_rate', '0.01', '--decay', '0.0005', '--batch_size', '16', '--rate', '0.7', '--layer_begin', '0',
                         '--layer_end', '324', '--layer_inter', '3', '--epoch_prune', '1','--decay_rate', '0.7'])
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
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}  # 获取args中的关键字参数（关键字参数在函数中是字典类型的）
    print_log(state, log)
    print_log("=> creating model '{}'".format(args.arch), log)

    print_log("=> parameter : {}".format(args), log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Compress Rate: {}".format(args.rate), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.workers, pin_memory=True)  # 数据加载器
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.workers, pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)  # 设置（加载）网络模型
    print_log("=> network :\n {}".format(net), log)

    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))  # GPU并行计算



    # define loss function (criterion标准) and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # 损失函数为交叉熵损失函数

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)  # 优化器

    if args.use_cuda:  #将数据放置在GPU中
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)  # 计算并记录最小损失函数和它的epoch,先定义成该类，后面进行更新
    # optionally resume from a checkpoint  从断点处继续训练模型（选择性的）
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)  # 从磁盘中读取文件
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']  # 将新的断点处作为新的epoch进行训练
            if args.use_state_dict:
                net.load_state_dict(checkpoint['state_dict'])  # 将state_dict中的parameters和buffers复制到它的后代中(重载)
            else:
                net = checkpoint['state_dict']
                
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载optimizer的状态（重载）
            print_log("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:  # 在验证集上进行计算
        time1 = time.time()
        validate(test_loader, net, criterion, log)
        time2 = time.time()
        print ('function took %0.3f ms' % ((time2-time1)*1000.0))
        return

    m = Mask(net, args.decay_rate)

    m.init_length()
    
    comp_rate = args.rate
    print("-"*10+"one epoch begin"+"-"*10)
    print("the compression rate now is %f" % comp_rate)

    val_acc_1, val_los_1 = validate(test_loader, net, criterion, log)

    print(" accu before is: %.3f %%" % val_acc_1)
    
    m.model = net
    
    m.init_mask(comp_rate)
#    m.if_zero()
    m.do_mask()
    # 更新模型
    net = m.model
#    m.if_zero()
    if args.use_cuda:
        net = net.cuda()
    val_acc_2, val_los_2 = validate(test_loader, net, criterion, log)
    print(" accu after is: %s %%" % val_acc_2)
    

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    decay_rate_init =1
    for epoch in range(args.start_epoch, args.epochs):

        # 使用线性退火方式
        m.decay_rate = float('%.4f' % (decay_rate_init * (1 - epoch / args.epochs)))
        print_log('the decay_rate now is :{}'.format(m.decay_rate),log)

        # gammas学习率的衰减系数，schedule是epoch列表在这些epoch进行学习率的衰减
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [each epoch(s):{:03.2f}] [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, epoch_time.val,current_learning_rate)
                                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)),
                  log)

        # train for one epoch
        train_acc, train_los = train(train_loader, net, criterion, optimizer, 0, log)

        # evaluate on validation set
        # val_acc_1, val_los_1 = validate(test_loader, net, criterion, log)
        # epoch_prune表示每多少个epoch进行prune
        if epoch % args.epoch_prune == 0 or epoch == args.epochs-1:
            # 继续更新模型
            m.model = net
            # 判断
            m.if_zero()
            m.all_zero_nums()
            m.init_mask(comp_rate)
            m.do_mask()
            m.if_zero()
            m.all_zero_nums()
            net = m.model
            if args.use_cuda:
                net = net.cuda()  
            
        val_acc_2, val_los_2 = validate(test_loader, net, criterion, log)
    
        
        is_best = recorder.update(epoch, train_los, train_acc, val_los_2, val_acc_2)
        # 保存检查点
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net,
            'recorder': recorder,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save_path, 'res_20_checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        #recorder.plot_curve( os.path.join(args.save_path, 'curve.png') )

    log.close()

# train function (forward向前传播, backward向后传播, update更新权重)
def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train model
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(non_blocking = True)
            input = input.cuda()
        # 包装一个tensor，记录input的变量,Variable会存储input的所有变化的中间值(通过调试发现并没有记录中间的量)，方便反向传播时梯度的更新
        # 在输入model后，input_var与input仍然相同
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)  #此处的损失函数为交叉熵函数

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()  # 梯度反向传播
        optimizer.step()  # 将参数进行一次更新

        # measure elapsed（流逝） time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
    return top1.avg, losses.avg

def validate(val_loader, model, criterion, log):
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

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

    return top1.avg, losses.avg

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

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


class Mask:
    def __init__(self, model,decay_rate):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.decay_rate = decay_rate
    # 权重剪枝
    def get_codebook(self, weight_torch,compress_rate,length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()
    
        weight_abs = np.abs(weight_np)  # 取权重的绝对值
        weight_sort = np.sort(weight_abs)  # 对权重进行排序
        
        threshold = weight_sort[int(length * (1-compress_rate))]  # 剪枝的权重临界值
        # 进行权重剪枝
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np !=1] = 0
        
        print("codebook done")
        return weight_np    # 返回剪枝二进制向量

    # 卷积核剪枝,此处的length是该层卷积核（a,b,c,d）的参数总量a*b*c*d
    def get_filter_codebook(self, weight_torch, compress_rate, length, decay_rate):
        codebook = np.ones(length)
        # 判断是否为卷积层
        if len(weight_torch.size()) == 4:  # 权重值为四维向量（a,b,c,d），a为输入通道数，b为输出通道数卷积，b=c为卷积核大小，（待定）
            filter_pruned_num = int(weight_torch.size()[0]*(1-compress_rate))  # 计算被剪枝的数量
            weight_vec = weight_torch.view(weight_torch.size()[0],-1)  # 维度变换为（a,bcd）
            norm2 = torch.norm(weight_vec,2,1)  # 此处P=2,表示计算的是2范数，dim=1表示进行压缩的是第二维度，即对每个卷积核进行计算其结果的维度是（a）
            norm2_np = norm2.cpu().numpy()  # 转换成numpy格式
            filter_index = norm2_np.argsort()[:filter_pruned_num]  # argsort返回数组值从小到达的索引值,此处求的是L2范数较小的前pruned_num个
#            norm1_sort = np.sort(norm1_np)
#            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] *weight_torch.size()[2] *weight_torch.size()[3]  # 计算核的长度，即卷积核的参数b*c*d（待定）
            # 创建密码本（用于决定是否将值置为0），寻找剪枝的卷积核参数，将其密码本位置置为0
            for x in range(0,len(filter_index)):
                codebook[filter_index[x] *kernel_length : (filter_index[x]+1) *kernel_length] *= decay_rate

            # print("filter codebook done")
        else:
            pass
        return codebook

    # 将x转换为tensor的float类型
    def convert2tensor(self,x):
        x = torch.FloatTensor(x)
        return x
    
    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size [index] = item.size()
        
        for index1 in self.model_size:
            for index2 in range(0,len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]
                    
    def init_rate(self, layer_rate):
        for index, item in enumerate(self.model.parameters()):  # 初始化所有层的压缩率为1
            self.compress_rate[index] = 1
        for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):  # 将选定层（也就是卷积层）的压缩率设置为选定值
            self.compress_rate[key] = layer_rate
        # different setting for different architecture
        if args.arch == 'resnet20':
            last_index = 57
        elif args.arch == 'resnet32':
            last_index = 93
        elif args.arch == 'resnet56':
            last_index = 165
        elif args.arch == 'resnet110':
            last_index = 327
        self.mask_index =[x for x in range (0,last_index,3)]  # 记录计算卷积层在参数列表中的索引
#        self.mask_index =  [x for x in range (0,330,3)]
    # 代码中的mask为是否对卷积层进行掩盖，即掩码
    def init_mask(self,layer_rate):
        self.init_rate(layer_rate)
        for index, item in enumerate(self.model.parameters()):
            # 对每个卷积层求其密码本
            if(index in self.mask_index):
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index], self.model_length[index],self.decay_rate)
                self.mat[index] = self.convert2tensor(self.mat[index])
                if args.use_cuda:
                    self.mat[index] = self.mat[index].cuda()
        print("mask Ready")

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            # 对每个卷积层进行操作
            if(index in self.mask_index):
                # 将该卷积层的所有参数拉成一维向量，并与该层密码本相乘(即剪枝操作),再还原成原来的size
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        print("mask Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
#            if(index in self.mask_index):
            if(index == 0):
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()
                
                print("The first layer number of nonzero weight is %d, zero is %d" %( np.count_nonzero(b),len(b)- np.count_nonzero(b)))

    def all_zero_nums(self):
        non_zero = 0
        zero = 0
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                # if (index == 0):
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()
                non_zero += np.count_nonzero(b)
                zero += len(b)-np.count_nonzero(b)
            if index == max(self.mask_index):
                print("number of nonzero weight is %d, zero is %d" % (non_zero, zero))
if __name__ == '__main__':
    main()
