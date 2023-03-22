# https://github.com/pytorch/vision/blob/master/torchvision/models/__init__.py
import argparse
import os
import shutil
import pdb, time
from collections import OrderedDict
import torchvision.datasets as dset
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import convert_secs2time, time_string, time_file_str
import models
import numpy as np
'''
用于加载已经训练的模型或者是训练过程中最好的模型

'''
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data_path', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--save_dir', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=5, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# compress rate
parser.add_argument('--rate', type=float, default=0.9, help='compress rate of model')
parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
parser.add_argument('--skip_downsample', type=int, default=1, help='compress layer of model')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--eval_small', dest='eval_small', action='store_true', help='whether a big or small model')
parser.add_argument('--small_model', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

args = parser.parse_args(['./data/cifar.python', '-a', 'resnet20', '--workers', '0', '--batch-size', '32',
                        # '--resume', 'best_model_test/res_20_checkpoint.pth.tar',
                        '--small_model', 'best_model_test/model_best.pth.tar','--dataset','cifar10',
                        '-e', '--eval_small', '--save_dir', './logs/infer_small_model'])
args.use_cuda = torch.cuda.is_available()



def main():
    best_prec1 = 0

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, 'gpu-time.{}.log'.format(args.arch)), 'w')

    # create model
    print_log("=> creating model '{}'".format(args.arch), log)
    # load Imagenet model

    # load cifar model
    # model = models.__dict__[args.arch](10)
    print_log("=> Model : {}".format(args.arch), log)
    print_log("=> parameter : {}".format(args), log)
    print_log("Compress Rate: {}".format(args.rate), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("Skip downsample : {}".format(args.skip_downsample), log)

    # optionally resume from a checkpoint


    cudnn.benchmark = True
    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)  # assert后为真，则继续向下进行，否则输出后面的报错信息


    test_transform = transforms.Compose(
        [transforms.ToTensor(),  # 数据格式转换成tensor
         transforms.Normalize(mean, std)])  # 同正则化
    if args.dataset == 'cifar10':
        # root=cifar-10-batches-py的根目录，train=是训练集，transform=数据的转换操作，download=从网络下载数据，并进行数据初始化操作
        # train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        # train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)
    # Data loading code
    # valdir = os.path.join(args.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    model = models.__dict__[args.arch](num_classes)

    # model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))  # GPU并行计算

    criterion = nn.CrossEntropyLoss().cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            state_dict = checkpoint['state_dict']
            state_dict = remove_module_dict(state_dict)
            model.load_state_dict(state_dict)
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)

    if args.evaluate:
        print_log("eval true", log)
        if not args.eval_small:
            big_model = model.cuda()
            print_log('Evaluate: big model', log)
            print_log('big model accu: {}'.format(validate(val_loader, big_model, criterion, log)), log)
        else:
            print_log('Evaluate: small model', log)
            if args.small_model:
                if os.path.isfile(args.small_model):
                    print_log("=> loading small model '{}'".format(args.small_model), log)
                    small_model = torch.load(args.small_model)
                    # for x, y in zip(small_model.named_parameters(), model.named_parameters()):
                    #     print_log("name of layer: {}\n\t *** small model {}\n\t *** big model {}".format(x[0], x[1].size(),y[1].size()), log)
                    model = small_model['state_dict']
                    if args.use_cuda:
                        model = model.cuda()
                    print_log('small model accu: {}'.format(validate(val_loader, model, criterion, log)), log)
                else:
                    print_log("=> no small model found at '{}'".format(args.small_model), log)
        return


def validate(val_loader, model, criterion, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        if args.use_cuda:
            input, target = input.cuda(), target.cuda(non_blocking = True)
        with torch.no_grad():
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

        # measure elapsed time
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

    print_log(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                           error1=100 - top1.avg), log)

    return top1.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()




class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def remove_module_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


if __name__ == '__main__':
    main()