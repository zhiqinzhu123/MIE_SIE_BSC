# coding=utf-8
# python train.py --cfg=BraTS18

import argparse
import os
import time
import logging
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

from thop import profile

cudnn.benchmark = True

import numpy as np
import models

from data import datasets
from data.sampler import CycleSampler
from data.data_utils import init_fn
from utils import Parser, criterions, count_params

from predict import AverageMeter

# import setproctitle

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--cfg', default='BASE', type=str,
                    help='Your detailed configuration of the network')  # 模型名称
parser.add_argument('-gpu', '--gpu', default='0', type=str,
                    help='Supprot one GPU & multiple GPUs.')  # GPU选卡
parser.add_argument('-batch_size', '--batch_size', default=2, type=int,
                    help='Batch size')  # batch_size
parser.add_argument('-restore', '--restore', default='model_last.pth', type=str)  # 从xx个已存的模型继续训练
path = os.path.dirname(__file__)

# parse arguments
args = parser.parse_args()
args = Parser(args.cfg, log='train').add_args(args)
ckpts = args.makedir()
args.resume = os.path.join(ckpts, args.restore)  # specify the epoch


def main():
    # GPU配置
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 模型，优化器，损失函数配置
    Network = getattr(models, args.net)
    model = Network(**args.net_params)
    # print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    # model = torch.nn.DataParallel(model)
    flops, params = profile(model, inputs=(4, 128, 128, 128))
    print(flops)
    print(params)
    optimizer = getattr(torch.optim, args.opt)(model.parameters(), **args.opt_params)
    criterion = getattr(criterions, args.criterion)

    # 检查已保存的模型
    msg = ''
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            msg = ("=> loaded checkpoint '{}' (iter {})".format(args.resume, checkpoint['iter']))
        else:
            msg = "=> no checkpoint found at '{}'".format(args.resume)
    else:
        msg = '-------------- New training session ----------------'
    msg += '\n' + str(args)
    logging.info(msg)

    # 数据加载
    Dataset = getattr(datasets, args.dataset)

    train_list = os.path.join(args.train_data_dir, args.train_all_list)
    train_set = Dataset(train_list, root=args.train_data_dir, pm=args.pm, transforms=args.train_transforms)

    num_iters = args.num_iters or (len(train_set) * args.num_epochs) // args.batch_size
    num_iters -= args.start_iter
    train_sampler = CycleSampler(len(train_set), num_iters * args.batch_size)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        collate_fn=train_set.collate,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=init_fn)

    start = time.time()

    enum_batches = len(train_set) / float(args.batch_size)  # nums_batch per epoch

    losses = AverageMeter()
    torch.set_grad_enabled(True)

    for i, data in enumerate(train_loader, args.start_iter):

        # elapsed_bsize = int( i / enum_batches)+1
        epoch = int((i + 1) / enum_batches)
        # setproctitle.setproctitle("Epoch:{}/{}".format(elapsed_bsize,args.num_epochs))

        # actual training
        adjust_learning_rate(optimizer, epoch, args.num_epochs, args.opt_params.lr)
        if not args.weight_type:  # compatible for the old version
            args.weight_type = 'square'

        data = [t.cuda(non_blocking=True) for t in data]

        if args.pm:
            x, target, pointmap = data[:3]
            output = model(x)
            loss = criterion(output, target, pointmap)
        else:
            x, target = data[:2]
            print("*********************************************")
            print(x.shape)
            print("*********************************************")
            output = model(x)
            loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), target.numel())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % int(enum_batches * args.save_freq) == 0 \
                or (i + 1) % int(enum_batches * (args.num_epochs - 1)) == 0 \
                or (i + 1) % int(enum_batches * (args.num_epochs - 2)) == 0 \
                or (i + 1) % int(enum_batches * (args.num_epochs - 3)) == 0 \
                or (i + 1) % int(enum_batches * (args.num_epochs - 4)) == 0:
            file_name = os.path.join(ckpts, 'model_epoch_{}.pth'.format(epoch))
            torch.save({
                'iter': i,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

        msg = 'Iter {0:}, Epoch {1:.4f}, Loss {2:.4f}'.format(i + 1, (i + 1) / enum_batches, losses.avg)
        logging.info(msg)

        losses.reset()

    i = num_iters + args.start_iter
    file_name = os.path.join(ckpts, 'model_last.pth')
    torch.save({
        'iter': i,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
    },
        file_name)

    msg = 'total time: {:.4f} minutes'.format((time.time() - start) / 60)
    logging.info(msg)


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - epoch / MAX_EPOCHES, power), 8)


if __name__ == '__main__':
    main()
