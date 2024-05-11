# coding=utf-8
import argparse
import os
import logging
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

cudnn.benchmark = True

import numpy as np

import models
from data import datasets
from utils import Parser, str2bool

from predict import validate_softmax

parser = argparse.ArgumentParser()

parser.add_argument('-cfg', '--cfg', default='BASE', required=True, type=str,
                    help='Your detailed configuration of the network')  # 详细配置

parser.add_argument('-mode', '--mode', default=0, required=True, type=int, choices=[0, 1, 2],  # 测试模式
                    help='0 for cross-validation on the training set; '  # 0 用于训练集的交叉训练集的验证
                         '1 for validing on the validation set; '  # 1 用于训练集的交叉验证集的验证
                         '2 for testing on the testing set.')  # 2 用于验证集上进行验证

parser.add_argument('-gpu', '--gpu', default='6,7', type=str)  # GPU 选择

parser.add_argument('-is_out', '--is_out', default=True, type=str2bool,  # 是否保存测试nii
                    help='If ture, output the .nii file')

parser.add_argument('-verbose', '--verbose', default=True, type=str2bool,  # 是否输出更多测试信息
                    help='If True, print more infomation of the debuging output')

parser.add_argument('-use_TTA', '--use_TTA', default=True, type=str2bool,  # 是否使用TTA
                    help='It is a postprocess approach.')

parser.add_argument('-postprocess', '--postprocess', default=True, type=str2bool,  # 是否使用后处理
                    help='Another postprocess approach.')

parser.add_argument('-save_format', '--save_format', default='nii', choices=['nii', 'npy'], type=str,  # 保存格式
                    help='[nii] for submission; [npy] for models ensemble')

parser.add_argument('-snapshot', '--snapshot', default=True, type=str2bool,
                    help='If True, saving the snopshot figure of all samples.')  # 是否保存snopshot

parser.add_argument('-restore', '--restore', default='model_last.pth', type=str,  # 保存的模型路径
                    help='The path to restore the model.')

path = os.path.dirname(__file__)

args = parser.parse_args()
args = Parser(args.cfg, log='train').add_args(args)
args.gpu = str(args.gpu)
ckpts = args.makedir()
args.resume = os.path.join(ckpts, args.restore)  # specify the epoch


def main():
    # setup environments and seeds
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    Network = getattr(models, args.net)  #
    model = Network(**args.net_params)

    model = torch.nn.DataParallel(model).cuda()
    print(args.resume)
    assert os.path.isfile(args.resume), "no checkpoint found at {}".format(args.resume)
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_iter = checkpoint['iter']
    model.load_state_dict(checkpoint['state_dict'])
    msg = ("=> loaded checkpoint '{}' (iter {})".format(args.resume, checkpoint['iter']))

    msg += '\n' + str(args)
    logging.info(msg)

    if args.mode == 0:
        root_path = args.train_data_dir
        data_list = args.train_list
        is_scoring = True
    elif args.mode == 1:
        root_path = args.train_data_dir
        data_list = args.valid_list
        is_scoring = True
    elif args.mode == 2:
        root_path = args.test_data_dir
        data_list = args.test_list
        is_scoring = False
    else:
        raise ValueError

    Dataset = getattr(datasets, args.dataset)
    valid_list = os.path.join(root_path, data_list)
    valid_set = Dataset(valid_list, root=root_path, pm=args.pm, transforms=args.test_transforms)

    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        collate_fn=valid_set.collate,
        num_workers=10,
        pin_memory=True)

    if args.is_out:
        out_dir = './output/{}'.format(args.cfg)
        os.makedirs(os.path.join(out_dir, 'submission'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'snapshot'), exist_ok=True)
    else:
        out_dir = ''

    logging.info('-' * 50)
    logging.info(msg)

    with torch.no_grad():
        validate_softmax(
            valid_loader,
            model,
            cfg=args.cfg,
            savepath=out_dir,
            save_format=args.save_format,
            names=valid_set.names,
            scoring=is_scoring,
            verbose=args.verbose,
            use_TTA=args.use_TTA,
            snapshot=args.snapshot,
            postprocess=args.postprocess,
            cpu_only=False)


if __name__ == '__main__':
    main()
