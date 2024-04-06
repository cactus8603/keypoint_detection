import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random

import torch
import math
import numpy as np
import timm
import argparse
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from ignite.handlers import create_lr_scheduler_with_warmup
from tensorboardX import SummaryWriter
# from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from torch.cuda import amp
from utils.dataset import ImgDataSet
from utils.utils import read_spilt_data, get_loader, train_one_epoch, evaluate
from utils.train import train

def create_parser():
    parser = argparse.ArgumentParser()

    # training details

    # scheduler parameter
    parser.add_argument("--lrf", default=0.0005, type='float', help="")
    parser.add_argument("--momentum", default=0.937, type='float', help="")
    parser.add_argument("--weight_decay", default=0.00005, type='float', help="")

    parser.add_argument("--epoch", default=100, type=int, help="")
    
    # parser.add_argument("", default="", type='float', help="")

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # get args
    args = create_parser()
    train(args)


    

    