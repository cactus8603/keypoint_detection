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

# train function
def train(args):
    cudnn.benchmark = True

    # set gpu of each multiprocessing
    # torch.cuda.set_device(ddp_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get dataLoader
    train_loader, val_loader = get_loader(args_dict) 
    print("Get data loader successfully")

    ### import timm model, see more in model.txt ###

    # model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=args_dict['n_classes'])
    model = 0

    # setting optim
    pg = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(pg, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # setting lr scheduler as cosine annealing
    lf = lambda x: ((1 + math.cos(x * math.pi / args.cosanneal_cycle)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lf)
    
    # setting warm up info
    # if args.warmup'] :
    #     warmup = create_lr_scheduler_with_warmup(
    #         scheduler, 
    #         warmup_start_value=args.warmup_start_value'],
    #         warmup_end_value=args.lr'],
    #         warmup_duration=args.warmup_step'],
    #     )

    start_epoch = 0

    model = model.to(device)
    score = 0
    
    # setting Automatic mixed precision
    scaler = amp.GradScaler()

    # eval
    val_loss, val_acc = evaluate(
        model=model, 
        data_loader=val_loader,
        device=device,
        epoch=args.epoch,
        classes=args.n_classes
    )

    # start training
    # for epoch in range(start_epoch, args.epoch']):
        
    #     # train 
    #     train_loss, train_acc = train_one_epoch(
    #         model=model, 
    #         optimizer=opt,
    #         data_loader=train_loader,
    #         device=ddp_gpu,
    #         epoch=epoch,
    #         scaler=scaler,
    #         args_dict=args_dict
    #     )

    #     scheduler.step()

    #     # eval
    #     val_loss, val_acc = evaluate(
    #         model=model, 
    #         data_loader=val_loader,
    #         device=ddp_gpu,
    #         epoch=epoch,
    #         classes=args_dict['n_classes']
    #     )

        
        # write info into summarywriter in main worker
        # if is_main_worker(ddp_gpu):
        #     tags = ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        #     tb_writer.add_scalar(tags[0], train_loss, epoch)
        #     tb_writer.add_scalar(tags[1], train_acc, epoch)
        #     tb_writer.add_scalar(tags[2], val_loss, epoch)
        #     tb_writer.add_scalar(tags[3], val_acc, epoch)
        #     tb_writer.add_scalar(tags[4], opt.param_groups[0]['lr'], epoch)

        #     # save model every two epoch 
        #     if (epoch % args_dict['save_frequency'] == 0 and epoch >= 10):
        #         save_path = os.path.join(args_dict['model_save_path'], "model_{}_{:.3f}_.pth".format(epoch, val_acc))
        #         torch.save(model, save_path)
        #     elif (epoch >= 10 and score < val_acc):
        #         save_path = os.path.join(args_dict['model_save_path'], "model_{}_{:.3f}_.pth".format(epoch, val_acc))
        #         torch.save(model, save_path)
        #         score = val_acc