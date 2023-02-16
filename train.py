
import torch
import cv2
import os
import math
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
# from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage

from utils.dataset import ImgDataSet
from utils.utils import read_spilt_data, get_loader, train_one_epoch, evaluate
from utils.parser import parser_args
from model.Vit import Vit

def train(args_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_loader(args_dict) 

    if not os.path.exists(args_dict['model_save_path']):
        os.mkdir(args_dict['model_save_path'])
    tb_writer = SummaryWriter(args_dict['model_save_path'])

    model = Vit(args_dict).to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(pg, lr=args_dict['lr'], momentum=args_dict['momentum'], weight_decay=args_dict['weight_decay'])
    lf = lambda x: ((1 + math.cos(x * math.pi / args_dict['epoch'])) / 2) * (1 - args_dict['lrf']) + args_dict['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lf)

    print("Start Training")
    for epoch in range(args_dict['epoch']):

        train_loss, train_acc = train_one_epoch(
            model=model, 
            optimizer=opt,
            data_loader=train_loader,
            device=device,
            epoch=epoch
        )

        val_loss, val_acc = evaluate(
            model=model, 
            data_loader=val_loader,
            device=device,
            epoch=epoch
        )

        scheduler.step()

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], opt.param_groups[0]['lr'], epoch)

        if (epoch % 10 == 0):
            save_path = args_dict['model_save_path'] + "/model_{}.pth".format(epoch)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(model.state_dict(), save_path)



if __name__ == '__main__':
    args = parser_args()
    args_dict = vars(args)


    
    train(args_dict)

    # model = Vit(args_dict=args_dict)
    # # summary(model(), (1,3,224,224), device='cpu')
    # pred = model(x)
    # print(pred)

    