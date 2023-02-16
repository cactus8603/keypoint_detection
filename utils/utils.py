import os
import random
import torch
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader

def read_spilt_data(args_dict):
    random.seed(0)
    assert os.path.exists(args_dict['data_path']), "data path:{} does not exist".format(args_dict['data_path'])

    font_class = glob(os.path.join(args_dict['data_path'], '*'))
    font_class.sort()
    font_class_indices = dict((k, v) for v, k in enumerate(font_class))
    # print(font_class_indices)

    train_data = []
    train_label = []
    val_data = []
    val_label = []

    for cla in font_class:
        img = glob(os.path.join(cla, '*'))
        # print(img)
        img_class = font_class_indices[cla]
        # print(img_class)

        spilt_point = random.sample(img, k=int(len(img) * args_dict['spilt_rate']))
        
        for img_path in img:
            if img_path in spilt_point:
                train_data.append(img_path)
                train_label.append(img_class)
            else:
                val_data.append(img_path)
                val_label.append(img_class)
    
    return train_data, train_label, val_data, val_label

def get_loader(args_dict):
    train_data, train_label, val_data, val_label = read_spilt_data(args_dict)

    train_loader = DataLoader(
        train_data,
        batch_size=args_dict['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=args_dict['num_workers'] 
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=args_dict['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=args_dict['num_workers'] 
    )

    return train_loader, val_loader

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)

    for i, (img, label) in enumerate(data_loader):
        img, label = img.to(device), label.to(device)
        sample_num += img.shape[0]

        pred = model(img)
        pred_class = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_class, label).sum()
        loss = loss_function(pred, label)
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "train epoch:{}, loss:{:.5f}, acc:{:.5f}".format(epoch, accu_loss.item()/(i+1), accu_num.item() / sample_num)
    
        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item()/(i+1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader)

    for i, (img, label) in enumerate(data_loader):
        img, label = img.to(device), label.to(device)
        sample_num += img.shape[0]

        pred = model(img)
        pred_class = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_class, label).sum()

        loss = loss_function(pred, label)
        accu_loss += loss

        data_loader.desc = "valid epoch:{}, loss:{.5f}, acc:{.5f}".format(epoch, accu_loss.item()/(i+1), accu_num.item() / sample_num)
    
    return accu_loss.item()/(i+1), accu_num.item() / sample_num

