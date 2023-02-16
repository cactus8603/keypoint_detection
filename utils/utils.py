import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from .dataset import ImgDataSet

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

    train_dataset = ImgDataSet(train_data, train_label, args_dict)
    val_dataset = ImgDataSet(val_data, val_label, args_dict)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args_dict['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=args_dict['num_workers'] 
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args_dict['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=args_dict['num_workers'] 
    )

    return train_loader, val_loader

def get_confusion_matrix(y_true, y_pred):
    y_true, y_pred = y_true.cpu().detach().numpy() , y_pred.cpu().detach().numpy() 
    cm = np.array(confusion_matrix(y_true, y_pred),dtype=float)
    tmp = np.append(y_true, y_pred)
    ele = np.unique(tmp)
    for idx in range(0,14):
        if (idx not in ele):

            cm = np.insert(cm, idx, np.zeros(cm.shape[0]), axis=1)
            cm = np.insert(cm, idx, np.zeros(cm.shape[1]), axis=0)
    # print(cm.shape)
    return 

def WP_score(cm):
    # FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    # TN = cm.sum() - (FP + FN + TP)

    WP = 0
    for idx in range(0,14):
        precision = float(TP[idx] / (TP[idx]+FN[idx]))
        # recall =  float(TP[idx] / (TP[idx]+FP[idx]))
        # f1 = 2 * (precision*recall) / (precision+recall)
        # print("Type:{}, f1-score:{}".format(idx+1, f1))

        WP += precision*(TP[idx]+FN[idx])
    
    return 

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
        # print(img.shape)
        # print(label.shape)
        # break
        sample_num += img.shape[0]

        pred = model(img)
        p = F.softmax(pred, dim=1)
        accu_num += (p.argmax(1) == label.argmax(1)).type(torch.float).sum().item()

        # pred_class = torch.max(pred, dim=1)[1]
        # accu_num += torch.eq(pred_class, label).sum()

        loss = loss_function(pred, label)
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "train epoch:{}, loss:{:.5f}, acc:{:.5f}".format(epoch, accu_loss.item()/(i+1), accu_num.item() / sample_num)
    
        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (i+1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader)

    cm = np.zeros((142, 142),dtype=float)

    for i, (img, label) in enumerate(data_loader):
        img, label = img.to(device), label.to(device)
        sample_num += img.shape[0]

        pred = model(img)
        p = F.softmax(pred, dim=1)
        accu_num += (p.argmax(1) == label.argmax(1)).type(torch.float).sum().item()

        # pred_class = torch.max(pred, dim=1)[1]
        # accu_num += torch.eq(pred_class, label).sum()

        loss = loss_function(pred, label)
        accu_loss += loss

        cm += get_confusion_matrix(label.argmax(1), p.argmax(1))

    WP = WP_score(cm) / sample_num
    data_loader.desc = "valid epoch:{}, loss:{.5f}, acc:{.5f}, WP={.5f}".format(epoch, accu_loss.item()/(i+1), accu_num.item() / sample_num, WP)

    
    return accu_loss.item()/(i+1), accu_num.item() / sample_num

