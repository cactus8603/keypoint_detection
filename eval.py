import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import numpy as np
import cv2
import argparse
# from utils.utils import get_eval_loader, evaluate, get_loader
# from utils.parser import parser_args
import torch.distributed as dist
from torchvision.transforms import transforms
import torch.nn.functional as F
import json
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Normalize

from glob import glob

# python eval.py --model_path  /data/model_performance/style/from_handover/swin_large_patch4_window7_224/model_40_0.994_.pth --data_path ./eval/ARHeiB5Ultra

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/model_performance/style/from_handover/swin_large_patch4_window7_224/model_40_0.994_.pth", type=str, help="path to model checkpoint")
    parser.add_argument("--data_path", default="./eval/ARHeiB5Ultra", type=str, help="path to val data")
    parser.add_argument("--port", default=8993, type=int, help="port of dist")
    args = parser.parse_args()
    return args

# def eval(args_dict):

#     device = torch.device("cuda")

#     # get dataLoader
    
#     val_loader = get_eval_loader(args_dict) 
#     # train_loader, val_loader = get_loader(args_dict) 
    
#     # load model 
#     # model = Vit(args_dict).to(device)
#     init_model_path = os.path.dirname(args_dict['load_model_path']) + "/init_model.pt"
#     # torch.save(model, init_model_path)
    
#     print(init_model_path)
#     model = torch.load(init_model_path, map_location=device)
#     model.load_state_dict(torch.load(args_dict['load_model_path']), strict=True)
#     print("Load model successfully")

#     # eval
#     val_loss, val_acc, cm = evaluate(
#         model=model, 
#         data_loader=val_loader,
#         device=device,
#         epoch=-1,
#         classes=args_dict['n_classes']
#     )

#     save_path = os.path.dirname(args_dict['load_model_path']) + '/result-' + str(os.path.basename(args_dict['val_data_path']))
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     # np.savetxt(save_path + '/cm.csv', cm, delimiter=',')

#     with open(save_path + '/result.txt', 'w') as f:
#         f.write("acc:" + str(val_acc) + "\n")
#         f.write("loss:" + str(val_loss))

def load_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dist.init_process_group(
        backend='nccl',
        init_method="tcp://127.0.0.1:" + str(args.port),
        world_size=1,
        rank=0,
    )

    init_model_path = os.path.dirname(args.model_path) + "/init_model.pt"
    model = torch.load(init_model_path, map_location=device)
    model.load_state_dict(torch.load(args.model_path), strict=True)
    
    print("load model successful")

    return model

def eval_singel(path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        ToPILImage(),
        Resize((224, 224)), 
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    img = cv2.imread(path)
    # label = self.img_label[idx]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.zeros_like(img)
    img[:,:,0] = gray
    img[:,:,1] = gray
    img[:,:,2] = gray

    img_tensor = transform(img).unsqueeze(0).to(device)
    # label_tensor = torch.zeros(self.num_classes)
    # label_tensor[label] = 1.
    

    pred = model(img_tensor)
    p = F.softmax(pred, dim=1)
    idx = str(int(p.argmax(1)))
    
    data = json.load(open('./cfgs/font_class.json'))
    print(os.path.basename(data[idx]))

if __name__ == '__main__':
    # get args
    args = create_parser()

    model = load_model(args)

    chars = glob(os.path.join(args.data_path, '*.png'))
    for char in chars:
        eval_singel(char, model)

    # for folder in folders:
    #     print(folder)
    #     chars = glob(os.path.join(folder, "*.png"))
    #     for char in chars:
    #         eval_singel(char, model)
    
    # eval_singel("./ref_img/U4F60_JasonHandwriting4.png")





