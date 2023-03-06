import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import cv2

from utils.utils import get_eval_loader, evaluate, get_loader
from utils.parser import parser_args
from torchvision.transforms import transforms
import torch.nn.functional as F
import json
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Normalize
from model.vit import Vit

def eval(args_dict):

    device = torch.device("cuda")

    # get dataLoader
    val_loader = get_eval_loader(args_dict) 
    args_dict['use_ddp'] = False
    # train_loader, val_loader = get_loader(args_dict) 
    
    # load model 
    # model = Vit(args_dict).to(device)
    init_model_path = os.path.dirname(args_dict['load_model_path']) + "/init_model.pt"
    # torch.save(model, init_model_path)
    
    print(init_model_path)
    model = torch.load(init_model_path, map_location=device)
    model.load_state_dict(torch.load(args_dict['load_model_path']), strict=True)
    print("Load model successfully")

    # eval
    val_loss, val_acc, cm = evaluate(
        model=model, 
        data_loader=val_loader,
        device=device,
        epoch=-1,
        classes=args_dict['n_classes']
    )

    save_path = os.path.dirname(args_dict['load_model_path']) + '/result'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # np.savetxt(save_path + '/cm.csv', cm, delimiter=',')

    with open(save_path + '/result.txt', 'w') as f:
        f.write("acc:" + str(val_acc) + "\n")
        f.write("loss:" + str(val_loss))

def eval_singel(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_model_path = os.path.dirname(args_dict['load_model_path']) + "/init_model.pt"
    model = torch.load(init_model_path, map_location=device)
    model.load_state_dict(torch.load(args_dict['load_model_path']), strict=True)

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
    args = parser_args()
    args_dict = vars(args)

    # eval(args_dict)
    eval_singel("./1.png")





