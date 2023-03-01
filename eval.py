import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np

from utils.utils import get_eval_loader, evaluate
from utils.parser import parser_args
from model.module import vit

def eval(args_dict):

    device = torch.device("cuda")

    # get dataLoader
    val_loader = get_eval_loader(args_dict) 
    
    # load model 
    model = Vit(args_dict).to(device)
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


if __name__ == '__main__':
    # get args
    args = parser_args()
    args_dict = vars(args)

    eval(args_dict)




