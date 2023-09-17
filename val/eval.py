import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
from utils import get_loader, load_model, evaluate

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="style", type=str, help="content/style")
    parser.add_argument("--data_path", default="/code/val/Dataset/byFont", type=str, help="path to val data")
    parser.add_argument("--port", default=8673, type=int, help="port of dist")
    parser.add_argument("--batch_size", default=128, type=int, help="") # 4, 128, 32, 56, 56
    parser.add_argument("--num_workers", default=6, type=int, help="")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # get args
    args = create_parser()
    args.n_classes = 17350
    
    if args.mode == 'content':
        args.n_classes = 4807
        args.model_path = './cfgs/content/model_0.982_.pth'
        args.dict_path = './cfgs/char_classes_4807.json'
    elif args.mode == 'style':
        args.n_classes = 173
        args.model_path = './cfgs/style/model_0.989_.pth'
        args.dict_path = './cfgs/font_classes_173.json'
    
    model = load_model(args)
    
    val_loader = get_loader(args)
    acc = evaluate(model, val_loader)
    print('average accuracy: {}%'.format(acc*100))

