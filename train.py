import argparse
import yaml
from torchvision.transforms import Compose, Resize, ToTensor
from torchsummary import summary
from PIL import Image

from model.Vit import Vit

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", default="config.yaml", nargs='?', help="path to config file")
    
    args = parser.parse_args()
    return args

def parser_args():
    args = create_parser()
    if args.config_path:
        with open(args.config_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        
        arg_dict = args.__dict__
        for key, value in data.items():
            # print(key, value)
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].extend(value)
            else:
                arg_dict[key] = value
    return args
    
if __name__ == '__main__':
    args = parser_args()
    args_dict = vars(args)
    # print(args_dict)
    # print(args.batch_size)

    img = Image.open('./1.png')
    transform = Compose([Resize((224, 224)), ToTensor()])
    x = transform(img)
    x = x.unsqueeze(0)
    # print(x.shape)

    model = Vit(args_dict=args_dict)
    # summary(model(), (1,1,224,224), device='cpu')
    pred = model(x)
    print(pred)

    