import os
import torch
import cv2
import numpy as np
from model.module import *
import argparse
import yaml
from torchvision.transforms import transforms
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Normalize
import numpy as np

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

img = cv2.imread('test.png')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.zeros_like(img)
img[:,:,0] = gray
img[:,:,1] = gray
img[:,:,2] = gray

trans = transforms.Compose([
            ToPILImage(),
            Resize((224, 224)), 
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

cv2.imshow('image',img)
cv2.waitKey(0)

print("Load img", img.shape)
img = trans(img).unsqueeze(0)
print("Unsqueeze img", img.shape)

# get args
args = parser_args()
args_dict = vars(args)

patch = PatchEmbeddings(args_dict).forward(img)
print("Patch Embed",patch.shape)



x = TransformerEncoder(args_dict).forward(patch)
print("After transformer encoder", x.shape)

x = TransformerDecoder(args_dict).forward(x)
print("After transformer decoder", x.shape)

stitch = transforms.Compose([
            ToPILImage(),
        ])

patch = Reconstruction(args_dict).forward(patch)
patch = stitch(patch[0])

img2 = cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2BGR)
print("To cv2",img2.shape)

cv2.imshow('image',img2)
cv2.waitKey(0)

x = Reconstruction(args_dict).forward(x)
x = stitch(x[0])
img3 = cv2.cvtColor(np.asarray(x), cv2.COLOR_RGB2BGR)
print("To cv2",img3.shape)

cv2.imshow('image',img3)
cv2.waitKey(0)



