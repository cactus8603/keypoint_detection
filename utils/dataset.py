import torch
# import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Normalize

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class ImgDataSet(Dataset):
    def __init__(self, img_data, img_label, args_dict):
        super().__init__()
        self.img_data = img_data
        self.img_label = img_label
        self.num_classes = args_dict['n_classes']

        self.transform = transforms.Compose([
            ToPILImage(),
            Resize((224, 224)), 
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, idx):
        img = cv2.imread(self.img_data[idx])
        label = self.img_label[idx]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.zeros_like(img)
        img[:,:,0] = gray
        img[:,:,1] = gray
        img[:,:,2] = gray

        img_tensor = self.transform(img)
        label_tensor = torch.zeros(self.num_classes)
        label_tensor[label] = 1.
        # print(img_tensor.shape)
        # print(label_tensor)
        # print(label_tensor.shape)

        return img_tensor, label_tensor
    
    def __len__(self):
        return len(self.img_data)

