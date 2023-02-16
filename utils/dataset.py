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
    def __init__(self, img_data, img_label):
        super().__init__()
        self.img_data = img_data
        self.img_label = img_label

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

        img_tensor = self.transform(img).unsqueeze(0)
        label_tensor = torch.from_numpy(label)

        return img_tensor, label_tensor

