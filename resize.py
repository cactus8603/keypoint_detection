from turtle import width
import cv2

from glob import glob
import os
from tqdm import tqdm


folder = glob(os.path.join('diff1107', '*'))
progress = tqdm(total = len(folder))

for ff in folder:
    imgs = glob(os.path.join(folder[0], '*'))
    for img in imgs:
        src = cv2.imread(img)
        dsize = (224,224)
        output = cv2.resize(src, dsize)
        cv2.imwrite(img,output)
    progress.update(1)


