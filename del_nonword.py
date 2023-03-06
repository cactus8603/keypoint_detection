import os
from glob import glob

# glob all font folder
font_folder = glob(os.path.join("./image", '*'))
# print(font_folder)

# glob each word in folder
for font in font_folder:
    imgs = glob(os.path.join(font, '*'))
    for img in imgs:
        stats = os.stat(img)
        size = stats.st_size
        if size < 1000:
            os.remove(img)
    os.remove(os.join)
