import os
from tqdm import tqdm
from glob import glob

# glob all font folder
font_folder = glob(os.path.join("./image/all_word", '*'))
# print(font_folder)

progress = tqdm(total=len(font_folder))

# glob each word in folder
for font in font_folder:
    imgs = glob(os.path.join(font, '*'))
    for img in imgs:
        stats = os.stat(img)
        size = stats.st_size
        if size < 1000:
            os.remove(img)
    # os.remove(os.join)

    progress.update(1)
