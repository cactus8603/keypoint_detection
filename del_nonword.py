import os
from tqdm import tqdm
from glob import glob

def del_empty_img():
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

def count_few_folder():
    code_folder = glob(os.path.join("../SplitFontFileAugData1", '*'))

    progress = tqdm(total=len(code_folder))

    few_code = []
    for code in code_folder:
        imgs = glob(os.path.join(code, '*'))
        if len(imgs) == 0:
            few_code.append(code)
        progress.update(1)

    print(len(few_code))


if __name__ == '__main__':
    count_few_folder()