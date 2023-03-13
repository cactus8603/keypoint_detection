import pickle
from fontTools.ttLib import TTFont
from fontTools.ttLib import TTCollection
from glob import glob
from PIL import ImageFont, Image, ImageDraw, ImageOps
import argparse
import os
import tensorflow as tf
import numpy as np
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# python gen.py --[要生成的字形檔們] --[要生成的字(格式為{unicode:chinese word}的字典)] --[生成結果資料夾的名稱]

# sample:
# python gen.py --font_set diff_font_files_1107 --word_set uni2word_stroke.pkl --dst_dir_name has_stroke

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--canvas_size', default=(224,224), help='畫布的大小')
parser.add_argument('--font_size', type=int, default=200, help='font size')
parser.add_argument('--font_set', type=str, default='../../data/Font/diff_font_files_1107', help='要生成的font set')
parser.add_argument('--word_set', type=str, default='./word_list/unicode2chara.pkl', help='要生成的word set')
parser.add_argument('--dst_dir_name', type=str, default='./image/byFont', help='生成的圖片放在哪個資料夾')

opt = parser.parse_args()

# 建立存檔位置
if not(os.path.exists(f'{opt.dst_dir_name}')):
    os.mkdir(f'{opt.dst_dir_name}')


def checkFontFile(file_path):
    # 檢查該字型檔能生成哪些字並回傳
    # ------------- TTC ---------------------------------
    if file_path.split('.')[-1].lower() == 'ttc':
        collection = TTCollection(file_path)
        fonts = list(collection)
        unicodes_set = set() # 所有能生成的字
        for font in fonts:
            for table in font['cmap'].tables:
                for table in font['cmap'].tables:
                    for i in table.cmap.keys():
                        unicodes_set.add(str(hex(i)).replace('0x', ''))

        return unicodes_set

    # ------------- TTF or OTF ---------------------------
    else:
        font = TTFont(file_path)
        unicodes_set = set() # 所有能生成的字

        for table in font['cmap'].tables:
            for i in table.cmap.keys():
                    unicodes_set.add(str(hex(i)).replace('0x', ''))

        return unicodes_set
    # ---------------------------------------------------




def word2image(unicode, word, font_path):
    img = Image.new("RGB", (opt.canvas_size[0]+300, opt.canvas_size[1]+300), (255,255,255)) # 空畫布 (先預設大 置中時再弄要的siz)
    draw = ImageDraw.Draw(img) # 實例化 Drew object在 空畫布上
    font = ImageFont.truetype(font_path, opt.font_size+200) # 字形檔和大小 (先預設大 置中時再弄要的size)
    draw.text((0,0), word, font=font, fill=(0,0,0)) # 畫在空畫布上(offset先不用 置中時再順便加入)

    # ------------置中並統一大小----------------
    gray_img = img.convert('L')
    gray_img = gray_img.point( lambda p: 255 if p > 127.5 else 0 )
    gray_img = np.bitwise_not(gray_img) # 換成黑底白字
    row, col = gray_img.shape[0], gray_img.shape[1]

    # 檢查白點獲得字的邊界
    for r in range(0, row):
        if(np.any(gray_img[r,:])):
            top = r
            break
    for r in range(row-1, -1, -1):
        if(np.any(gray_img[r,:])):
            bottom = r
            break
    for w in range(0, col):
        if(np.any(gray_img[:,w])):
            left = w
            break
    for w in range(col-1, -1, -1):
        if(np.any(gray_img[:,w])):
            right = w
            break

    try:
        sub_img = gray_img[top:bottom, left:right] #切割到只有字的大小
    except UnboundLocalError:
        # HanyiSentyMeadow 竟然出現明明cmap有該字 但生成出來是空白的狀況
        # 空白的話 top bottom left right 就不會被宣告 會UnboundLocalError
        return

    # tensorflow 有方便的 在保持比例不變的情況下縮放的功能 所以採用
    sub_img = tf.expand_dims(sub_img, axis=-1) # 為 tf 增加channel 維
    sub_img = tf.image.resize(sub_img, size=[opt.font_size, opt.font_size], preserve_aspect_ratio=True, antialias=True) # 保持比例的前提下resize
    # ----- 計算上下左右各要補多少到canvas size -----
    pad_h = opt.canvas_size[0] - sub_img.shape[0]
    pad_w = opt.canvas_size[1] - sub_img.shape[1]
    if pad_h%2==0:
        pad_top = int(pad_h/2)
        pad_bottom = int(pad_h/2)
    else:
        pad_top = math.ceil(pad_h/2)
        pad_bottom = math.floor(pad_h/2)
    if pad_w%2==0:
        pad_left = int(pad_w/2)
        pad_right = int(pad_w/2)
    else:
        pad_left = math.ceil(pad_w/2)
        pad_right = math.ceil(pad_w/2)

    sub_img = tf.cast(tf.squeeze(sub_img), dtype=tf.uint8).numpy() # 刪掉channel維, 轉unit8, 變numpy array
    sub_img = np.bitwise_not(sub_img)
    sub_img = Image.fromarray(sub_img) # 轉pillow image
    result_img = ImageOps.expand(sub_img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=255) # padding到 canvas_size

    # ByFont
    fontname = font_path.split('/')[-1].split('.')[0]
    if not(os.path.exists(f'{opt.dst_dir_name}/{fontname}')):
        os.mkdir(f'{opt.dst_dir_name}/{fontname}')

    result_img.save(f'{opt.dst_dir_name}/{fontname}/{unicode}.jpg')

    return f'{opt.dst_dir_name}/{fontname}/{unicode}.jpg'

    # ByUnicode
    # fontname = font_path.split('/')[-1].split('.')[0]
    # if not(os.path.exists(f'{opt.dst_dir_name}/{unicode}')):
    #     os.mkdir(f'{opt.dst_dir_name}/{unicode}')

    # result_img.save(f'{opt.dst_dir_name}/{unicode}/{fontname}.jpg')

    # return f'{opt.dst_dir_name}/{unicode}/{fontname}.jpg'





with open(f'{opt.word_set}', 'rb') as r: # 獲得要生成的所有字
    uni2word = pickle.load(r)

font_paths = glob(f'{opt.font_set}/*') # 獲得所有要生成的字形檔
for i, font_path in enumerate(font_paths): # 所有要生成的字型
    unicode_can_gen = list(checkFontFile(font_path))
    for j, unicode in enumerate(uni2word.keys()): # 所有想生成的字
        # print(j, unicode)
        # unicode = str(unicode)
        if unicode in unicode_can_gen: # 檢查是否能生成
            # print(type(unicode), type(uni2word[unicode]))
            # print((unicode), (uni2word[unicode]))
            # print(str(unicode), str(uni2word[unicode]))
            filename = word2image(unicode, str(uni2word[unicode]), font_path)
            print(f'font:{i+1}/{len(font_paths)} word:{j+1}/{len(uni2word.keys())} - {filename}')