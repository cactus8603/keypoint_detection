from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import os
import numpy as np
import pathlib
import argparse
from fontTools.ttLib.ttFont import TTFont

parser = argparse.ArgumentParser(description='Obtaining characters from .ttf')
parser.add_argument('--ttf_path', type=str, default='../../data/Font/diff_font_files_1107',help='ttf directory')
parser.add_argument('--chara', type=str, default='useword.txt',help='characters')
parser.add_argument('--save_path', type=str, default='./image/diff1107',help='images directory')
parser.add_argument('--img_size', type=int, default=224, help='The size of generated images')
parser.add_argument('--chara_size', type=int, default=196, help='The size of generated characters')
# parser.add_argument('--img_size', type=int, default=76, help='The size of generated images')
# parser.add_argument('--chara_size', type=int, default=57, help='The size of generated characters')
args = parser.parse_args()

file_object = open(args.chara,encoding='utf-8')   
try:
	characters = file_object.read()
finally:
    file_object.close()


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    return img

def draw_example(ch, src_font, canvas_size, x_offset, y_offset):
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    example_img.paste(src_img, (0, 0))
    return example_img

data_dir = args.ttf_path
data_root = pathlib.Path(data_dir)
# print(data_root)

all_image_paths = list(data_root.glob('*.*'))
# all_image_paths.append(list(data_root.glob('*.ttf')))
# all_image_paths.append(list(data_root.glob('*.ttc')))
all_image_paths = [str(path) for path in all_image_paths]
# print(len(all_image_paths))
# for i in range (len(all_image_paths)):
#     print(all_image_paths[i])
print(all_image_paths)

seq = list()

for (label,item) in zip(range(len(all_image_paths)),all_image_paths):
    src_font = ImageFont.truetype(item, size = args.chara_size)

    # print(os.path.basename(item).split('.')[0])
    # label += 193
    for (chara,cnt) in zip(characters, range(len(characters))):
        # trans usuall words to utf-8
        t = str(chara.encode('unicode_escape').decode('utf-8')[1:].upper()) + '_' + str(os.path.basename(item).split('.')[0])
        # trans rare words to uni-han
        # t = str(chara.encode('unicode_escape').decode('utf-8')[5:14].upper())
        print('chara:{}, src_font:{}, arg.img_size:{}'.format(t, src_font, args.img_size))
        img = draw_example(chara, src_font, args.img_size, (args.img_size-args.chara_size)/2, (args.img_size-args.chara_size)/2)
        font_name = os.path.splitext(os.path.basename(item))[0]
        path_full = os.path.join(args.save_path, font_name)
        
        if not os.path.exists(path_full):
            os.mkdir(path_full)
        img = img.convert('L')
        # img.save(os.path.join(path_full, "%04d.png" % (cnt)))
        img.save(os.path.join(path_full, "%s.png" % (t)))
        