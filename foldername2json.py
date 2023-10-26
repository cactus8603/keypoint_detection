from glob import glob
import os
import json

data_path = "/data/pin/dataset/SplitFontFileAugData1"
filename = "./cfgs/char_classes_17350.json"

idx = 0
img_dict = {}

imgs = glob(os.path.join(data_path, '*'))
imgs.sort()
for img in imgs:
    basename = os.path.basename(img).split('.')[0]
    # print(basename)
    img_dict[str(idx)] = str(basename)
    idx += 1
    
# print(imgs)
print(img_dict)
json_file = json.dumps(img_dict, indent=4)
with open(filename, 'w') as f:
    f.write(json_file)

# data = json.load(open('./cfgs/font_class.json'))
# data = json.dumps(data, indent=4)

# with open('./cfgs/font_class_142.json', 'w') as f:
#     f.write(data)
