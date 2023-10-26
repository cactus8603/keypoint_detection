import os
import shutil
import argparse
from tqdm import tqdm
from glob import glob

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_spilt_file", default='/code/Font/cfgs/spilt_data/byUnicode/train.txt', help='path to file recording train unicode/font')
    parser.add_argument("--val_spilt_file", default='/code/Font/cfgs/spilt_data/byUnicode/val.txt', help='path to file recordig val unicode/font')
    parser.add_argument("--src_path", default='/code/Font/byUnicode', help='path to source dataset')
    parser.add_argument("--train_dst_path", default='/code/Font/spilt/byUnicode/train', help='path to dst train dataset')
    parser.add_argument("--val_dst_path", default='/code/Font/spilt/byUnicode/val', help='path to dst val dataset')
    parser.add_argument("--total_types", default=4807, type=int, help='number of files in src_path, 170 for byUnicode, 4807 for byFont')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfgs = get_parser()

    with open(cfgs.train_spilt_file, 'r') as f:
        train = f.readlines()
        train = [v.strip() for v in train]

    # print(train[0])

    with open(cfgs.val_spilt_file, 'r') as f:
        val = f.readlines()
        val = [v.strip() for v in val]

    folder = glob(os.path.join(cfgs.src_path, '*'))
    # print(folder)
    # folder = glob(os.path.join('bPOP00B', '*'))


    if not os.path.exists(cfgs.train_dst_path):
        os.mkdir(cfgs.train_dst_path)

    if not os.path.exists(cfgs.val_dst_path):
        os.mkdir(cfgs.val_dst_path)

    pbar = tqdm(total=len(folder))
    for font in folder:
        train_chr = 0
        val_chr = 0

        chars = glob(os.path.join(font, '*'))
        font_basename = os.path.basename(font)
        folder_name = os.path.splitext(font_basename)[0]

        if not os.path.exists(os.path.join(cfgs.train_dst_path, folder_name)):
            os.mkdir(os.path.join(cfgs.train_dst_path, folder_name))
        train_save_path = os.path.join(cfgs.train_dst_path, folder_name)

        if not os.path.exists(os.path.join(cfgs.val_dst_path, folder_name)):
            os.mkdir(os.path.join(cfgs.val_dst_path, folder_name))
        val_save_path = os.path.join(cfgs.val_dst_path, folder_name)

        # tmp = 0
        
        for uni in chars:
            basename = os.path.basename(uni)
            # print(basename)
            unicode = os.path.splitext(basename)[0]
            # print(unicode)
            # break
            if unicode in train:
                dst_path = os.path.join(train_save_path, basename)
                shutil.copyfile(uni, dst_path)
                train_chr += 1

            if unicode in val:
                dst_path = os.path.join(val_save_path, basename)
                shutil.copyfile(uni, dst_path)
                val_chr += 1
            
        pbar.set_description()
        pbar.update(1)

        # print(train_chr, val_chr, train_chr+val_chr)
        # if fontname not in data:
    #         shutil.rmtree(font)
    #         tmp += 1

    # print(len(data))
    # print(tmp)

    # shutil.rmtree('123')





