import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import torchvision.transforms as TF

import numpy as np
import torch

from utils.patchApplier import PatchTransformer, PatchApplier
from utils.utils import setup_seed
import cv2

setup_seed(7)

def findAllFile(patch):
    l = []
    for root, ds, fs in os.walk(patch):
        for filename in fs:
            l.append(root+filename)
    return l

def main(opt):
    OUTPUT_DIR = opt.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    patch_path = opt.patch_path    
    
    patch = cv2.imread(patch_path)  # BGR
    # Convert
    patch = patch.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    patch = np.ascontiguousarray(patch)
    patch = torch.from_numpy(patch).cuda()
    #print(patch.dtype)
    patch = patch / 255
    patch = patch.unsqueeze(0)
    
    patch_transformer = PatchTransformer(ratio_h=opt.ratio_h, ratio_w=opt.ratio_w).cuda()
    patch_applier = PatchApplier().cuda()    
    
    img_path_list = findAllFile(opt.image_path)
    
    for img_path in img_path_list:
        im = cv2.imread(img_path)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).cuda()
        im = im / 255
        im = im.unsqueeze(0)
        
        ann_patch = opt.label_path + img_path.split('/')[-1][0:-3] + 'txt'
        
        label = []
        with open(ann_patch) as t:
            c = t.read().strip().splitlines()
            for line in c:
                list_tmp = [0.0]
                for number in line.split(' '):
                    list_tmp.append(float(number))
                label.append(list_tmp)
            label = torch.from_numpy(np.array(label)).cuda()

        patch_tf, patch_mask_tf = patch_transformer(patch, label, im)
        imgWithPatch = patch_applier(im, patch_tf, patch_mask_tf)
        
        img_save = imgWithPatch[0]
        im = TF.ToPILImage()(img_save)
        im.save(OUTPUT_DIR+img_path.split('/')[-1])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_path', type=str, default='', help='patch file path')
    parser.add_argument('--ratio_h', type=float, default='0.23', help='ratio_h')
    parser.add_argument('--ratio_w', type=float, default='0.17', help='ratio_w')
    parser.add_argument('--image_path', type=str, default='', help='dataset image file path')
    parser.add_argument('--label_path', type=str, default='', help='dataset label file path')
    parser.add_argument('--output_dir', type=str, default='', help='output dir')
    opt = parser.parse_args()
    print(vars(opt))
    return opt
    
    
    
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
