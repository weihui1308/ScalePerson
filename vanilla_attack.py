import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import shutil

try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import yolov5.val as validate  # for end-of-epoch mAP
from yolov5.models.experimental import attempt_load
from yolov5.models.yolo import Model
from yolov5.utils.autoanchor import check_anchors
from yolov5.utils.autobatch import check_train_batch_size
from yolov5.utils.callbacks import Callbacks
from yolov5.utils.dataloaders import create_dataloader
from yolov5.utils.downloads import attempt_download, is_url
from yolov5.utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from yolov5.utils.loggers import Loggers
from yolov5.utils.loggers.comet.comet_utils import check_comet_resume
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.metrics import fitness
from yolov5.utils.plots import plot_evolve
from yolov5.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)
from yolov5.models.common import DetectMultiBackend
                               
# -------------------------------------------
from utils import PatchTransformer, PatchApplier, setup_seed
import val_patch
import torchvision.transforms as TF
from matplotlib import pyplot as plt
from loss import TotalVariation, NPSCalculator
# -------------------------------------------

# ------------------
setup_seed(2023)
# ------------------
RESULTS_DIR = 'results'

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'config/dataset.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default="/home/yolov5s.pt", help='model path(s)')
    parser.add_argument('--cfg', type=str, default='yolov5/models/yolov5s.yaml', help='model.yaml')
    parser.add_argument('--hyp', type=str, default=ROOT / 'yolov5/data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--batch_size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    opt = parser.parse_args() 
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Adversarial patch
    patch_w, patch_h = 300, 300
    adv_patch = torch.rand(3, patch_w, patch_h).to(device)  # rgb
    adv_patch.requires_grad_(True)
    
    # Create detection model
    hyp = opt.hyp
    hyp = 'yolov5/data/hyps/hyp.scratch-low.yaml'
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    weights = opt.weights
    print(opt.weights)
    ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=80, anchors=hyp.get('anchors')).to(device)  # create
    exclude = []   # exclude keys
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load
    LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    
    
    # Optimizer
    optimizer_patch = torch.optim.Adam([adv_patch], lr = 1e-2, amsgrad=True)
    
    scheduler_factory = lambda x: lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
    scheduler_patch = scheduler_factory(optimizer_patch)   
        
    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    # Batch size
    batch_size = opt.batch_size

    # Dataloader
    data_dict = check_dataset(opt.data)
    print(data_dict)
    train_path, val_path = data_dict['train'], data_dict['val']
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              opt.batch_size // WORLD_SIZE,
                                              gs,
                                              opt.single_cls,
                                              hyp=opt.hyp,
                                              augment=False,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=opt.workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    val_loader = create_dataloader(val_path,
                                   imgsz,
                                   opt.batch_size // WORLD_SIZE * 2,
                                   gs,
                                   opt.single_cls,
                                   hyp=opt.hyp,
                                   cache=None if opt.noval else opt.cache,
                                   rect=True,
                                   rank=-1,
                                   workers=opt.workers * 2,
                                   pad=0.5,
                                   prefix=colorstr('val: '))[0]
    
    model.eval()
    
    nb = len(train_loader)
    
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
    
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    best_fitness = 100.0
    
    # tv loss
    total_variation = TotalVariation().cuda()
    # nps loss
    nps_calculator = NPSCalculator(printability_file="/home/30values.txt", patch_side=300).cuda()

    patch_mloss_list = []
    patch_mOBJloss_list = []
    patch_mTVloss_list = []
    
    for epoch in range(opt.epochs):  # epoch ------------------------------------------------------------------
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'loss', 'TV_loss', 'OBJ_loss', 'labels', 'img_size'))
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        patch_mTVloss = torch.zeros(1, device=device)  # mean tvlosses
        patch_mloss = torch.zeros(1, device=device)  # mean tvlosses
        patch_mOBJloss = torch.zeros(1, device=device)  # mean tvlosses
               
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------        

            targets = targets.to(device)
            imgs = imgs.to(device, non_blocking=True).float() / 255      # (n, c, h, w)
            
            optimizer_patch.zero_grad()
            
            patch_tf, patch_mask_tf = patch_transformer(adv_patch, targets, imgs)
            imgWithPatch = patch_applier(imgs, patch_tf, patch_mask_tf)
            out, _ = model(imgWithPatch)
            obj_confidence = out[:, :, 4] * out[:, :, 5]
            max_obj_confidence, _ = torch.max(obj_confidence, dim=1)
            obj_loss = torch.mean(max_obj_confidence)
            tv_loss = total_variation(adv_patch)
            nps_loss = nps_calculator(adv_patch)

            patch_loss = obj_loss * 10 + tv_loss * 20 + nps_loss * 10
            # print('patch_loss: ', patch_loss)
            patch_loss.backward() 
            optimizer_patch.step()
            adv_patch.data.clamp_(0, 1)
            
            patch_mloss = (patch_mloss * i + patch_loss.detach()) / (i + 1)
            patch_mOBJloss = (patch_mOBJloss * i + obj_loss.detach()) / (i + 1)
            patch_mTVloss = (patch_mTVloss * i + tv_loss.detach()) / (i + 1)

            # ---------------------------------------------------------------

            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                                     (f'{epoch}/{opt.epochs - 1}', mem, patch_mloss, patch_mTVloss, patch_mOBJloss, targets.shape[0], imgs.shape[-1]))

        
        patch_mloss_list.append(patch_mloss.detach().cpu().item())
        patch_mOBJloss_list.append(patch_mOBJloss.detach().cpu().item())
        patch_mTVloss_list.append(patch_mTVloss.detach().cpu().item())
        
        # val
        results, maps, _ = val_patch.run(data_dict,
                                         patch=adv_patch,
                                         batch_size=opt.batch_size // WORLD_SIZE * 2,
                                         imgsz=imgsz,
                                         model=model,
                                         single_cls=opt.single_cls,
                                         dataloader=val_loader,
                                         plots=False
                                         )
        
        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi < best_fitness:
            best_fitness = fi
        
        if best_fitness == fi:
            #print(best_fitness)
            img_save = adv_patch.data
            im = TF.ToPILImage()(img_save)
            im.save(RESULTS_DIR+'/'+'best.png')
            print('---update the best patch in epoch: ---> ', epoch)
        if epoch % 20 == 0:
            img_save = adv_patch.data
            im = TF.ToPILImage()(img_save)
            im.save(RESULTS_DIR+'/'+str(epoch)+".png")
        
        with open('results/log.txt', 'a+') as f:
            f.write(f'patch_mOBJloss: {patch_mOBJloss}, mAP: {fi[0]:4f},  epoch: {epoch}\n')

    
    plt.clf()
    plt.plot(patch_mloss_list)
    plt.savefig(f'{RESULTS_DIR}/patch_mloss.pdf')
    
    plt.clf()
    plt.plot(patch_mOBJloss_list)
    plt.savefig(f'{RESULTS_DIR}/patch_mOBJloss.pdf')
    
    plt.clf()
    plt.plot(patch_mTVloss_list)
    plt.savefig(f'{RESULTS_DIR}/patch_mTVloss.pdf')
    plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
