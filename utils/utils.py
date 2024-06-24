import torch
import torch.nn as nn
import torchvision.transforms as TF
import cv2
import os
import numpy as np
import random
import kornia.geometry.transform as KT  
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torchvision.utils import save_image
from PIL import Image
from torchvision import models


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
