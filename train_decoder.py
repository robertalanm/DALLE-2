import torch
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms as T
from pathlib import Path
import os
from tqdm import tqdm
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, OpenAIClipAdapter
from dalle2_pytorch.tokenizer import SimpleTokenizer
from dalle2_pytorch.optimizer import get_optimizer
from torchvision.datasets.coco import CocoCaptions