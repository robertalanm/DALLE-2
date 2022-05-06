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

from config.default import *


def create_dataset():
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(input_image_size),
        T.CenterCrop(input_image_size),
        T.ToTensor()
    ])

    train_data = CocoCaptions(
        root=train_img_path,
        annFile=train_annot_path,
        transform=transform
    )

    return transform, train_data


def create_model():
    OpenAIClip = OpenAIClipAdapter()

    prior_network = DiffusionPriorNetwork(
        dim = 512,
        depth = 6,
        dim_head = 64,
        heads = 8
    )

    diffusion_prior = DiffusionPrior(
        net = prior_network,
        clip = OpenAIClip,
        timesteps = 100,
        cond_drop_prob = 0.2
    ).to(device)

    if os.path.exists(diff_save_path):
        dp = torch.load(diff_save_path)
        diffusion_prior.load_state_dict(dp.state_dict())

    return diffusion_prior


def train(diffusion_prior, train_data):

    train_size = len(train_data)
    idx_list = range(0, train_size, batch_size)

    tokenizer = SimpleTokenizer()
    opt = get_optimizer(diffusion_prior.parameters())
    sched = ExponentialLR(opt, gamma=0.01)

    for curr_epoch in range(epoch):
        print("Run training diffusion prior ...")
        print(f"Epoch {curr_epoch+1} / {epoch}")
        
        for batch_idx in tqdm(idx_list):
            if (batch_idx + batch_size) > train_size - 1:
                iter_idx = range(batch_idx, train_size, 1)
            else:
                iter_idx = range(batch_idx, batch_idx+batch_size, 1)

            image_list = []
            text_list = []
            
            for curr_idx in iter_idx:
                image, target = train_data[curr_idx]
                image = image.unsqueeze(0).to(device)

                text = tokenizer.tokenize(target).to(device)

                text_size = len(text)
                for i in range(text_size):
                    image_list.append(image)
                
                text_list.append(text)

            text = torch.cat(text_list, dim=0).to(device)
            image = torch.cat(image_list, dim=0).to(device)
        
            loss = diffusion_prior(text, image)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if batch_idx != 0 and batch_idx % 100 == 0:
                torch.save(diffusion_prior, diff_save_path)
                sched.step()

            if batch_idx % 1000 == 0:
                print(f"loss: {loss.data}")

if __name__ == "__main__":
    _, train_data = create_dataset()
    diffusion_prior = create_model()
    train(diffusion_prior, train_data)
