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
    # openai pretrained clip - defaults to ViT/B-32
    OpenAIClip = OpenAIClipAdapter()

    unet = Unet(
        dim = 128,
        image_embed_dim = 512,
        cond_dim = 128,
        channels = 3,
        dim_mults=(1, 2, 4, 8)
    ).to(device)

    # decoder, which contains the unet and clip

    decoder = Decoder(
        unet = unet,
        clip = OpenAIClip,
        timesteps = 100,
        image_cond_drop_prob = 0.1,
        text_cond_drop_prob = 0.5,
        condition_on_text_encodings=True
    ).to(device)

    if os.path.exists(decoder_save_path):
        dec = torch.load(decoder_save_path)
        decoder.load_state_dict(dec.state_dict())

    return decoder



def train(decoder, train_data):
    train_size = len(train_data)
    idx_list = range(0, train_size, batch_size)

    tokenizer = SimpleTokenizer()

    opt = get_optimizer(decoder.parameters())
    sched = ExponentialLR(opt, gamma=0.01)

    for curr_epoch in range(epoch):
        print("Run training decoder ...")
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

            loss = decoder(image, text) # this can optionally be decoder(images, text) if you wish to condition on the text encodings as well, though it was hinted in the paper it didn't do much
            opt.zero_grad()
            loss.backward()
            opt.step()

            if batch_idx != 0 and batch_idx % 100 == 0:
                torch.save(decoder, decoder_save_path)
                sched.step()
            
            if batch_idx % 1000 == 0:
                print(f"loss: {loss.data}")

    torch.save(decoder, decoder_save_path)


if __name__ == "__main__":
    _, train_data = create_dataset()
    decoder = create_model()
    train(decoder, train_data)