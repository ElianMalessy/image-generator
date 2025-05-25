import torch
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import DataLoader

from models import dim_size, device
from models.vae import VAE
from models.train_vae import train as train_vae
from models.train_flow import train as train_flow

import argparse

parser = argparse.ArgumentParser(description="Image Generator")
parser.add_argument('--vae', action='store_true')
parser.add_argument('--flow', action='store_true')


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(dim_size),
        transforms.CenterCrop(dim_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    dataset = CelebA(root="./data", split='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    if parser.parse_args().vae:
        train_vae(dataloader)
    elif parser.parse_args().flow:
        vae = VAE()
        vae.load_state_dict(torch.load("vae_model.pth"))
        vae = vae.to(device)
        vae.eval()
        vae = torch.compile(vae)
        train_flow(dataloader, vae)
