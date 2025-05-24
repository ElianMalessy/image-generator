from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import DataLoader
from models import dim_size
from models.train_vae import train


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(dim_size),
        transforms.CenterCrop(dim_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    dataset = CelebA(root="./data", split='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    train(dataloader)
