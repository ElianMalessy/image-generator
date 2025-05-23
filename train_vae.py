import torch
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from vae import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dim_size = 32
size = dim_size * dim_size * 3


def train(dataloader, num_epochs=20):
    vae_model = VAE().to(device)
    vae = torch.compile(vae_model)

    optimizer = torch.optim.Adam(params=vae_model.parameters(), lr=1e-4)

    for i in range(num_epochs):
        print(f'Epoch {i+1}/{num_epochs}')

        for images, _ in dataloader:
            x = images.to(device)
            optimizer.zero_grad()

            x_hat, mu, log_var, _ = vae(x)

            reconstruction_loss = torch.sum((x - x_hat)**2, dim=[1,2,3])
            kl_div = 0.5 * torch.sum((mu**2) + torch.exp(log_var) - log_var - 1, dim=1)
            ELBO = reconstruction_loss.mean() + kl_div.mean()

            ELBO.backward()
            optimizer.step()

    torch.save(vae_model.state_dict(), "vae_model.pth")


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(dim_size),
        transforms.CenterCrop(dim_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    dataset = CelebA(root="./data", split='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    train(dataloader)


