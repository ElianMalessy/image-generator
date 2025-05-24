import torch
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import DataLoader

from vae import VAE
from ebm import EBM, langevin_dynamics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dim_size = 32
size = dim_size * dim_size * 3

def train(dataloader, vae, num_epochs=20):
    ebm = EBM().to(device)
    optimizer = torch.optim.Adam(params=ebm.parameters(), lr=1e-4)

    for i in range(num_epochs):
        print(f'Epoch {i+1}/{num_epochs}')

        for images, _ in dataloader:
            x = images.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                z, _, _ = vae(x)

            x0 = torch.randn_like(images).to(device)
            
            with torch.no_grad():
                z0, _, _ = vae(x0)

            z_star = langevin_dynamics(z0, ebm)

            contrastive_divergence = ebm(z).mean() - ebm(z_star).mean()
            contrastive_divergence.backward()
            optimizer.step()


    torch.save(ebm.state_dict(), "ebm_model.pth")



if __name__ == '__main__':
    vae = VAE()
    vae.load_state_dict(torch.load("vae_model.pth"))
    vae = vae.to(device)
    vae.eval()
    vae = torch.compile(vae)

    transform = transforms.Compose([
        transforms.Resize(dim_size),
        transforms.CenterCrop(dim_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CelebA(root="./data", split='train', transform=transform)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    train(dataloader, vae)
