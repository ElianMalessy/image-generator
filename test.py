import torch
from torchvision.utils import save_image
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchvision import transforms

from models.vae import VAE
from models.ebm import EBM, langevin_dynamics
from models import dim_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size = dim_size * dim_size * 3

def unnormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    return tensor * std + mean

def test(vae, ebm, x0=None):
    if x0 is None:
        x0 = torch.randn(1, dim_size*dim_size*3).to(device)
    else:
        img = unnormalize(x0.cpu())
        save_image(img, 'input.png')


    x0 = x0.view(1, 3, dim_size, dim_size)

    mu, logvar = vae.encode(x0)
    z = vae.reparameterize(mu, logvar)
    # z_prime = langevin_dynamics(z, ebm)
    x_hat = vae.decoder(z)

    img = unnormalize(x_hat.cpu())
    save_image(img, 'output.png')


if __name__ == '__main__':
    vae = VAE()
    vae.load_state_dict(torch.load("vae_model.pth"))
    vae = vae.to(device)
    vae.eval()

    ebm = EBM()
    # ebm.load_state_dict(torch.load("ebm_model.pth"))
    # ebm = ebm.to(device)
    # ebm.eval()

    # transform = transforms.Compose([
    #     transforms.Resize(dim_size),
    #     transforms.CenterCrop(dim_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # dataset = CelebA(root="./data", split='test', transform=transform)
    # x0 = dataset[0][0].to(device)


    # test(vae, ebm, x0)
    test(vae, ebm)
