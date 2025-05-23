import torch
from torchvision.utils import save_image
from vae import VAE
from ebm import EBM, langevin_dynamics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dim_size = 32
size = dim_size * dim_size * 3

def test(vae, ebm):
    x0 = torch.randn(1, dim_size*dim_size*3).to(device)
    x0 = x0.view(1, 3, 32, 32)

    z = vae(x0)
    z_prime = langevin_dynamics(z, ebm)
    x_hat = vae.decoder(z_prime)

    img = x_hat.cpu()
    save_image(img, 'output.png')


if __name__ == '__main__':
    vae = VAE()
    vae.load_state_dict(torch.load("vae_model.pth"))
    vae = vae.to(device)
    vae.eval()

    ebm = EBM()
    ebm.load_state_dict(torch.load("ebm_model.pth"))
    ebm = ebm.to(device)
    ebm.eval()

    test(vae, ebm)
