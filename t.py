import torch
from models.vae import VAE
from models.unet import UNet
from torchvision.utils import save_image

@torch.no_grad()
def sample_latents(z0, unet, shape=(1, 4, 16, 16), T=1000, device='cuda'):
    betas = torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0).to(device)
    

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        t_norm  = t_batch.float() / (T - 1)
        eps_pred = unet(z0, t_norm)

        a_t = alphas[t]
        a_bar_t = alpha_bars[t]
        beta_t = betas[t]

        # Predict the mean of posterior q(z_{t-1} | z_t, z_0)
        if t > 0:
            noise = torch.randn_like(z0)
        else:
            noise = torch.zeros_like(z0)

        z0 = (
            1 / a_t.sqrt() * (z0 - ((1 - a_t) / (1 - a_bar_t).sqrt()) * eps_pred)
            + beta_t.sqrt() * noise
        )

    return z0

def unnormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    return tensor * std + mean

def generate_images(unet, vae):
    shape=(1,4,16,16)
    z0 = torch.randn(shape, device=device)
    
    x_hat = vae.decode(z0)
    img = unnormalize(x_hat.cpu())
    save_image(img, 'output1.png')

    z = sample_latents(z0, unet, shape, device=device)
    x_hat = vae.decode(z)
    img = unnormalize(x_hat.cpu())
    save_image(img, 'output2.png')

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = VAE().to(device)
    unet = UNet().to(device)
    vae.load_state_dict(torch.load("vae_model.pth"))
    unet.load_state_dict(torch.load("unet_model.pth"))
    vae.eval()
    unet.eval()
    images = generate_images(unet, vae)

