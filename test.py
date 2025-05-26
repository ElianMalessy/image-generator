import torch
from torchvision.utils import save_image
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchvision import transforms

from models.vae import VAE
from models.flow import Flow
# from models.ebm import EBM, langevin_dynamics
from models import dim_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unnormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    return tensor * std + mean

def test(vae, flow, x0=None):
    if x0 is not None:
        x0 = x0.view(1, 3, dim_size, dim_size)

        mu, logvar = vae.encode(x0)
        z = vae.reparameterize(mu, logvar)
    else:
        z = torch.randn(1, 64, 4, 4).to(device)



    x_hat = vae.decode(z)
    img = unnormalize(x_hat.cpu())
    save_image(img, 'output1.png')

    # z = flow.transform(z)
    # x_hat = vae.decoder(z)
    #
    # img = unnormalize(x_hat.cpu())
    # save_image(img, 'output2.png')


if __name__ == '__main__':
    vae = VAE()
    vae.load_state_dict(torch.load("vae_model.pth"))
    vae = vae.to(device)
    vae.eval()

    flow = Flow()
    flow.load_state_dict(torch.load("flow_model.pth"))
    flow = flow.to(device)
    flow.eval()

    # ebm = EBM()
    # ebm.load_state_dict(torch.load("ebm_model.pth"))
    # ebm = ebm.to(device)
    # ebm.eval()

    # transform = transforms.Compose([
    #     transforms.Resize(dim_size),
    #     transforms.CenterCrop(dim_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # dataset = CelebA(root="./data", split='train', transform=transform)
    # x0 = dataset[0][0].to(device)


    # test(vae, flow, x0)
    test(vae, flow)
