import torch
from torchvision.utils import save_image
from torchvision.datasets import CelebA
from torchvision import transforms

from models.vae import VAE
from models.flow import Flow
# from models.ebm import EBM, langevin_dynamics
from models import dim_size


import argparse

parser = argparse.ArgumentParser(description="Image Generator")
parser.add_argument('--reconstruct', action='store_true')
parser.add_argument('--generate', action='store_true')
parser.add_argument('--flow', action='store_true')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unnormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    return tensor * std + mean

labels = [ '5_o_Clock_Shadow ','Arched_Eyebrows ','Attractive ','Bags_Under_Eyes ','Bald ','Bangs ','Big_Lips ','Big_Nose ','Black_Hair ','Blond_Hair ','Blurry ','Brown_Hair ','Bushy_Eyebrows ','Chubby ','Double_Chin ','Eyeglasses ','Goatee ','Gray_Hair ','Heavy_Makeup ','High_Cheekbones ','Male ','Mouth_Slightly_Open ','Mustache ','Narrow_Eyes ','No_Beard ','Oval_Face ','Pale_Skin ','Pointy_Nose ','Receding_Hairline ','Rosy_Cheeks ','Sideburns ','Smiling ','Straight_Hair ','Wavy_Hair ','Wearing_Earrings ','Wearing_Hat ','Wearing_Lipstick ','Wearing_Necklace ','Wearing_Necktie ','Young ']

def sparse_random_labels(num_attrs=40, num_active=1):
    # Choose `num_active` indices to be 1
    idx = torch.randperm(num_attrs)[:num_active]
    label = torch.full((num_attrs,), -1, dtype=torch.int)
    label[idx] = 1
    return label

def test(vae, flow=None, x0=None):
    if x0 is not None:
        x0 = x0.view(1, 3, dim_size, dim_size)

        mu, logvar = vae.encode(x0)
        z = vae.reparameterize(mu, logvar)
    else:
        z = torch.randn(1, 4, 16, 16).to(device)
        # z = torch.randn(1, 64).to(device)



    x_hat = vae.decode(z)
    img = unnormalize(x_hat.cpu())
    save_image(img, 'output1.png')

    if flow:
        rand_labels = sparse_random_labels()
        rand_labels = rand_labels.to(device).float().unsqueeze(0)
        z = flow.transform(z, rand_labels)
        x_hat = vae.decode(z)

        img = unnormalize(x_hat.cpu())
        save_image(img, 'output2.png')
        active_attrs = [i for i in range(len(labels)) if rand_labels[0, i] == 1]
        for i in active_attrs:
            print(labels[i], end=', ')


if __name__ == '__main__':
    vae = VAE()
    vae.load_state_dict(torch.load("vae_model.pth"))
    vae = vae.to(device)
    vae.eval()

    flow = None
    if parser.parse_args().flow:
        flow = Flow()
        flow.load_state_dict(torch.load("flow_model.pth"))
        flow = flow.to(device)
        flow.eval()

    # ebm = EBM()
    # ebm.load_state_dict(torch.load("ebm_model.pth"))
    # ebm = ebm.to(device)
    # ebm.eval()

    if parser.parse_args().reconstruct:
        transform = transforms.Compose([
            transforms.Resize(dim_size),
            transforms.CenterCrop(dim_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = CelebA(root="./data", split='train', transform=transform)
        x0 = dataset[0][0].to(device)

        test(vae, flow, x0)

    elif parser.parse_args().generate:
        test(vae, flow)

