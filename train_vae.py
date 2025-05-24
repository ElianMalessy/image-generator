import torch
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import DataLoader
from vae import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dim_size = 32
size = dim_size * dim_size * 3


def train(dataloader, num_epochs=100, patience=5):
    vae_model = VAE().to(device)
    vae = torch.compile(vae_model)

    optimizer = torch.optim.Adam(params=vae_model.parameters(), lr=1e-4)

    num_batches = len(dataloader)

    total_steps = num_epochs * num_batches
    global_step = 0
    warmup_steps = int(0.5 * total_steps)

    best_elbo = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        epoch_elbo = 0.0

        for images, _ in dataloader:
            x = images.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                noise = torch.randn_like(x) * 0.1
                x_noisy = torch.clamp(x + noise, -1, 1)

            z, mu, log_var = vae(x)
            x_hat = vae.decoder(z)

            reconstruction_loss = torch.sum((x - x_hat)**2, dim=[1,2,3]).mean()

            ratio = global_step / warmup_steps
            beta = min(0.2, ratio)
            free_nats = max(1.0 - ratio, 0.2)

            kl_per_dim = 0.5 * (mu**2) + torch.exp(log_var) - log_var - 1
            kl_per_dim = torch.clamp(kl_per_dim - free_nats, min=0.0)
            kl_div = torch.sum(kl_per_dim, dim=1).mean() 
            ELBO = reconstruction_loss + beta * kl_div

            print(f"Reconstruction loss: {reconstruction_loss.item() / 128} KL loss:, {beta * kl_div.item() / 128}")
            ELBO.backward()
            optimizer.step()

            epoch_elbo += ELBO.item()
            global_step += 1

        avg_elbo = epoch_elbo / num_batches
        print(f"Average ELBO: {avg_elbo:.4f}")

        # Early stopping logic:
        if avg_elbo < best_elbo:
            best_elbo = avg_elbo
            epochs_no_improve = 0
            torch.save(vae_model.state_dict(), "vae_model.pth")
            print("Model improved and saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

        print()

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


