import torch
from models.vae import VAE
from models import device


def train(dataloader, num_epochs=100, patience=5):
    vae_model = VAE().to(device)
    vae_model.load_state_dict(torch.load("vae_model.pth"))
    vae_model = vae_model.to(device)
    vae = torch.compile(vae_model, mode="max-autotune")

    optimizer = torch.optim.Adam(params=vae_model.parameters(), lr=1e-4)
    num_batches = len(dataloader)

    best_elbo = float('inf')
    epochs_no_improve = 0

    scaler = torch.GradScaler()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        epoch_elbo = 0.0
        epoch_reconstruction = 0.0
        epoch_kl_div = 0.0

        for images, _ in dataloader:
            x = images.to(device)
            B = x.size(0)
            optimizer.zero_grad()

            with torch.no_grad():
                noise = torch.randn_like(x) * 0.1
                x_noisy = torch.clamp(x + noise, -1, 1)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                x_hat, mu, log_var = vae(x_noisy)

            reconstruction_loss = torch.sum((x - x_hat)**2, dim=[1,2,3]).mean()
            kl_div = torch.sum(0.5 * (mu.pow(2) + torch.exp(log_var) - log_var - 1), dim=1).mean()

            ELBO = reconstruction_loss + kl_div

            scaler.scale(ELBO).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_elbo += ELBO.item() / B
            epoch_reconstruction += reconstruction_loss.item() / B
            epoch_kl_div += kl_div.item() / B

        avg_elbo = epoch_elbo / num_batches
        avg_reconstruction = epoch_reconstruction / num_batches
        avg_kl_div = epoch_kl_div / num_batches

        print(f"Average reconstruction loss: {avg_reconstruction}")
        print(f"Average KL divergence: {avg_kl_div}")
        print(f"Average ELBO: {avg_elbo}")

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

