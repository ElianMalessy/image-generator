import torch
from models.vqvae import VAE, initialize_codebook_kmeans
from models import device


def train(dataloader, num_epochs=100, patience=5):
    vae_model = VAE().to(device)
    vae = torch.compile(vae_model, mode="max-autotune")
    initialize_codebook_kmeans(vae, dataloader, device)

    optimizer = torch.optim.Adam(
        params=list(vae_model.encoder.parameters()) + list(vae_model.decoder.parameters()),
        lr=1e-4
    )
    vq_optimizer = torch.optim.Adam(params=[vae_model.vq.embeddings], lr=1e-5)

    num_batches = len(dataloader)

    best_elbo = float('inf')
    epochs_no_improve = 0

    scaler = torch.GradScaler()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        epoch_loss = 0.0
        epoch_reconstruction = 0.0

        for images, _ in dataloader:
            x = images.to(device)
            B = x.size(0)
            optimizer.zero_grad()

            # with torch.no_grad():
            #     noise = torch.randn_like(x) * 0.1
            #     x_noisy = torch.clamp(x + noise, -1, 1)
            # x_hat, mu, log_var = vae(x_noisy)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                x_hat, vq_loss = vae(x)
                reconstruction_loss = torch.sum((x - x_hat)**2, dim=[1,2,3]).mean()
                loss = reconstruction_loss + vq_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(vq_optimizer)
            scaler.update()


            epoch_loss += loss.item() / B
            epoch_reconstruction += reconstruction_loss.item() / B

        avg_elbo = epoch_loss / num_batches
        avg_reconstruction = epoch_reconstruction / num_batches

        print(f"Average reconstruction loss: {avg_reconstruction}")
        print(f"Average loss: {avg_elbo}")

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

