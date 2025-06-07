import torch
from models.vae import VAE
from models import device


def train(dataloader, num_epochs=100, patience=5):
    vae_model = VAE().to(device)
    vae_model.load_state_dict(torch.load("vae_model.pth"))
    vae_model = vae_model.to(device)

    # vae = torch.compile(vae_model)
    vae = torch.compile(vae_model, mode="max-autotune")

    optimizer = torch.optim.Adam(params=vae_model.parameters(), lr=1e-4)

    num_batches = len(dataloader)
    # batch_num = 0
    # warmup_steps = num_batches * 5

    best_elbo = float('inf')
    epochs_no_improve = 0

    scaler = torch.GradScaler()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        epoch_elbo = 0.0
        epoch_reconstruction = 0.0
        epoch_kl = 0.0

        for images, _ in dataloader:
            x = images.to(device)
            B = x.size(0)
            optimizer.zero_grad()

            # with torch.no_grad():
            #     noise = torch.randn_like(x) * 0.1
            #     x_noisy = torch.clamp(x + noise, -1, 1)
            # x_hat, mu, log_var = vae(x_noisy)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                x_hat, mu, log_var = vae(x)

                reconstruction_loss = torch.sum((x - x_hat)**2, dim=[1,2,3]).mean()
                kl_per_dim = 0.5 * (mu.pow(2) + torch.exp(log_var) - log_var - 1)
                # psi = min(batch_num / warmup_steps, 1.0) * 0.5
                # kl_excess_per_dim = torch.clamp(kl_per_dim - psi, min=0.0)
                # kl_div = kl_excess_per_dim.sum(dim=[1,2,3]).mean()
                kl_div = kl_per_dim.sum(dim=[1,2,3]).mean()


                ELBO = reconstruction_loss + kl_div

            scaler.scale(ELBO).backward()
            scaler.step(optimizer)
            scaler.update()
            # ELBO.backward()
            # optimizer.step()


            epoch_elbo += ELBO.item() / B
            epoch_reconstruction += reconstruction_loss.item() / B
            epoch_kl += kl_div.item() / B
            # batch_num += 1

        avg_elbo = epoch_elbo / num_batches
        avg_reconstruction = epoch_reconstruction / num_batches
        avg_kl = epoch_kl / num_batches

        print(f"Average reconstruction loss: {avg_reconstruction}")
        print(f"Average KL loss:, {avg_kl}")
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

