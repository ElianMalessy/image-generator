import torch
from models.unet import UNet
from models import device

def train(dataloader, vae, num_epochs=100, patience=5):
    unet_model = UNet().to(device)
    unet_model.load_state_dict(torch.load("unet_model.pth"))
    unet_model = unet_model.to(device)
    unet = torch.compile(unet_model, mode="max-autotune")
    optimizer = torch.optim.Adam(params=unet_model.parameters(), lr=1e-4)

    num_batches = len(dataloader)
    best_loss = float('inf')
    epochs_no_improve = 0

    T = 1000
    betas = torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0).to(device)


    scaler = torch.GradScaler()
    for i in range(num_epochs):
        print(f'Epoch {i+1}/{num_epochs}')
        epoch_loss = 0.0

        for images, _ in dataloader:
            x = images.to(device)
            B = x.size(0)

            optimizer.zero_grad()

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    mu, logvar = vae.encode(x)
                z = vae.reparameterize(mu, logvar)

            t = torch.randint(0, T, (x.size(0),), device=device)
            eps = torch.randn_like(z)
            a_bar = alpha_bars[t].view(-1, 1, 1, 1)
            zt = a_bar.sqrt() * z + (1 - a_bar).sqrt() * eps
            t_norm = t.float() / (T - 1) 

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                eps_pred = unet(zt, t_norm)

            loss = ((eps_pred - eps)**2).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() / B


        avg_loss = epoch_loss / num_batches
        print(f"Average loss: {avg_loss}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(unet_model.state_dict(), "unet_model.pth")
            print("Model improved and saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

        print()
