import torch
from models.flow import Flow
from models import device

def train(dataloader, vae, num_epochs=100, patience=5):
    flow_model = Flow().to(device)
    flow = torch.compile(flow_model, mode="max-autotune")
    optimizer = torch.optim.Adam(params=flow_model.parameters(), lr=1e-4)

    num_batches = len(dataloader)
    best_loss = float('inf')
    epochs_no_improve = 0

    for i in range(num_epochs):
        print(f'Epoch {i+1}/{num_epochs}')
        epoch_loss = 0.0

        for images, labels in dataloader:
            x = images.to(device)
            y = labels.to(device).float()
            B = x.size(0)

            optimizer.zero_grad()

            with torch.no_grad():
                mu, logvar = vae.encode(x)
                z = vae.reparameterize(mu, logvar)
                z0 = torch.randn_like(z, device=device)


            t = torch.rand(B, device=device).unsqueeze(-1)
            z_interpolated = (1-t)*z0 + (t*z)
            v_hat = flow(z_interpolated, t, y)
            v = z - z0

            loss = torch.mean((v - v_hat)**2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() / B


        avg_loss = epoch_loss / num_batches
        print(f"Average loss: {avg_loss}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(flow_model.state_dict(), "flow_model.pth")
            print("Model improved and saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

        print()
