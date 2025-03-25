# train_mnist_ddpm.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm
import os

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

image_size = 32
batch_size = 128
num_epochs = 10
num_classes = 10
drop_label_prob = 0.2
model_save_path = "mnist_ddpm.pth"

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1, 1]
])

train_dataset = datasets.MNIST(root=".", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = UNet2DModel(
    sample_size=image_size,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
    class_embed_type="timestep",
    num_class_embeds=num_classes,
).to(device)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
model.train()
for epoch in range(num_epochs):
    pbar = tqdm(train_loader)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        noise = torch.randn_like(x)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x.shape[0],), device=device).long()
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        drop_mask = torch.rand(y.shape[0], device=device) < drop_label_prob
        y_dropped = y.clone()
        y_dropped[drop_mask] = -1

        noise_pred = model(noisy_x, timesteps, class_labels=y_dropped).sample
        loss = nn.MSELoss()(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
