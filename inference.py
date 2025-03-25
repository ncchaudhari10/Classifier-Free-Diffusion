# generate_mnist_ddpm.py
import torch
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 32
num_classes = 10
model_path = "mnist_ddpm.pth"
num_samples = 8
guidance_scale = 3  #0 = Uncondtioned, 1 = Conditioned, 0-1 = Interpolation, >1 = Extrapolation
target_label = 3
save_path = "generated_digits.png"

# Load model
model = UNet2DModel(
    sample_size=image_size,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
    class_embed_type="timestep",
    num_class_embeds=num_classes,
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
model.eval()

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Start from pure noise samples
samples = torch.randn((num_samples, 1, image_size, image_size), device=device)

# Denoising loop with CFG
for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
    t_batch = torch.full((num_samples,), t, device=device).long()
    cond_labels = torch.full((num_samples,), target_label, device=device).long()
    uncond_labels = torch.full((num_samples,), -1, device=device).long()

    input_samples = torch.cat([samples] * 2)
    input_t = torch.cat([t_batch] * 2)
    input_labels = torch.cat([uncond_labels, cond_labels])

    with torch.no_grad():
        noise_preds = model(input_samples, input_t, class_labels=input_labels).sample
        noise_uncond, noise_cond = noise_preds.chunk(2)
        guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    samples = noise_scheduler.step(guided_noise, t, samples).prev_sample

# Rescale from [-1, 1] to [0, 1] for visualization
samples = (samples.clamp(-1, 1) + 1) / 2
samples *= 255.0
grid = samples.detach().cpu().numpy()

# Save generated images to disk
fig, axs = plt.subplots(1, num_samples, figsize=(num_samples, 1))
for i, ax in enumerate(axs):
    ax.imshow(grid[i, 0], cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"Generated image saved to {save_path}")
