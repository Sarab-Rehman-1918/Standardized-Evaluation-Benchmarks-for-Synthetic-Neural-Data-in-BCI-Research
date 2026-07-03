"""
diffusion_baseline.py

Denoising Diffusion Probabilistic Model (DDPM) baseline for generating
synthetic EEG segments. Implements a lightweight diffusion model with
reduced timesteps to keep it trainable on CPU.

Process:
    Forward: gradually adds Gaussian noise to real EEG over T timesteps
    Reverse: trains a network to predict and remove noise step by step
    Generation: starts from pure noise, iteratively denoises to get synthetic EEG
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------------
# 1. Noise Schedule
# ----------------------------------------------------------------------

def make_beta_schedule(timesteps=100, beta_start=1e-4, beta_end=0.02):
    """
    Creates a linear noise schedule (beta values) for the diffusion process.

    Parameters:
        timesteps (int): total number of diffusion steps
        beta_start (float): starting noise level
        beta_end (float): ending noise level

    Returns:
        dict of precomputed schedule tensors
    """
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
    }


# ----------------------------------------------------------------------
# 2. Denoising Network (U-Net style MLP)
# ----------------------------------------------------------------------

class DenoisingNetwork(nn.Module):
    def __init__(self, input_dim, timesteps=100):
        """
        A lightweight MLP that predicts the noise added at each timestep.

        Parameters:
            input_dim (int): flattened EEG segment size (channels * time)
            timesteps (int): total diffusion timesteps (for time embedding)
        """
        super(DenoisingNetwork, self).__init__()

        # time embedding: maps timestep index to a learned vector
        self.time_embedding = nn.Embedding(timesteps, 128)

        self.network = nn.Sequential(
            nn.Linear(input_dim + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x_noisy, t):
        """
        Parameters:
            x_noisy (tensor): noisy EEG segment, shape (batch, input_dim)
            t (tensor): timestep indices, shape (batch,)

        Returns:
            tensor: predicted noise, shape (batch, input_dim)
        """
        t_emb = self.time_embedding(t)
        x_input = torch.cat([x_noisy, t_emb], dim=1)
        return self.network(x_input)


# ----------------------------------------------------------------------
# 3. Forward Diffusion: add noise to real data
# ----------------------------------------------------------------------

def forward_diffusion(x_0, t, schedule):
    """
    Adds noise to a clean EEG segment at timestep t.

    Parameters:
        x_0 (tensor): clean segment, shape (batch, input_dim)
        t (tensor): timestep indices, shape (batch,)
        schedule (dict): precomputed noise schedule

    Returns:
        x_noisy (tensor): noisy segment at timestep t
        noise (tensor): the actual noise that was added
    """
    noise = torch.randn_like(x_0)

    sqrt_alpha = schedule["sqrt_alphas_cumprod"][t].unsqueeze(1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alphas_cumprod"][t].unsqueeze(1)

    x_noisy = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    return x_noisy, noise


# ----------------------------------------------------------------------
# 4. Prepare EEG Data for Training
# ----------------------------------------------------------------------

def prepare_data(segments, batch_size=32):
    """
    Converts a list of EEG segment arrays into a PyTorch DataLoader.

    Parameters:
        segments (list of np.ndarray): each shape (channels, time)
        batch_size (int): training batch size

    Returns:
        DataLoader, input_dim, mean, std
    """
    flat = np.array([seg.flatten() for seg in segments], dtype=np.float32)

    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    flat = (flat - mean) / std

    tensor = torch.tensor(flat)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return loader, flat.shape[1], mean, std


# ----------------------------------------------------------------------
# 5. Train the Diffusion Model
# ----------------------------------------------------------------------

def train_diffusion(segments, timesteps=100, epochs=30,
                     batch_size=32, learning_rate=1e-3):
    """
    Trains a DDPM on real EEG segments.

    Parameters:
        segments (list of np.ndarray): each shape (channels, time)
        timesteps (int): number of diffusion steps
        epochs (int): training epochs
        batch_size (int): batch size
        learning_rate (float): learning rate

    Returns:
        model (DenoisingNetwork): trained denoising network
        schedule (dict): noise schedule used during training
        mean (np.ndarray): normalization mean
        std (np.ndarray): normalization std
        segment_shape (tuple): original shape of one segment
    """
    segment_shape = segments[0].shape
    loader, input_dim, mean, std = prepare_data(segments, batch_size)

    schedule = make_beta_schedule(timesteps=timesteps)
    model = DenoisingNetwork(input_dim=input_dim, timesteps=timesteps)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print(f"Training Diffusion Model | input_dim={input_dim} | "
          f"timesteps={timesteps} | epochs={epochs}")

    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for (batch,) in loader:
            batch_size_actual = batch.size(0)

            # sample random timesteps for each item in batch
            t = torch.randint(0, timesteps, (batch_size_actual,), dtype=torch.long)

            # forward diffusion: add noise
            x_noisy, noise = forward_diffusion(batch, t, schedule)

            # predict the noise
            predicted_noise = model(x_noisy, t)

            # loss: how well did we predict the noise
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.6f}")

    return model, schedule, mean, std, segment_shape


# ----------------------------------------------------------------------
# 6. Reverse Diffusion: generate synthetic EEG
# ----------------------------------------------------------------------

@torch.no_grad()
def generate_synthetic_segments(model, schedule, mean, std,
                                  segment_shape, n_samples=50, timesteps=100):
    """
    Generates synthetic EEG by iteratively denoising from pure Gaussian noise.

    Parameters:
        model (DenoisingNetwork): trained denoising network
        schedule (dict): noise schedule
        mean (np.ndarray): normalization mean from training
        std (np.ndarray): normalization std from training
        segment_shape (tuple): (channels, time) shape of output segments
        n_samples (int): number of synthetic segments to generate
        timesteps (int): number of diffusion steps

    Returns:
        list of np.ndarray: each shape (channels, time)
    """
    model.eval()
    input_dim = segment_shape[0] * segment_shape[1]

    # start from pure noise
    x = torch.randn(n_samples, input_dim)

    # iteratively denoise from T → 0
    for t_idx in reversed(range(timesteps)):
        t_tensor = torch.full((n_samples,), t_idx, dtype=torch.long)

        predicted_noise = model(x, t_tensor)

        alpha = schedule["alphas"][t_idx]
        alpha_cumprod = schedule["alphas_cumprod"][t_idx]
        beta = schedule["betas"][t_idx]

        # compute denoised estimate
        x = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
        )

        # add noise for all steps except the last
        if t_idx > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta) * noise

    # denormalize
    generated = x.numpy()
    generated = generated * std + mean

    synthetic_segments = []
    for i in range(n_samples):
        segment = generated[i].reshape(segment_shape)
        synthetic_segments.append(segment)

    return synthetic_segments


# ----------------------------------------------------------------------
# 7. Save and Load Model
# ----------------------------------------------------------------------

def save_diffusion(model, path="data/synthetic/diffusion_model.pt"):
    """Saves trained diffusion model weights to disk."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Diffusion model saved to {path}")


def load_diffusion(input_dim, timesteps=100,
                    path="data/synthetic/diffusion_model.pt"):
    """Loads a previously trained diffusion model from disk."""
    model = DenoisingNetwork(input_dim=input_dim, timesteps=timesteps)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# ----------------------------------------------------------------------
# 8. Quick test / entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing Diffusion Model with simulated EEG segments...")

    n_channels = 23
    n_time = 2560
    n_segments = 100

    real_segments = [np.random.randn(n_channels, n_time).astype(np.float32)
                     for _ in range(n_segments)]

    model, schedule, mean, std, segment_shape = train_diffusion(
        real_segments,
        timesteps=100,
        epochs=30,
        batch_size=32,
    )

    print("\nGenerating synthetic segments...")
    synthetic = generate_synthetic_segments(
        model, schedule, mean, std, segment_shape,
        n_samples=20, timesteps=100
    )

    print(f"\nGenerated {len(synthetic)} synthetic segments")
    print(f"Segment shape: {synthetic[0].shape}")
    print(f"Value range: [{synthetic[0].min():.4f}, {synthetic[0].max():.4f}]")

    save_diffusion(model)