"""
gan_baseline.py

Generative Adversarial Network (GAN) baseline for generating synthetic EEG segments.
Trains a Generator and Discriminator in adversarial fashion on real EEG segments.

Architecture:
    Generator: latent noise vector → synthetic EEG segment
    Discriminator: EEG segment → real or fake probability
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------------
# 1. Generator Network
# ----------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        """
        Parameters:
            latent_dim (int): size of input noise vector
            output_dim (int): flattened size of one EEG segment (channels * time)
        """
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


# ----------------------------------------------------------------------
# 2. Discriminator Network
# ----------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        """
        Parameters:
            input_dim (int): flattened size of one EEG segment (channels * time)
        """
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# ----------------------------------------------------------------------
# 3. Prepare EEG Data for Training
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

    # normalize to [-1, 1] range (matches Generator's Tanh output)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    flat = (flat - mean) / std
    flat = np.clip(flat, -1, 1)

    tensor = torch.tensor(flat)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return loader, flat.shape[1], mean, std


# ----------------------------------------------------------------------
# 4. Train the GAN
# ----------------------------------------------------------------------

def train_gan(segments, latent_dim=64, epochs=50, batch_size=32, learning_rate=2e-4):
    """
    Trains a GAN on real EEG segments.

    Parameters:
        segments (list of np.ndarray): each shape (channels, time)
        latent_dim (int): size of the noise vector
        epochs (int): number of training epochs
        batch_size (int): batch size
        learning_rate (float): learning rate for both networks

    Returns:
        generator (Generator): trained generator model
        mean (np.ndarray): normalization mean used during training
        std (np.ndarray): normalization std used during training
        segment_shape (tuple): original shape of one segment (channels, time)
    """
    segment_shape = segments[0].shape
    loader, input_dim, mean, std = prepare_data(segments, batch_size)

    generator = Generator(latent_dim=latent_dim, output_dim=input_dim)
    discriminator = Discriminator(input_dim=input_dim)

    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    print(f"Training GAN | input_dim={input_dim} | latent_dim={latent_dim} | epochs={epochs}")

    for epoch in range(epochs):
        total_d_loss = 0
        total_g_loss = 0

        for (real_batch,) in loader:
            batch_size_actual = real_batch.size(0)

            real_labels = torch.ones(batch_size_actual, 1)
            fake_labels = torch.zeros(batch_size_actual, 1)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()

            # real data loss
            real_output = discriminator(real_batch)
            d_loss_real = criterion(real_output, real_labels)

            # fake data loss
            z = torch.randn(batch_size_actual, latent_dim)
            fake_batch = generator(z).detach()
            fake_output = discriminator(fake_batch)
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_g.zero_grad()

            z = torch.randn(batch_size_actual, latent_dim)
            fake_batch = generator(z)
            fake_output = discriminator(fake_batch)

            # generator wants discriminator to classify fakes as real
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        if (epoch + 1) % 10 == 0:
            avg_d = total_d_loss / len(loader)
            avg_g = total_g_loss / len(loader)
            print(f"  Epoch {epoch + 1}/{epochs} | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f}")

    return generator, mean, std, segment_shape


# ----------------------------------------------------------------------
# 5. Generate Synthetic EEG Segments
# ----------------------------------------------------------------------

def generate_synthetic_segments(generator, mean, std, segment_shape,
                                  n_samples=50, latent_dim=64):
    """
    Generates synthetic EEG segments by sampling from the GAN generator.

    Parameters:
        generator (Generator): trained generator model
        mean (np.ndarray): normalization mean from training
        std (np.ndarray): normalization std from training
        segment_shape (tuple): (channels, time) shape of output segments
        n_samples (int): number of synthetic segments to generate
        latent_dim (int): size of the noise vector

    Returns:
        list of np.ndarray: each shape (channels, time)
    """
    generator.eval()

    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        generated = generator(z).numpy()

    # denormalize
    generated = generated * std + mean

    synthetic_segments = []
    for i in range(n_samples):
        segment = generated[i].reshape(segment_shape)
        synthetic_segments.append(segment)

    return synthetic_segments


# ----------------------------------------------------------------------
# 6. Save and Load Model
# ----------------------------------------------------------------------

def save_gan(generator, path="data/synthetic/gan_generator.pt"):
    """Saves trained GAN generator weights to disk."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(generator.state_dict(), path)
    print(f"Generator saved to {path}")


def load_gan(latent_dim, output_dim, path="data/synthetic/gan_generator.pt"):
    """Loads a previously trained GAN generator from disk."""
    generator = Generator(latent_dim=latent_dim, output_dim=output_dim)
    generator.load_state_dict(torch.load(path))
    generator.eval()
    return generator


# ----------------------------------------------------------------------
# 7. Quick test / entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing GAN with simulated EEG segments...")

    n_channels = 23
    n_time = 2560
    n_segments = 100

    real_segments = [np.random.randn(n_channels, n_time).astype(np.float32)
                     for _ in range(n_segments)]

    generator, mean, std, segment_shape = train_gan(
        real_segments,
        latent_dim=64,
        epochs=50,
        batch_size=32,
    )

    synthetic = generate_synthetic_segments(
        generator, mean, std, segment_shape, n_samples=20
    )

    print(f"\nGenerated {len(synthetic)} synthetic segments")
    print(f"Segment shape: {synthetic[0].shape}")
    print(f"Value range: [{synthetic[0].min():.4f}, {synthetic[0].max():.4f}]")

    save_gan(generator)