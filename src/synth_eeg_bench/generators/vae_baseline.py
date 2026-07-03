"""
vae_baseline.py

Variational Autoencoder (VAE) baseline for generating synthetic EEG segments.
Trains on real EEG segments and generates new synthetic ones that can be
evaluated by the metrics suite.

Architecture:
    Encoder: flattens EEG segment → latent mean + log variance
    Decoder: samples latent vector → reconstructed EEG segment
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------------
# 1. VAE Model Architecture
# ----------------------------------------------------------------------

class EEG_VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        """
        Parameters:
            input_dim (int): flattened size of one EEG segment (channels * time)
            latent_dim (int): size of the latent space
        """
        super(EEG_VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mean, logvar


# ----------------------------------------------------------------------
# 2. VAE Loss Function
# ----------------------------------------------------------------------

def vae_loss(reconstructed, original, mean, logvar):
    """
    Combined reconstruction loss + KL divergence.
    """
    reconstruction_loss = nn.functional.mse_loss(reconstructed, original, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence


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
        DataLoader, input_dim (int)
    """
    # flatten each segment: (channels, time) → (channels * time,)
    flat = np.array([seg.flatten() for seg in segments], dtype=np.float32)

    # normalize to zero mean, unit variance
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    flat = (flat - mean) / std

    tensor = torch.tensor(flat)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, flat.shape[1], mean, std


# ----------------------------------------------------------------------
# 4. Train the VAE
# ----------------------------------------------------------------------

def train_vae(segments, latent_dim=64, epochs=20, batch_size=32, learning_rate=1e-3):
    """
    Trains a VAE on real EEG segments.

    Parameters:
        segments (list of np.ndarray): each shape (channels, time)
        latent_dim (int): latent space dimensionality
        epochs (int): number of training epochs
        batch_size (int): batch size
        learning_rate (float): learning rate

    Returns:
        model (EEG_VAE): trained model
        mean (np.ndarray): normalization mean used during training
        std (np.ndarray): normalization std used during training
        segment_shape (tuple): original shape of one segment (channels, time)
    """
    segment_shape = segments[0].shape
    loader, input_dim, mean, std = prepare_data(segments, batch_size)

    model = EEG_VAE(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training VAE | input_dim={input_dim} | latent_dim={latent_dim} | epochs={epochs}")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch)
            loss = vae_loss(reconstructed, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader.dataset)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.4f}")

    return model, mean, std, segment_shape


# ----------------------------------------------------------------------
# 5. Generate Synthetic EEG Segments
# ----------------------------------------------------------------------

def generate_synthetic_segments(model, mean, std, segment_shape,
                                  n_samples=50, latent_dim=64):
    """
    Generates synthetic EEG segments by sampling from the VAE latent space.

    Parameters:
        model (EEG_VAE): trained VAE model
        mean (np.ndarray): normalization mean from training
        std (np.ndarray): normalization std from training
        segment_shape (tuple): (channels, time) shape of output segments
        n_samples (int): number of synthetic segments to generate
        latent_dim (int): latent space dimensionality

    Returns:
        list of np.ndarray: each shape (channels, time)
    """
    model.eval()
    synthetic_segments = []

    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        generated = model.decode(z).numpy()

    # denormalize
    generated = generated * std + mean

    for i in range(n_samples):
        segment = generated[i].reshape(segment_shape)
        synthetic_segments.append(segment)

    return synthetic_segments


# ----------------------------------------------------------------------
# 6. Save and Load Model
# ----------------------------------------------------------------------

def save_vae(model, path="data/synthetic/vae_model.pt"):
    """Saves trained VAE model weights to disk."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_vae(input_dim, latent_dim=64, path="data/synthetic/vae_model.pt"):
    """Loads a previously trained VAE model from disk."""
    model = EEG_VAE(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# ----------------------------------------------------------------------
# 7. Quick test / entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing VAE with simulated EEG segments...")

    sfreq = 256
    n_channels = 23
    n_time = 2560
    n_segments = 100

    # simulate real EEG segments
    real_segments = [np.random.randn(n_channels, n_time).astype(np.float32)
                     for _ in range(n_segments)]

    # train VAE
    model, mean, std, segment_shape = train_vae(
        real_segments,
        latent_dim=64,
        epochs=20,
        batch_size=32
    )

    # generate synthetic segments
    synthetic = generate_synthetic_segments(
        model, mean, std, segment_shape, n_samples=20
    )

    print(f"\nGenerated {len(synthetic)} synthetic segments")
    print(f"Segment shape: {synthetic[0].shape}")
    print(f"Value range: [{synthetic[0].min():.4f}, {synthetic[0].max():.4f}]")

    # save model
    save_vae(model)