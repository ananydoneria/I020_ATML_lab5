"""
WGAN (Wasserstein GAN) Implementation on CIFAR-10 Dataset

This script implements the WGAN architecture as described in:
"Wasserstein GAN" by Arjovsky et al. (https://arxiv.org/abs/1701.07875)

Key features:
- Wasserstein distance for stable training
- Weight clipping to enforce Lipschitz continuity
- Critic (discriminator) trained with multiple iterations per generator update
- Low learning rate to ensure stable convergence
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


class Generator(nn.Module):
    """Generator network for WGAN on CIFAR-10"""
    
    def __init__(self, z_dim=100, img_channels=3):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        # Initial layer: (z_dim) -> (512, 4, 4)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            
            # (512, 4, 4) -> (256, 8, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            
            # (256, 8, 8) -> (128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            
            # (128, 16, 16) -> (64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            
            # (64, 32, 32) -> (img_channels, 32, 32)
            nn.ConvTranspose2d(64, img_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        return self.main(z)


class Critic(nn.Module):
    """Critic network for WGAN (replacing discriminator)
    
    Note: Outputs real-valued scores (not probabilities) that estimate Wasserstein distance
    """
    
    def __init__(self, img_channels=3):
        super(Critic, self).__init__()
        
        self.main = nn.Sequential(
            # Input: (img_channels, 32, 32)
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (64, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (128, 8, 8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (256, 4, 4)
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output layer (no sigmoid - outputs unbounded Wasserstein estimate)
        self.output = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        return self.output(x)


def weights_init(m):
    """Initialize network weights"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.02)


def clip_weights(critic, clip_value):
    """Enforce Lipschitz constraint via weight clipping"""
    for p in critic.parameters():
        p.data.clamp_(-clip_value, clip_value)


def train_wgan(batch_size=64, epochs=100, z_dim=100, learning_rate=5e-5, 
               critic_iterations=5, clip_value=0.01, device='cpu'):
    """
    Train WGAN on CIFAR-10
    
    Args:
        batch_size: Batch size for training
        epochs: Number of epochs to train
        z_dim: Dimensionality of latent space
        learning_rate: Learning rate for both networks
        critic_iterations: Number of critic updates per generator update
        clip_value: Weight clipping bound
        device: Device to train on ('cpu' or 'cuda')
    
    Returns:
        dict: Training history and metrics
    """
    
    # Setup data loader
    print("[*] Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Initialize networks
    print("[*] Initializing networks...")
    generator = Generator(z_dim=z_dim).to(device)
    critic = Critic().to(device)
    
    generator.apply(weights_init)
    critic.apply(weights_init)
    
    # Optimizers (RMSprop recommended for WGAN)
    opt_g = optim.RMSprop(generator.parameters(), lr=learning_rate)
    opt_c = optim.RMSprop(critic.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'epoch': [],
        'critic_loss': [],
        'generator_loss': [],
        'wasserstein_distance': []
    }
    
    print(f"[*] Starting training for {epochs} epochs...")
    print(f"    Critic iterations per gen update: {critic_iterations}")
    print(f"    Weight clip value: {clip_value}")
    print(f"    Device: {device}\n")
    
    for epoch in range(epochs):
        critic_loss_sum = 0.0
        gen_loss_sum = 0.0
        wasserstein_sum = 0.0
        num_batches = 0
        
        for batch_idx, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)
            batch_size_actual = real_images.size(0)
            
            # ------- Train Critic -------
            for _ in range(critic_iterations):
                # Real batch
                critic.zero_grad()
                critic_real = critic(real_images)
                loss_real = -critic_real.mean()
                
                # Fake batch
                z = torch.randn(batch_size_actual, z_dim, 1, 1, device=device)
                fake_images = generator(z).detach()
                critic_fake = critic(fake_images)
                loss_fake = critic_fake.mean()
                
                # Total critic loss (Wasserstein distance estimate)
                critic_loss = loss_real + loss_fake
                critic_loss.backward()
                opt_c.step()
                
                # Enforce Lipschitz constraint
                clip_weights(critic, clip_value)
            
            # ------- Train Generator -------
            generator.zero_grad()
            z = torch.randn(batch_size_actual, z_dim, 1, 1, device=device)
            fake_images = generator(z)
            critic_fake = critic(fake_images)
            gen_loss = -critic_fake.mean()
            gen_loss.backward()
            opt_g.step()
            
            # Record metrics
            critic_loss_sum += critic_loss.item()
            gen_loss_sum += gen_loss.item()
            wasserstein_sum += (loss_real + loss_fake).item()
            num_batches += 1
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Critic Loss: {critic_loss.item():.4f} | Gen Loss: {gen_loss.item():.4f}")
        
        # Epoch statistics
        avg_critic_loss = critic_loss_sum / num_batches
        avg_gen_loss = gen_loss_sum / num_batches
        avg_wasserstein = wasserstein_sum / num_batches
        
        history['epoch'].append(epoch + 1)
        history['critic_loss'].append(avg_critic_loss)
        history['generator_loss'].append(avg_gen_loss)
        history['wasserstein_distance'].append(avg_wasserstein)
        
        print(f"\n[Epoch {epoch+1}/{epochs}] "
              f"Critic Loss: {avg_critic_loss:.4f} | Gen Loss: {avg_gen_loss:.4f} | "
              f"Wasserstein: {avg_wasserstein:.4f}\n")
    
    print("[*] Training completed!")
    
    return {
        'generator': generator,
        'critic': critic,
        'history': history
    }


def generate_samples(generator, num_samples=16, z_dim=100, device='cpu'):
    """Generate samples from trained generator"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, 1, 1, device=device)
        samples = generator(z)
    return samples


def save_samples_grid(samples, filename, num_cols=4):
    """Save samples as a grid image"""
    samples = samples.detach().cpu()
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    
    num_samples = len(samples)
    num_rows = (num_samples + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3*num_rows))
    axes = axes.flatten()
    
    for idx, sample in enumerate(samples):
        img = sample.permute(1, 2, 0).numpy()
        axes[idx].imshow(img)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"[*] Saved sample grid to {filename}")
    plt.close()


def plot_training_history(history, output_dir='reports'):
    """Plot training metrics"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Critic Loss
    axes[0].plot(history['epoch'], history['critic_loss'], label='Critic Loss', color='blue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Critic Loss over Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Generator Loss
    axes[1].plot(history['epoch'], history['generator_loss'], label='Generator Loss', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Generator Loss over Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Wasserstein Distance
    axes[2].plot(history['epoch'], history['wasserstein_distance'], label='Wasserstein Distance', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Distance')
    axes[2].set_title('Wasserstein Distance over Epochs')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.png', dpi=100, bbox_inches='tight')
    print(f"[*] Saved training history plot to {output_dir}/training_history.png")
    plt.close()


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("[*] GPU/Device Information:")
    print(f"    PyTorch version: {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    GPU device count: {torch.cuda.device_count()}")
        print(f"    Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"    Selected device: {device.upper()}\n")
    
    # Training hyperparameters
    config = {
        'batch_size': 64,
        'epochs': 100,
        'z_dim': 100,
        'learning_rate': 5e-5,
        'critic_iterations': 5,
        'clip_value': 0.01,
    }
    
    print("[*] Training Configuration:")
    for key, value in config.items():
        print(f"    {key}: {value}")
    print()
    
    # Train WGAN
    results = train_wgan(
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        z_dim=config['z_dim'],
        learning_rate=config['learning_rate'],
        critic_iterations=config['critic_iterations'],
        clip_value=config['clip_value'],
        device=device
    )
    
    generator = results['generator']
    history = results['history']
    
    # Create output directory
    Path('reports').mkdir(exist_ok=True)
    
    # Generate and save samples
    print("\n[*] Generating final samples...")
    samples = generate_samples(generator, num_samples=16, z_dim=config['z_dim'], device=device)
    save_samples_grid(samples, 'reports/generated_samples.png')
    
    # Plot training history
    print("[*] Plotting training history...")
    plot_training_history(history, 'reports')
    
    # Save training configuration and history
    config_output = {
        'config': config,
        'history': history,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('reports/training_config.json', 'w') as f:
        # Convert tensors/numpy to native Python types for JSON
        history_json = {
            k: [float(v) for v in history[k]] if isinstance(history[k], list) else history[k]
            for k in history
        }
        config_output['history'] = history_json
        json.dump(config_output, f, indent=4)
    
    print("[*] Saved training configuration to reports/training_config.json")
    print("\n[*] WGAN training complete!")
    print("[*] Check the 'reports' directory for generated samples and training plots.")


if __name__ == '__main__':
    main()
