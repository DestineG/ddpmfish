import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class VAEEncoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes=10):
        super(VAEEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim + num_classes, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3_mean = torch.nn.Linear(64, latent_dim)
        self.fc3_logvar = torch.nn.Linear(64, latent_dim)

    def forward(self, x, label=None):
        if label is not None:
            one_hot = F.one_hot(label, num_classes=10).to(x.device).float()
            x = torch.cat([x, one_hot], dim=1)
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc3_mean(h), self.fc3_logvar(h)


class VAEDecoder(torch.nn.Module):
    def __init__(self, latent_dim, output_dim, num_classes=10):
        super(VAEDecoder, self).__init__()
        self.fc1 = torch.nn.Linear(latent_dim + num_classes, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, output_dim)

    def forward(self, z, label=None):
        if label is not None:
            one_hot = F.one_hot(label, num_classes=10).to(z.device).float()
            z = torch.cat([z, one_hot], dim=1)
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))


class VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes=10):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(input_dim, latent_dim, num_classes)
        self.decoder = VAEDecoder(latent_dim, input_dim, num_classes)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, label=None):
        mean, logvar = self.encoder(x, label)
        z = self.reparameterize(mean, logvar)
        output = self.decoder(z, label)
        return output, mean, logvar


def save_samples(model, latent_dim, device, epoch, out_dir="samples"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        # 每行对应一个类别
        fig, axes = plt.subplots(10, 10, figsize=(10, 10))
        for digit in range(10):
            label = torch.full((10,), digit, dtype=torch.long, device=device)
            z = torch.randn(10, latent_dim).to(device)
            samples = model.decoder(z, label).cpu().view(-1, 1, 28, 28)
            for i in range(10):
                ax = axes[digit, i]
                ax.imshow(samples[i][0], cmap='gray')
                ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"epoch_{epoch:03d}.png"))
        plt.close(fig)


def train_vae(model, dataloader, optimizer, num_epochs=10, device='cpu', latent_dim=20, exp_dir="exps/vae_base"):
    model.to(device)
    model.train()

    # 用于记录三类 loss
    history = {
        'total': [],
        'recon': [],
        'kl': []
    }
    
    tq = tqdm(range(1, num_epochs + 1), desc='Training VAE', ascii=True, leave=False)

    for epoch in tq:
        total_epoch_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for input, label in dataloader:
            input = input.view(input.size(0), -1).to(device)
            label = label.to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar = model(input, label)

            # 1. 重构误差
            recon_loss = F.binary_cross_entropy(x_recon, input, reduction='sum')
            # 2. KL 散度项
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        # 计算该 epoch 的平均值
        num_samples = len(dataloader.dataset)
        history['total'].append(total_epoch_loss / num_samples)
        history['recon'].append(total_recon_loss / num_samples)
        history['kl'].append(total_kl_loss / num_samples)
        
        tq.set_postfix({'loss': f'{history["total"][-1]:.2f}', 'recon': f'{history["recon"][-1]:.2f}'})

        # 采样
        save_samples(model, latent_dim, device, epoch, out_dir=os.path.join(exp_dir, "samples"))

    # Loss 曲线
    loss_curve_path = os.path.join(exp_dir, f"{os.path.basename(exp_dir)}_loss_curves.png")
    # 创建 1 行 3 列的子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, num_epochs + 1)

    # 1. Total Loss 子图
    axes[0].plot(epochs, history['total'], color='black', linewidth=2)
    axes[0].set_title('Total Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Reconstruction Loss 子图
    axes[1].plot(epochs, history['recon'], color='tab:blue', linewidth=2)
    axes[1].set_title('Reconstruction Loss (BCE)', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 3. KL Divergence 子图
    axes[2].plot(epochs, history['kl'], color='tab:red', linewidth=2)
    axes[2].set_title('KL Divergence Loss', fontsize=14)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(f"VAE Training Metrics - {os.path.basename(exp_dir)}", fontsize=16)
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(mnist, batch_size=128, shuffle=True)

    input_dim = 28 * 28
    latent_dim = 20
    vae = VAE(input_dim, latent_dim)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    num_epochs = 100
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    exp_dir = "exps/vae_base_control_v1"
    os.makedirs(exp_dir, exist_ok=True)
    train_vae(vae, dataloader, optimizer, num_epochs, device, latent_dim, exp_dir=exp_dir)
