import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class VAEEncoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAEEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3_mean = torch.nn.Linear(64, latent_dim)
        self.fc3_logvar = torch.nn.Linear(64, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mean = self.fc3_mean(h)
        logvar = self.fc3_logvar(h)
        return mean, logvar


class VAEDecoder(torch.nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.fc1 = torch.nn.Linear(latent_dim, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h))
        output = torch.sigmoid(self.fc3(h))
        return output


class VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(input_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, input_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        output = self.decoder(z)
        return output, mean, logvar


def save_samples(model, latent_dim, device, epoch, out_dir="samples"):
    """采样并保存16张图像为PNG"""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)
        samples = model.decoder(z).cpu().view(-1, 1, 28, 28)

    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i][0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(save_path)
    plt.close(fig)


def train_vae(model, dataloader, optimizer, num_epochs=10, device='cpu', latent_dim=20, exp_dir="exps/vae_base"):
    model.to(device)
    model.train()

    losses = []  # 收集每个epoch平均loss
    tq = tqdm(range(1, num_epochs + 1), desc='Training VAE', ascii=True, leave=False)

    for epoch in tq:
        total_loss = 0
        for batch, _ in dataloader:
            batch = batch.view(batch.size(0), -1).to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch)

            # 重构误差 (binary cross-entropy)
            recon_loss = F.binary_cross_entropy(x_recon, batch, reduction='sum')
            # KL 散度项
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        losses.append(avg_loss)
        tq.set_postfix({'loss': f'{avg_loss:.4f}'})

        # 🔹 每个 epoch 保存采样结果
        save_samples(model, latent_dim, device, epoch, out_dir=os.path.join(exp_dir, "samples"))

    print("✅ 训练完成！生成样本保存在 ./samples/ 目录中。")

    # 🔹 绘制loss曲线
    loss_curve_path = os.path.join(exp_dir, "loss_curve.png")
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_epochs + 1), losses, color='blue', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("VAE Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Loss 曲线已保存为 {loss_curve_path}")


# git@github.com:DestineG/ddpmfish.git
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

    exp_dir = "exps/vae_base"
    os.makedirs(exp_dir, exist_ok=True)
    train_vae(vae, dataloader, optimizer, num_epochs, device, latent_dim, exp_dir=exp_dir)
