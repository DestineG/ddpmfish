import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class VAEEmbedding(torch.nn.Module):
    def __init__(self, num_classes=10, embedding_dim=10):
        super(VAEEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(num_classes, embedding_dim)
        self.fc1 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, labels):
        embedded = self.embedding(labels)
        h = torch.relu(self.fc1(embedded))
        return torch.relu(self.fc2(h))

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
        return self.fc3_mean(h), self.fc3_logvar(h)


class VAEDecoder(torch.nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.fc1 = torch.nn.Linear(latent_dim, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))


class classifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(classifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)


class VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes=10):
        super(VAE, self).__init__()
        self.embedding = VAEEmbedding(num_classes, embedding_dim=latent_dim)
        self.encoder = VAEEncoder(input_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim + latent_dim, input_dim)
        self.classifier = classifier(latent_dim, num_classes)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, label):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        label_embedding = self.embedding(label)
        z = torch.cat([z, label_embedding], dim=1)
        output = self.decoder(z)
        class_embedding = self.classifier(label_embedding)
        return output, mean, logvar, class_embedding


def save_samples(model, latent_dim, device, epoch, out_dir="samples"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        # 每行对应一个类别
        fig, axes = plt.subplots(10, 10, figsize=(10, 10))
        for digit in range(10):
            label = torch.full((10,), digit, dtype=torch.long, device=device)
            z = torch.randn(10, latent_dim).to(device)
            label_embedding = model.embedding(label)
            z = torch.cat([z, label_embedding], dim=1)
            samples = model.decoder(z).cpu().view(-1, 1, 28, 28)
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

    # 创建历史记录字典
    history = {
        'total': [],
        'recon': [],
        'kl': [],
        'class': []
    }
    
    tq = tqdm(range(1, num_epochs + 1), desc='Training VAE', ascii=True, leave=False)

    for epoch in tq:
        # 每个 epoch 重置累加器
        epoch_recon = 0
        epoch_kl = 0
        epoch_class = 0
        epoch_total = 0
        
        for input, label in dataloader:
            input = input.view(input.size(0), -1).to(device)
            label = label.to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar, class_embedding = model(input, label)

            # 1. 重构误差 (BCE)
            recon_loss = F.binary_cross_entropy(x_recon, input, reduction='sum')
            # 2. KL 散度
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # 3. 分类损失 (注意：此处建议加上 reduction='sum' 以保持量级一致)
            class_loss = F.cross_entropy(class_embedding, label, reduction='sum')
            
            loss = recon_loss + kl_loss + class_loss

            loss.backward()
            optimizer.step()

            # 累计
            epoch_total += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            epoch_class += class_loss.item()

        # 计算平均值并存入 history
        n = len(dataloader.dataset)
        history['total'].append(epoch_total / n)
        history['recon'].append(epoch_recon / n)
        history['kl'].append(epoch_kl / n)
        history['class'].append(epoch_class / n)

        tq.set_postfix({'loss': f'{history["total"][-1]:.2f}', 'class': f'{history["class"][-1]:.2f}'})

        # 采样生成图片
        save_samples(model, latent_dim, device, epoch, out_dir=os.path.join(exp_dir, "samples"))

    # Loss 曲线
    loss_curve_path = os.path.join(exp_dir, f"{os.path.basename(exp_dir)}_loss_curves.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = range(1, num_epochs + 1)

    # Total Loss
    axes[0, 0].plot(epochs, history['total'], color='black', lw=1.5)
    axes[0, 0].set_title('Total Loss')

    # Reconstruction Loss
    axes[0, 1].plot(epochs, history['recon'], color='blue', lw=1.5)
    axes[0, 1].set_title('Reconstruction Loss (BCE)')

    # KL Divergence
    axes[1, 0].plot(epochs, history['kl'], color='red', lw=1.5)
    axes[1, 0].set_title('KL Divergence')

    # Classification Loss
    axes[1, 1].plot(epochs, history['class'], color='green', lw=1.5)
    axes[1, 1].set_title('Classification Loss')

    # 统一格式化
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(mnist, batch_size=128, shuffle=True)

    input_dim = 28 * 28
    latent_dim =20
    vae = VAE(input_dim, latent_dim)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    num_epochs = 100
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    exp_dir = "exps/vae_base_control_v4"
    os.makedirs(exp_dir, exist_ok=True)
    train_vae(vae, dataloader, optimizer, num_epochs, device, latent_dim, exp_dir=exp_dir)
