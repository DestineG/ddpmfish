import os
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# --- 1. 数据准备 ---
def get_dataloader(batch_size, image_size=(28, 28)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 2. 扩散过程参数 ---
class DiffusionProcess:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = T
        self.device = device
        self.beta = torch.linspace(beta_start, beta_end, T, device=device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def sample_q(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        # 重参数化技巧：从 x0 采样 t 时刻的图片
        sqrt_ab = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_ab = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise

# --- 3. 核心组件：时间编码与卷积块 ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # 生成等比数列频率
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # 计算角度并拼接 sin 和 cos
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.relu(self.conv1(x))
        # 注入融合了时间+条件的信息
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2] # 扩展为 [B, C, 1, 1]
        h = h + time_emb
        return self.relu(self.conv2(h))

# --- 4. 条件扩散模型 (U-Net 简化版) ---
class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        time_dim = 128
        
        # 时间编码支路
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # 类别编码支路：10个数字映射到128维
        self.label_emb = nn.Embedding(10, time_dim)

        # 网络结构
        self.down1 = Block(1, 64, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(64, 128, time_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.mid = Block(128, 128, time_dim)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_block1 = Block(192, 64, time_dim)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up_block2 = Block(96, 32, time_dim)
        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x, t, labels=None):
        t_emb = self.time_mlp(t)
        if labels is not None:
            # 融合条件信息
            t_emb = t_emb + self.label_emb(labels)

        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool1(x1), t_emb)
        x_mid = self.mid(self.pool2(x2), t_emb)
        
        out = self.up1(x_mid)
        out = torch.cat((out, x2), dim=1) 
        out = self.up_block1(out, t_emb)

        out = self.up2(out)
        out = torch.cat((out, x1), dim=1)
        out = self.up_block2(out, t_emb)
        return self.final_conv(out)

# --- 5. 训练与测试逻辑 ---
def train(epochs, model, dataloader, diffusionProcess, optimizer, device, exp_dir):
    model.train()
    os.makedirs(exp_dir, exist_ok=True)
    for epoch in range(epochs):
        tq = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for x0, labels in tq:
            x0, labels = x0.to(device), labels.to(device)
            t = torch.randint(0, diffusionProcess.T, (x0.size(0),), device=device)
            noise = torch.randn_like(x0)
            xt = diffusionProcess.sample_q(x0, t, noise)
            
            # 传入 labels 进行有条件训练
            predicted_noise = model(xt, t, labels)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tq.set_postfix(loss=loss.item())
    torch.save(model.state_dict(), os.path.join(exp_dir, 'model.pth'))

def test(model, diffusionProcess, device, exp_dir):
    model.eval()
    # 准备 80 个样本：0-9 每个数字生成 8 个
    n_classes = 10
    samples_per_class = 8
    labels = torch.arange(n_classes).repeat_interleave(samples_per_class).to(device)
    
    with torch.no_grad():
        x = torch.randn(len(labels), 1, 28, 28, device=device)
        for t in reversed(range(diffusionProcess.T)):
            t_tensor = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            # 推理时带上指定的 labels
            predicted_noise = model(x, t_tensor, labels)

            alpha_t = diffusionProcess.alpha[t]
            alpha_bar_t = diffusionProcess.alpha_bar[t]
            sqrt_one_minus_ab = torch.sqrt(1 - alpha_bar_t)
            
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / sqrt_one_minus_ab) * predicted_noise)
            if t > 0:
                x += torch.sqrt(diffusionProcess.beta[t]) * torch.randn_like(x)

        save_image((x.clamp(-1, 1) + 1) / 2, os.path.join(exp_dir, 'result.png'), nrow=samples_per_class)

if __name__ == "__main__":
    # 配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = "./exps/diffusion_control"
    
    # 初始化
    process = DiffusionProcess(device=device)
    model = DiffusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = get_dataloader(batch_size=128)
    
    # 运行
    train(5, model, loader, process, optimizer, device, exp_dir)
    test(model, process, device, exp_dir)