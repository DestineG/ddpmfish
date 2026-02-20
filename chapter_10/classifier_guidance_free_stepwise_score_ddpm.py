import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# --- 1. 数据加载 ---
def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# --- 2. 扩散过程控制 ---
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
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise

# --- 3. 时间与类别嵌入 ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        inv_freq = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device) / (half_dim - 1)
        )
        t = time.float().unsqueeze(1)
        emb = t * inv_freq.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

# --- 4. 基础模块 ---
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.cond_proj = nn.Linear(cond_emb_dim, out_ch)
        self.act = nn.ReLU()

    def forward(self, x, t_emb, c_emb):
        h = self.act(self.conv1(x))
        time = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        cond = self.cond_proj(c_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time + cond
        h = self.act(self.conv2(h))
        return h

# --- 5. 条件扩散模型（U-Net 结构） ---
class CondDiffusionModel(nn.Module):
    def __init__(self, time_dim=128, num_classes=10):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        # 类别嵌入
        self.num_classes = num_classes
        self.cond_emb = nn.Embedding(num_classes + 1, time_dim)  # 最后一类为无条件 (-1 → num_classes)

        # U-Net 主体
        self.down1 = Block(1, 64, time_dim, time_dim)
        self.down2 = Block(64, 128, time_dim, time_dim)
        self.mid = Block(128, 128, time_dim, time_dim)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_block1 = Block(192, 64, time_dim, time_dim)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up_block2 = Block(96, 32, time_dim, time_dim)
        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x, t, y):
        # 时间嵌入
        t_emb = self.time_mlp(t)
        # 条件嵌入（y=-1 → 无条件）
        y_emb = self.cond_emb(y)
        x1 = self.down1(x, t_emb, y_emb)
        x2 = self.down2(F.max_pool2d(x1, 2), t_emb, y_emb)
        x_mid = self.mid(F.max_pool2d(x2, 2), t_emb, y_emb)
        out = torch.cat((self.up1(x_mid), x2), dim=1)
        out = self.up_block1(out, t_emb, y_emb)
        out = torch.cat((self.up2(out), x1), dim=1)
        out = self.up_block2(out, t_emb, y_emb)
        return self.final_conv(out)

# --- 6. 训练过程 ---
def train(epochs, model, dataloader, diff_proc, optimizer, device):
    model.train()
    for epoch in range(epochs):
        tq = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", ascii=True, leave=False)
        for x0, labels in tq:
            x0, labels = x0.to(device), labels.to(device)
            t = torch.randint(0, diff_proc.T, (x0.size(0),), device=device)
            noise = torch.randn_like(x0)
            xt = diff_proc.sample_q(x0, t, noise)

            # 50% 概率使用无条件（y=-1）
            mask = torch.rand(x0.size(0), device=device) < 0.5
            labels_cond = labels.clone()
            labels_cond[mask] = model.num_classes  # num_classes 表示无条件类别

            pred_noise = model(xt, t, labels_cond)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tq.set_postfix(loss=loss.item())

# --- 7. 采样 (classifier-free guidance) ---
def sample_with_guidance(model, diff_proc, device, target_class, guidance_scale=3.0, num_samples=16):
    model.eval()
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    y_cond = torch.full((num_samples,), target_class, device=device, dtype=torch.long)
    y_uncond = torch.full((num_samples,), model.num_classes, device=device, dtype=torch.long)

    print(f"Generating class {target_class} with scale={guidance_scale}...")
    for t_int in reversed(range(diff_proc.T)):
        t = torch.full((x.size(0),), t_int, device=device, dtype=torch.long)
        with torch.no_grad():
            eps_uncond = model(x, t, y_uncond)
            eps_cond = model(x, t, y_cond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            ab = diff_proc.alpha_bar[t_int]
            beta_t = diff_proc.beta[t_int]
            x = (1 / torch.sqrt(diff_proc.alpha[t_int])) * (x - (beta_t / torch.sqrt(1 - ab)) * eps)
            if t_int > 0:
                x += torch.sqrt(beta_t) * torch.randn_like(x)

    x = (x.clamp(-1, 1) + 1) / 2
    os.makedirs("results", exist_ok=True)
    save_image(x, f"results/cond_guided_class_{target_class}.png", nrow=4)
    print("Done! Image saved.")

# --- 8. 主函数 ---
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    diff_proc = DiffusionProcess(T=300, device=device)  # T=300 加快调试
    model = CondDiffusionModel(time_dim=128, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = get_dataloader(batch_size=128)

    # 训练
    train(epochs=3, model=model, dataloader=dataloader, diff_proc=diff_proc, optimizer=optimizer, device=device)

    # 条件采样
    sample_with_guidance(model, diff_proc, device, target_class=3, guidance_scale=5.0, num_samples=16)
