import os
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
import math


def get_dataloader(batch_size, image_size=(28, 28)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

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
        sqrt_ab = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_ab = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
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
        # 第一层卷积
        h = self.relu(self.conv1(x))
        # 注入时间信息：将时间嵌入映射到与特征图相同的通道数
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2] # 扩展维度以匹配特征图 [B, C, 1, 1]
        # 时间嵌入与特征融合
        h = h + time_emb
        # 第二层卷积
        return self.relu(self.conv2(h))

class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        time_dim = 128
        
        # 1. 时间编码器
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # 2. 下采样部分 (Encoder)
        self.down1 = Block(1, 64, time_dim)
        self.pool1 = nn.MaxPool2d(2) # 28 -> 14
        self.down2 = Block(64, 128, time_dim)
        self.pool2 = nn.MaxPool2d(2) # 14 -> 7

        # 3. 中间层 (Bottleneck)
        self.mid = Block(128, 128, time_dim)

        # 4. 上采样部分 (Decoder)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2) # 7 -> 14
        self.up_block1 = Block(192, 64, time_dim) # 64(up) + 128(skip) = 192
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2) # 14 -> 28
        self.up_block2 = Block(96, 32, time_dim) # 32(up) + 64(skip) = 96

        # 输出层
        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x, t):
        # 时间嵌入
        t = self.time_mlp(t)

        # Encoder
        x1 = self.down1(x, t)        # [B, 64, 28, 28]
        x2 = self.down2(self.pool1(x1), t) # [B, 128, 14, 14]
        
        # Mid
        x_mid = self.mid(self.pool2(x2), t) # [B, 128, 7, 7]

        # Decoder
        # 第一次上采样并拼接来自 x2 的特征
        out = self.up1(x_mid)
        out = torch.cat((out, x2), dim=1) 
        out = self.up_block1(out, t)

        # 第二次上采样并拼接来自 x1 的特征
        out = self.up2(out)
        out = torch.cat((out, x1), dim=1)
        out = self.up_block2(out, t)

        return self.final_conv(out)

def train(epochs, model, dataloader, diffusionProcess, optimizer, device, exp_dir):
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(epochs):
        tq = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', ascii=True, leave=False)
        for x0, _ in tq:
            x0 = x0.to(device)
            t = torch.randint(0, diffusionProcess.T, (x0.size(0),), device=device)
            noise = torch.randn_like(x0)
            xt = diffusionProcess.sample_q(x0, t, noise)
            
            # 仍然预测噪声，若预测score会导致loss不稳定
            predicted_noise = model(xt, t)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if tq.n % 20 == 0:
                tq.set_postfix(loss=loss.item())
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth'))
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'final_model.pth'))

def test(model, diffusionProcess, device, exp_dir):
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'final_model.pth'), map_location=device))
    model.eval()
    os.makedirs(os.path.join(exp_dir, 'samples'), exist_ok=True)

    with torch.no_grad():
        x = torch.randn(64, 1, 28, 28, device=device)

        for t in reversed(range(diffusionProcess.T)):
            t_tensor = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_tensor)

            alpha_t = diffusionProcess.alpha[t]
            alpha_bar_t = diffusionProcess.alpha_bar[t]
            beta_t = diffusionProcess.beta[t]

            x = (1 / torch.sqrt(alpha_t)) * (
                    x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
                )

            if t > 0:
                x += torch.sqrt(beta_t) * torch.randn_like(x)

            # 可视化中间过程
            if t % 200 == 0 or t == 0:
                save_image((x.clamp(-1, 1) + 1) / 2, 
                           os.path.join(exp_dir, 'samples', f'step_{t:04d}.png'), nrow=8)

        # 最终结果
        x = (x.clamp(-1, 1) + 1) / 2
        final_path = os.path.join(exp_dir, 'samples', 'generated_final.png')
        save_image(x, final_path, nrow=8)
        print(f"保存生成图像: {final_path}")

if __name__ == "__main__":
    exp_dir = './exps/stepwise_score_diffusion'
    device = 'cpu'
    epochs = 3
    diffusionProcess = DiffusionProcess(device=device)
    dataloader = get_dataloader(batch_size=64)
    model = DiffusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #train(epochs, model, dataloader, diffusionProcess, optimizer, device, exp_dir)
    test(model, diffusionProcess, device, exp_dir)