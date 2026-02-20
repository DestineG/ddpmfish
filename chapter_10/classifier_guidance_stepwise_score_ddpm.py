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

# --- 2. 扩散过程控制 (DDPM 调度器) ---
class DiffusionProcess:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = T
        self.device = device
        # beta: [T]
        self.beta = torch.linspace(beta_start, beta_end, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # \bar{\alpha}_t

    def sample_q(self, x0, t, noise=None):
        """
        重参数化采样:
            x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * epsilon
        t: LongTensor shape (batch,)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # alpha_bar[t] -> shape (batch,)
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_ab = torch.sqrt(ab)
        sqrt_one_minus_ab = torch.sqrt(1.0 - ab)
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise

# --- 3. 基础组件 (时间嵌入与卷积块) ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        """
        dim: total embedding dimension (should be even)
        """
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, "time embedding dim must be even"

    def forward(self, time):
        """
        time: tensor shape (batch,) dtype long or float
        returns: (batch, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        # frequencies: [half_dim]
        inv_freq = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device, dtype=torch.float32) / (half_dim - 1)
        )
        # ensure float
        t = time.float().unsqueeze(1)  # (batch, 1)
        args = t * inv_freq.unsqueeze(0)  # (batch, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, dim)
        return emb

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        # time embedding to add per-channel bias
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x, t_emb):
        """
        x: (B, in_ch, H, W)
        t_emb: (B, time_emb_dim)  <- already processed by global time MLP
        """
        h = self.act(self.conv1(x))  # (B, out_ch, H, W)
        # map time embedding to (B, out_ch) and broadcast to (B, out_ch, H, W)
        time_emb = self.time_mlp(t_emb)  # (B, out_ch)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)  # (B, out_ch, 1, 1)
        h = h + time_emb
        h = self.act(self.conv2(h))
        return h

# --- 4. 生成模型 (U-Net style, small) ---
class DiffusionModel(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        # time embedding net: sin embeddings -> dense -> act
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # encoder / decoder blocks
        self.down1 = Block(1, 64, time_dim)    # output: (B,64,28,28)
        self.down2 = Block(64, 128, time_dim)  # expect input (B,64,14,14) after pooling
        self.mid = Block(128, 128, time_dim)   # input (B,128,7,7)
        # upsampling
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # (B,64,14,14)
        self.up_block1 = Block(64 + 128, 64, time_dim)  # concat channels: 64 + skip(128) = 192 -> but we will match shapes
        # Note: above reflects channels after concat; adjust carefully in forward
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # (B,32,28,28)
        self.up_block2 = Block(32 + 64, 32, time_dim)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, t):
        """
        x: (B,1,28,28)
        t: (B,) long or float - will be converted by time_mlp
        """
        # time embedding
        t_emb = self.time_mlp(t)  # (B, time_dim)

        # down
        x1 = self.down1(x, t_emb)                         # (B,64,28,28)
        x1_p = F.max_pool2d(x1, 2)                        # (B,64,14,14)
        x2 = self.down2(x1_p, t_emb)                      # (B,128,14,14)
        x2_p = F.max_pool2d(x2, 2)                        # (B,128,7,7)
        x_mid = self.mid(x2_p, t_emb)                     # (B,128,7,7)

        # up 1
        u1 = self.up1(x_mid)                              # (B,64,14,14)
        # concatenate with x2 (skip connection); x2 is (B,128,14,14)
        out = torch.cat((u1, x2), dim=1)                  # (B,64+128,14,14) => 192 channels
        out = self.up_block1(out, t_emb)                  # expected to produce (B,64,14,14)

        # up 2
        u2 = self.up2(out)                                # (B,32,28,28)
        out = torch.cat((u2, x1), dim=1)                  # (B,32+64,28,28) => 96 channels
        out = self.up_block2(out, t_emb)                  # (B,32,28,28)

        return self.final_conv(out)                       # (B,1,28,28)

# --- 5. 噪声分类器 (用于引导) ---
class NoiseClassifier(nn.Module):
    """ 分类器：识别带噪声的 x_t 图像的真实标签 """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),   # -> (B,32,14,14)
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # -> (B,64,7,7)
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.conv(x)

# --- 6. 训练与引导采样 ---
def train(epochs, model, classifier, dataloader, diff_proc, opt, cls_opt, device):
    model.train()
    classifier.train()
    for epoch in range(epochs):
        tq = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", ascii=True, leave=False)
        for x0, labels in tq:
            x0, labels = x0.to(device), labels.to(device)
            batch = x0.size(0)

            # 随机挑选时间步 t (0..T-1)
            t = torch.randint(0, diff_proc.T, (batch,), device=device, dtype=torch.long)

            # 随机噪声
            noise = torch.randn_like(x0)
            xt = diff_proc.sample_q(x0, t, noise)

            # 1. 训练扩散模型 (预测噪声)
            pred_noise = model(xt, t)
            loss_diff = F.mse_loss(pred_noise, noise)

            opt.zero_grad()
            loss_diff.backward()
            opt.step()

            # 2. 训练分类器 (识别加噪图像 xt)
            logits = classifier(xt.detach())   # detach 避免影响扩散模型参数
            loss_cls = F.cross_entropy(logits, labels)

            cls_opt.zero_grad()
            loss_cls.backward()
            cls_opt.step()

            tq.set_postfix(diff=loss_diff.item(), cls=loss_cls.item())

def guided_test(model, classifier, diff_proc, device, target_class, gamma=2.0, num_samples=16):
    """
    使用用户提供的公式进行分类器引导采样
    gamma: 分类器指引权重
    """
    model.eval()
    classifier.eval()

    # 1. 初始噪声 x_T
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    y = torch.full((num_samples,), target_class, device=device, dtype=torch.long)

    print(f"Generating samples for class {target_class} with gamma={gamma}...")

    for t_int in reversed(range(diff_proc.T)):
        t_tensor = torch.full((x.size(0),), t_int, device=device, dtype=torch.long)

        # --- A. 计算分类器梯度: ∇ log p(y|x_t) ---
        x.requires_grad_(True)
        logits = classifier(x)
        log_probs = F.log_softmax(logits, dim=-1)
        # 提取目标类别的 log 概率总和
        selected_log_probs = log_probs[range(len(y)), y].sum()
        grad_y_xt = torch.autograd.grad(selected_log_probs, x)[0]
        x = x.detach()

        with torch.no_grad():
            # 获取当前步的参数
            alpha_t = diff_proc.alpha[t_int]
            alpha_bar_t = diff_proc.alpha_bar[t_int]
            beta_t = diff_proc.beta[t_int]
            sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar_t)

            # --- B. 计算模型 Score: s_θ(x_t) ---
            # 根据关系: epsilon = -sqrt(1-ab) * score  =>  score = -epsilon / sqrt(1-ab)
            eps_theta_raw = model(x, t_tensor)
            score_theta = -eps_theta_raw / sqrt_one_minus_ab

            # --- C. 计算最终引导 Score: s(x_t | y) ---
            # 公式: s(x_t | y) = s_θ(x_t) + γ * ∇ log p_φ(y | x_t)
            final_score = score_theta + gamma * grad_y_xt

            # --- D. 转换回噪声预测因子: ε_θ(x_t, t) ---
            # 公式: ε_θ(x_t, t) = -sqrt(1 - α_bar_t) * s(x_t | y)
            eps_guided = -sqrt_one_minus_ab * final_score

            # --- E. DDPM 步进: x_{t-1} ---
            # 公式: x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√1-α_bar_t * ε_θ) + √β_t * z
            coeff_x = 1.0 / torch.sqrt(alpha_t)
            coeff_eps = (1.0 - alpha_t) / sqrt_one_minus_ab
            
            mean = coeff_x * (x - coeff_eps * eps_guided)

            if t_int > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

    # 保存结果
    x = (x.clamp(-1, 1) + 1.0) / 2.0
    os.makedirs("results", exist_ok=True)
    save_image(x, f"results/formula_guided_{target_class}.png", nrow=4)
    print("Sampling complete.")

# --- 7. 主流程 (可运行) ---
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    # 1. 初始化
    # 为了快速调试: 可以把 T 调小 (比如 100)；这里保留 1000 作为默认
    diff_proc = DiffusionProcess(T=1000, device=device)
    model = DiffusionModel(time_dim=128).to(device)
    classifier = NoiseClassifier().to(device)

    # 优化器
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cls_opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    # 数据载入：debug 时可把 batch_size 调小
    dataloader = get_dataloader(batch_size=128)

    # 2. 快速训练 (示例 3 epochs)
    # 如果在 CPU 上训练，请把 epochs 调小（例如 1）或把 T 在 DiffusionProcess 初始化时降到 100
    train(epochs=3, model=model, classifier=classifier, dataloader=dataloader,
          diff_proc=diff_proc, opt=opt, cls_opt=cls_opt, device=device)

    # 3. 生成指定数字 (例如数字 3)
    # 注意：full-guided sampling 迭代步 T=1000 次会很慢。可在 DiffusionProcess 初始化时将 T 设小以便测试。
    guided_test(model, classifier, diff_proc, device, target_class=3, gamma=5.0, num_samples=16)