import numpy as np
import os
import matplotlib.pyplot as plt

def load_data():
    txt_path = os.path.join(os.path.dirname(__file__), 'old_faithful.txt')
    return np.loadtxt(txt_path)

def log_gaussian_pdf(X, mu, cov):
    N, D = X.shape if len(X.shape) > 1 else (1, X.shape[0])
    cov += np.eye(D) * 1e-6
    diff = X - mu
    inv_cov = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(cov)
    
    if len(X.shape) == 1:
        exponent = -0.5 * (diff.T @ inv_cov @ diff)
    else:
        exponent = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)
        
    return -0.5 * (D * np.log(2 * np.pi) + logdet) + exponent

def draw_contours(ax, data, mus, sigmas, phis):
    """绘制概率密度的等高线"""
    # 创建网格
    x = np.linspace(data[:, 0].min() - 0.5, data[:, 0].max() + 0.5, 100)
    y = np.linspace(data[:, 1].min() - 5, data[:, 1].max() + 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # 将网格展平以便批量计算 PDF
    grid_points = np.c_[X.ravel(), Y.ravel()]
    
    Z_total = np.zeros(grid_points.shape[0])
    colors = ['navy', 'darkorange']
    
    for k in range(len(mus)):
        # 计算该分量的对数概率并转回常规概率
        log_pdf = log_gaussian_pdf(grid_points, mus[k], sigmas[k])
        pdf = np.exp(log_pdf)
        
        # 绘制单个分量的淡色等高线
        zk = pdf.reshape(X.shape)
        ax.contour(X, Y, zk, levels=5, colors=colors[k % len(colors)], alpha=0.3)
        
        # 累加加权概率
        Z_total += phis[k] * pdf
    
    # 绘制混合分布的整体等高线轮廓
    Z_total = Z_total.reshape(X.shape)
    contour = ax.contour(X, Y, Z_total, levels=10, cmap='viridis', linewidths=1.5)
    ax.clabel(contour, inline=True, fontsize=8)

def em_gmm_contour_version(data, K=2, max_iter=100):
    N, D = data.shape
    # 初始化
    mus = data[np.random.choice(N, K, replace=False)]
    sigmas = np.array([np.cov(data.T) for _ in range(K)])
    phis = np.ones(K) / K
    
    # 迭代至收敛或达到上限
    for i in range(max_iter):
        # E-Step (数值稳定写法)
        log_weighted_probs = np.zeros((N, K))
        # 计算分子的对数概率
        for k in range(K):
            log_weighted_probs[:, k] = np.log(phis[k] + 1e-10) + log_gaussian_pdf(data, mus[k], sigmas[k])

        # 计算分母的对数概率
        log_sum_probs = np.logaddexp.reduce(log_weighted_probs, axis=1)
        # 概率计算 exp(log p - log sum_p) = p / sum_p
        gamma = np.exp(log_weighted_probs - log_sum_probs[:, np.newaxis])

        # M-Step
        Nk = np.sum(gamma, axis=0) + 1e-10
        prev_mus = mus.copy()
        
        phis = Nk / N
        for k in range(K):
            mus[k] = (gamma[:, k] @ data) / Nk[k]
            diff = data - mus[k]
            sigmas[k] = (gamma[:, k] * diff.T) @ diff / Nk[k] + np.eye(D) * 1e-6

        if np.allclose(prev_mus, mus, atol=1e-3):
            print(f"Converged at iteration {i}")
            break

    # 最终绘图
    fig, ax = plt.subplots(figsize=(10, 7))
    # 绘制数据点，颜色根据最终响应度硬分配
    labels = np.argmax(gamma, axis=1)
    ax.scatter(data[labels==0, 0], data[labels==0, 1], s=20, c='navy', alpha=0.4, label='Cluster 1')
    ax.scatter(data[labels==1, 0], data[labels==1, 1], s=20, c='darkorange', alpha=0.4, label='Cluster 2')
    
    # 核心：绘制等高线
    draw_contours(ax, data, mus, sigmas, phis)
    
    ax.set_title("GMM Clustering with Density Contours")
    ax.set_xlabel("Eruption duration (min)")
    ax.set_ylabel("Waiting time (min)")
    ax.legend()
    plt.show()

# 执行
raw_data = load_data()
em_gmm_contour_version(raw_data)