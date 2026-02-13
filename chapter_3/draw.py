import os
import numpy as np
import matplotlib.pyplot as plt

# 单变量正态分布
def normal_distribution(mu, sigma, x):
    coefficient = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return coefficient * exponent

# 多维正态分布
def multivariate_normal_distribution(mu, Sigma, X):
    mu = np.array(mu)
    Sigma = np.array(Sigma)
    d = len(mu)
    det_sigma = np.linalg.det(Sigma)
    inv_sigma = np.linalg.inv(Sigma)
    norm_const = 1.0 / (np.power((2 * np.pi), d / 2) * np.sqrt(det_sigma))

    diff = X - mu
    exponent = -0.5 * np.sum(diff @ inv_sigma * diff, axis=1)
    return norm_const * np.exp(exponent)

# 绘制单变量正态分布
def draw_normal_distribution(mu, sigma, x_range, image_path):
    x_values = np.linspace(x_range[0], x_range[1], 500)
    y_values = normal_distribution(mu, sigma, x_values)
    plt.figure()
    plt.plot(x_values, y_values)
    plt.title(f'Normal Distribution (mu={mu}, sigma={sigma})')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid()
    plt.savefig(image_path)
    plt.close()

# 绘制二维多变量正态分布
def draw_multivariate_normal_distribution(mu, Sigma, x_range, image_path):
    x, y = np.mgrid[x_range[0]:x_range[1]:0.05, x_range[0]:x_range[1]:0.05]
    pos = np.dstack((x, y))
    pos_flat = pos.reshape(-1, 2)
    z = multivariate_normal_distribution(mu, Sigma, pos_flat).reshape(x.shape)

    plt.figure()
    plt.contourf(x, y, z, cmap='viridis')
    plt.colorbar(label='Density')
    plt.title(f'Multivariate Normal (mu={mu})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(image_path)
    plt.close()

# 示例使用
if __name__ == "__main__":
    image_dir = 'chapter_3/figures'
    os.makedirs(image_dir, exist_ok=True)
    draw_normal_distribution(mu=0, sigma=1, x_range=(-5, 5), image_path=os.path.join(image_dir, 'normal_distribution.png'))
    
    mu = [0, 1]
    Sigma = [[1, 0.5], [0.5, 1]]  # 协方差矩阵
    draw_multivariate_normal_distribution(mu, Sigma, x_range=(-5, 5), image_path=os.path.join(image_dir, 'multivariate_normal_distribution.png'))
