import os
import numpy as np
import matplotlib.pyplot as plt


# gmm
def gmm(x, pi, mu, sigma):
    K = len(pi)
    p = np.zeros_like(x)
    for k in range(K):
        p += pi[k] * (1 / (np.sqrt(2 * np.pi) * sigma[k])) * np.exp(-0.5 * ((x - mu[k]) / sigma[k]) ** 2)
    return p

# 一维GMM绘制
def plot_gmm_1d(pi, mu, sigma, image_path):
    x = np.linspace(-10, 10, 1000)
    p = gmm(x, pi, mu, sigma)
    plt.plot(x, p)
    plt.title('1D Gaussian Mixture Model')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.grid()
    plt.savefig(image_path)
    plt.close()

# 二维GMM绘制
def plot_gmm_2d(pi, mu, sigma, image_path):
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    Z = np.zeros(X.shape)
    K = len(pi)

    for k in range(K):
        mu_k = np.array(mu[k])
        Sigma_k = np.array(sigma[k])
        det = np.linalg.det(Sigma_k)
        inv = np.linalg.inv(Sigma_k)

        # 计算每个像素点的高斯密度
        diff = pos - mu_k
        exponent = np.einsum('...i,ij,...j->...', diff, inv, diff)
        coef = 1.0 / (2 * np.pi * np.sqrt(det))
        Z += pi[k] * coef * np.exp(-0.5 * exponent)

    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.title('2D Gaussian Mixture Model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(contour, label='p(x, y)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(image_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    image_dir = 'chapter_4/figures'
    os.makedirs(image_dir, exist_ok=True)

    # 1D GMM example
    pi_1d = [0.5, 0.5]
    mu_1d = [-2, 2]
    sigma_1d = [1, 1]
    plot_gmm_1d(pi_1d, mu_1d, sigma_1d, os.path.join(image_dir, 'gmm_1d.png'))

    # 2D GMM example
    pi_2d = [0.5, 0.5]
    mu_2d = [[-2, -2], [2, 2]]
    sigma_2d = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
    plot_gmm_2d(pi_2d, mu_2d, sigma_2d, os.path.join(image_dir, 'gmm_2d.png'))