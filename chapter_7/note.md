### VAE 的改造过程

$$
\begin{aligned}
\log p_{\theta}(x) &= \int q(z) \log \frac{p_{\theta}(x, z)}{q(z)} dz + \int q(z) \log \frac{q(z)}{p_{\theta}(z|x)} dz \\
&= ELBO(q,\theta) + D_{KL}(q(z) \| p_{\theta}(z|x))
\end{aligned}
$$

$em$ 算法步骤：
1. **E 步**：计算 $q(z)$。
$$
q(z) = p_{\theta}(z|x)
$$
1. **M 步**：...更新 $\theta$。

VAE的潜变量 $z$ 为高维连续变量，E 步展开如下：
$$
\begin{aligned}
q(z) &= p_{\theta}(z|x) \\
&= \frac{p_{\theta}(x, z)}{p_{\theta}(x)} \\
&= \frac{p_{\theta}(x|z) p(z)}{\int p_{\theta}(x|z) p(z) dz}
\end{aligned}
$$
由于 $em$ 算法中 E 步的计算涉及到对潜变量 $z$ 的积分，且 $p_{\theta}(x|z)$ 和 $p(z)$ 的形式可能较为复杂，因此直接计算 $q(z)$ 是不可行的。VAE 通过引入一个参数化的近似分布 $q_{\phi}(z|x)$ 来替代 $q(z)$，并通过优化 ELBO 来学习模型参数 $\theta$ 和近似分布参数 $\phi$。此时 VAE 的 ELBO 定义为：
$$
ELBO(q_{\phi}, \theta) = \int q_{\phi}(z|x) \log \frac{p_{\theta}(x, z)}{q_{\phi}(z|x)} dz
$$
通过最大化 ELBO，VAE 同时优化模型参数 $\theta$ 和近似分布参数 $\phi$，从而实现对数据分布的有效建模和潜变量的学习。由于对于不同的数据点 $x$，近似分布 $q_{\phi}(z|x)$ 是不同的。在数据量较大时，使用 $em$ 算法，需要为每个数据点单独计算 E 步，这在计算上是不可行的。因此 VAE 使用了一个神经网络(也就是Encoder)来参数化 $q_{\phi}(z|x)$，使得模型能够适应不同的数据点并学习到更丰富的潜变量表示。

### VAE 的 损失函数推导

VAE 的损失函数由两部分组成：重构误差和 KL 散度。重构误差衡量了模型生成的数据与原始数据之间的差异，而 KL 散度则衡量了近似分布 $q_{\phi}(z|x)$ 与先验分布 $p(z)$ 之间的差异。
$$
\begin{aligned}
\mathcal{L}(\theta, \phi; x) &= -ELBO(q_{\phi}, \theta) \\
&= -\int q_{\phi}(z|x) \log \frac{p_{\theta}(x, z)}{q_{\phi}(z|x)} dz \\
&= -\int q_{\phi}(z|x) \log \frac{p_{\theta}(x|z) p(z)}{q_{\phi}(z|x)} dz \\
&= -\int q_{\phi}(z|x) \log p_{\theta}(x|z) dz - \int q_{\phi}(z|x) \log \frac{p(z)}{q_{\phi}(z|x)} dz \\
&= -\mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] + D_{KL}(q_{\phi}(z|x) \| p(z))
\end{aligned}
$$
其中，第一项 $-\mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)]$ 是重构误差，衡量了模型生成的数据与原始数据之间的差异；第二项 $D_{KL}(q_{\phi}(z|x) \| p(z))$ 是 KL 散度，衡量了近似分布与先验分布之间的差异。通过最小化这个损失函数，VAE 可以同时优化模型参数 $\theta$ 和近似分布参数 $\phi$，从而实现对数据分布的有效建模和潜变量的学习。

### 想法
- Encoder 过程就是 $q_{\phi}(z|x)$，Decoder 过程就是 $p_{\theta}(x|z)$
- 先验分布 $p(z)$ 通常被设定为标准正态分布，即 $p(z) = \mathcal{N}(0, I)$，这样可以简化 KL 散度的计算，并且有助于模型学习到更有意义的潜变量表示。
- 目前没看懂第一项损失为何是重构误差，后续再搞吧