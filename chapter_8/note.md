### 数学原理

#### ddpm 加噪

单步加噪公式为：
$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$
其中 $\alpha_t = 1 - \beta_t$，$\beta_t$ 是预先定义的噪声调度参数。

t步加噪公式为：
$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$
其中 $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。

#### ddpm 去噪

单步去噪公式为：
$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sqrt{\beta_t} z, \quad z \sim \mathcal{N}(0, I)
$$
其中 $\epsilon_\theta(x_t, t)$ 是模型预测的噪声$

其证明位于 p188，证明的核心思想是利用条件概率的性质，将 $p(x_{t-1} | x_t)$ 表示为 $p(x_t | x_{t-1}) p(x_{t-1}) / p(x_t)$，然后通过计算 $p(x_t | x_{t-1})$ 和 $p(x_{t-1})$ 的分布，最终得到 $p(x_{t-1} | x_t)$ 的分布，从而得出去噪公式

### 训练效果

![](./figures/step_0800.png)
![](./figures/step_0600.png)
![](./figures/step_0400.png)
![](./figures/step_0200.png)
![](./figures/step_0000.png)

### 想法
- ddpm 模型的输出就是输入图像回溯time step的图像加噪到输入图像所用的噪声
- 在推理时，模型预测的是最终干净图像加噪到当前time step的噪声，然后将这个噪声当作上一个time step的图像加噪到当前time step的噪声，来还原得到上一个time step的图像；可以将模型输出看作是一个到最终干净图的梯度，不直接用来还原到干净图，而是在梯度方向上前进一小步，然后更新梯度，再前进一小步，直到还原到干净图像
