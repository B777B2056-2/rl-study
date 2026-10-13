# 强化学习与 LLM 微调学习笔记

> **格式约定**：`##/###` 标题为主干内容，`>` 引用块为补充说明、小知识点、类比、注意事项。

---

# 目录

- [第一章：强化学习基本概念](#第一章强化学习基本概念)
- [第二章：传统强化学习](#第二章传统强化学习)
- [第三章：LLM 微调](#第三章llm-微调)
- [第四章：Agentic RL](#第四章agentic-rl)
- [第五章：Agentic RL：Tool RL](#第五章agentic-rltool-rl)

---

# 第一章：强化学习基本概念

## 1.1 核心要素

强化学习的基本设定：**智能体（Agent）在环境（Environment）中不断尝试动作（Action），获得奖励（Reward），并根据长期回报调整策略（Policy）。**

| 概念 | 符号 | 含义 |
|---|---|---|
| 状态 | $s$ | 环境的当前情况 |
| 动作 | $a$ | 智能体的选择 |
| 奖励 | $r$ | 环境的即时反馈 |
| 策略 | $\pi(a\|s)$ | 状态到动作的映射（概率分布） |
| 折扣因子 | $\gamma \in [0,1]$ | 未来奖励的折扣 |

**目标**：找到策略 $\pi^{\ast}$，使期望累计折扣奖励最大：

$$
\pi^{\ast} = \arg\max_\pi \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1}\right]
$$

> $\mathbb{E}_\pi$：在策略 $\pi$ 下、考虑环境和动作随机性后的期望。
>
> $R_{t+1}$：第 $t+1$ 步的即时奖励。
>
> $\gamma^t$：折扣系数，越靠后的奖励权重越小。

---

## 1.2 回报与价值

**回报（Return）**：从时刻 $t$ 开始的累计折扣奖励。

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

**状态价值 $V(s)$**：在状态 $s$ 下，按某策略走，平均能拿到的回报。

**动作价值 $Q(s,a)$**：在状态 $s$ 下，先做动作 $a$，之后按某策略走，平均能拿到的回报。

两者关系：

$$
V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a)
$$

**最优时**：

$$
V^{\ast}(s) = \max_a Q^{\ast}(s,a)
$$

> **V 和 Q 的区别**：
>
> - $V(s)$ 是"状态好不好"
> - $Q(s,a)$ 是"在这个状态下做这个动作好不好"
> - 同一个状态，不同动作的 Q 值不同，V 是它们的加权平均

---

## 1.3 贝尔曼方程

**贝尔曼期望方程（给定策略 π）**：

$$
V^\pi(s) = \mathbb{E}_\pi\left[r + \gamma V^\pi(s') \mid s\right]
$$

**贝尔曼最优方程**：

$$
Q^{\ast}(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^{\ast}(s',a') \mid s,a\right]
$$

> **两个方程的关键区别**：最优方程含 $\max$，期望方程不含。这决定了 Q-learning 和 SARSA 的区别。
>
> **贝尔曼方程的本质**：价值的递归定义——"现在值多少 = 即时奖励 + 未来值多少"。

---

## 1.4 策略分类

### On-Policy vs Off-Policy

- **行为策略**：实际和环境交互、产生数据的策略
- **目标策略**：你想要评估和改进的策略

| | On-Policy | Off-Policy |
|---|---|---|
| 行为策略 = 目标策略 | ✅ | ❌ |
| 数据来源 | 当前策略 | 旧数据/其他策略 |
| 数据复用 | 用完就丢 | 可反复用 |
| 代表 | SARSA、PPO | Q-learning、DQN、SAC |

> **Q-learning 为什么是 Off-policy**：更新时用 $\max_{a'} Q(s',a')$，**假设下一步选最优**，不管实际走了哪个。
>
> **SARSA 为什么是 On-policy**：更新时用 $Q(s',a')$，其中 $a'$ 是**实际选出的动作**。

---

## 1.5 学习方式分类

### 基于价值（Value-Based）

**学什么**：动作价值 $Q(s,a)$。**选动作**：$a = \arg\max_a Q(s,a)$。

**代表**：Q-learning、SARSA、DQN。

**损失函数**：

$$
L = \left(Q_\theta(s,a) - \text{target}\right)^2
$$

### 基于策略（Policy-Based）

**学什么**：策略 $\pi_\theta(a|s)$。

**代表**：REINFORCE、PPO、A2C。

**学习目标**：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \gamma^t r_t\right]
$$

**策略梯度**：

$$
\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)\right]
$$

### Actor-Critic（两者结合）

**Actor** 学策略，**Critic** 学价值。**代表**：PPO、A2C、SAC。

> **三者对比**：
>
> | | Value-Based | Policy-Based | Actor-Critic |
> |---|---|---|---|
> | 学什么 | Q 值 | 策略 | 两者 |
> | 连续动作 | ❌ | ✅ | ✅ |
> | 方差 | 低 | 高 | 中 |
> | 代表 | DQN | REINFORCE | PPO |

---

## 1.6 信用分配

**信用分配**：把最终奖励分配给前面哪些动作。

**问题**：奖励延迟、稀疏，一步错可能全错，但不知道是哪一步。

**解决方法**：

| 方法 | 粒度 | 代表 |
|---|---|---|
| 贝尔曼方程 | 每步 | Q-learning、DQN |
| GAE | 每步（加权） | PPO |
| 组内归一化 | 回答级 | GRPO |

**GAE**：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$
A_t = \delta_t + \gamma\lambda A_{t+1}
$$

> $\delta_t$：TD 误差（时序差分误差），"实际拿到的"和"原来预测的"差多少。
>
> $A_t$：优势，动作比平均水平好多少。
>
> $\lambda$：控制"看多远"，0.95 是常用值。

---

## 1.7 与监督学习的区别

| | 监督学习 | 强化学习 |
|---|---|---|
| 数据 | $(x,y)$ 成对 | 只有 $s$，无标准答案 |
| 反馈 | 立刻、精确 | 延迟、可能有噪声 |
| 样本 | i.i.d. | 时序相关 |
| 学习信号 | "该输出什么" | "这个好不好" |
| 探索 | 不需要 | 必须 |

> **核心差异**：监督学习告诉你"**怎么做**"；强化学习只告诉你"**做得怎么样**"，具体怎么改要自己摸索。
>
> **类比**：
>
> - 监督学习 = 老师手把手教，每道题给标准答案
> - 强化学习 = 学生自己做题，老师只给一个分数

**在 LLM 里的对应**：

| | SFT（监督） | PPO（强化） |
|---|---|---|
| 数据 | (prompt, response) | prompt |
| 学习信号 | 每 token 的标签 | RM 分数 + KL |
| 探索 | ❌ | ✅ |
| 参考模型 | ❌ | ✅ |
| 稳定性 | 稳定 | 需调参 |

---

# 第二章：传统强化学习

## 2.1 Q-learning

### 学习目标

**逼近贝尔曼最优方程**：

$$
Q^{\ast}(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^{\ast}(s',a')\right]
$$

### 更新公式

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
$$

> $\alpha$：学习率，控制每次修正幅度，通常 0.1。
>
> $\gamma$：折扣因子，通常 0.99。
>
> 方括号整体：**TD 误差**，即"新估计和旧估计的差"。

**TD 误差**：

$$
\delta = \underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{TD Target}} - Q(s,a)
$$

> $\delta > 0$：实际比估计好 → 提高 Q 值。
>
> $\delta < 0$：实际比估计差 → 降低 Q 值。
>
> $\delta = 0$：估计准确，收敛。

### ε-贪婪探索

$$
a = \begin{cases} \text{随机动作} & \text{概率 } \epsilon \\ \arg\max_a Q(s,a) & \text{概率 } 1-\epsilon \end{cases}
$$

**ε 指数衰减**：

$$
\epsilon_t = \max(\epsilon_{\min}, \epsilon_0 \cdot \lambda^t)
$$

> $\epsilon_0$：初始探索率，通常 1.0。
>
> $\lambda$：衰减率，通常 0.995。
>
> $\epsilon_{\min}$：最低探索率，通常 0.05。

### 算法流程

```
1. 初始化 Q 表为 0
2. 对每个回合：
   a. 重置环境，得到初始状态 s
   b. 对每一步：
      - 用 ε-贪婪选动作 a
      - 执行 a，得到 r, s', done
      - 用公式更新 Q(s,a)
      - s ← s'
   c. 衰减 ε
```

### 特点

- **Off-policy**
- **基于价值**
- 只能处理离散、小规模状态

---

## 2.2 SARSA

### 学习目标

**逼近贝尔曼期望方程**：

$$
Q^\pi(s,a) = \mathbb{E}_\pi\left[r + \gamma Q^\pi(s',a')\right]
$$

### 更新公式

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma Q(s',a') - Q(s,a) \right]
$$

> **和 Q-learning 的唯一区别**：
>
> - Q-learning 用 $\max_{a'} Q(s',a')$
> - SARSA 用 $Q(s',a')$（实际选出的动作）

### 算法流程

```
1. 初始化 Q 表为 0
2. 对每个回合：
   a. 重置环境，得到 s
   b. 用 ε-贪婪选初始动作 a
   c. 对每一步：
      - 执行 a，得到 r, s', done
      - 用 ε-贪婪选下一动作 a'
      - 用公式更新 Q(s,a)（注意用 Q(s',a')）
      - s ← s', a ← a'
   d. 衰减 ε
```

### CliffWalking 例子

```
起点 S . . . . . . . . . . 终点 G
      X X X X X X X X X X 悬崖
```

> - Q-learning 学出**贴着悬崖走**的最短路径（假设不会探索到悬崖）
> - SARSA 学出**绕远路但更安全**的路径（考虑探索风险）

---

## 2.3 DQN

### 学习目标

**和 Q-learning 一样，逼近贝尔曼最优方程**，但用神经网络代替 Q 表：

$$
Q(s,a) \approx Q_\theta(s,a)
$$

### 关键技巧 1：经验回放

存储 $(s, a, r, s', done)$ 五元组，训练时随机采样一个 batch。

> **作用**：打破数据的时间相关性，提高样本效率。容量通常 1 万~100 万。

### 关键技巧 2：目标网络

$$
\text{target} = r + \gamma \max_{a'} Q_{\theta^-}(s',a')
$$

> $Q_{\theta^-}$：目标网络，参数 $\theta^-$ 定期从主网络复制。
>
> **作用**：让目标值稳定，避免"追自己尾巴"的震荡。
>
> **同步频率**：每 100~1000 步复制一次。

### 损失函数

$$
L(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a)\right)^2\right]
$$

### 算法流程

```
1. 初始化主网络 Q_θ 和目标网络 Q_θ-
2. 初始化回放池
3. 对每个回合：
   a. 重置环境，得到 s
   b. 对每一步：
      - 用 ε-贪婪选动作 a
      - 执行 a，得到 r, s', done
      - 存 (s, a, r, s', done) 到回放池
      - 从回放池采样一个 batch
      - 算 target = r + γ·max Q_θ-(s',a')
      - 算 loss = MSE(Q_θ(s,a), target)
      - 反向传播更新 θ
      - 每 N 步同步 θ- ← θ
   c. 衰减 ε
```

### Double DQN

$$
\text{target} = r + \gamma Q_{\theta^-}\left(s', \arg\max_{a'} Q_\theta(s',a')\right)
$$

> **作用**：缓解 Q 值高估。目标网络的噪声被 $\max$ 放大时会系统性高估；分开选和评估后，噪声不会被放大。

---

## 2.4 PPO（传统版）

### 学习目标

**直接最大化期望回报**，用 clip 限制每次策略更新幅度：

$$
\max_\theta \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A(s,a)\right]
$$

### 两个网络

- **Actor** $\pi_\theta(a|s)$：输出动作概率，被优化
- **Critic** $V_\phi(s)$：估计状态价值，用来算优势

### 策略损失（Clip）

$$
L^{\text{policy}} = -\mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]
$$

> $r_t = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$：**概率比**，新旧策略在同一动作上的概率之比。
>
> $A_t$：**优势**，表示动作比平均水平好多少。
>
> $\epsilon$：clip 范围，通常 0.2。
>
> $\min$：取两者较小的，保证优化目标保守。

**含义**：

- 好动作（$A > 0$）：提高概率，但 $r_t$ 超过 $1+\epsilon$ 就截断
- 坏动作（$A < 0$）：降低概率，但 $r_t$ 低于 $1-\epsilon$ 就截断

### 价值损失

$$
L^{\text{value}} = \mathbb{E}\left[(V_\phi(s_t) - R_t)^2\right]
$$

### 熵

$$
H = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)
$$

> 衡量策略的随机性，越大越随机。加入损失时前面是**负号**（最大化熵），鼓励探索。

### 总损失

$$
L = L^{\text{policy}} + 0.5 L^{\text{value}} - 0.01 H
$$

### 优势估计（GAE）

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

$$
A_t = \delta_t + \gamma\lambda A_{t+1}
$$

$$
R_t = A_t + V_\phi(s_t)
$$

> $\lambda$：GAE 参数，通常 0.95。
>
> - $\lambda = 0$：只用一步 TD 误差，偏差大、方差小
> - $\lambda = 1$：用完整回报，偏差小、方差大

### 算法流程

```
1. 用当前 Actor 和环境交互，收集一批轨迹
2. 用 Critic 算 V(s)，用 GAE 算优势 A(s,a)
3. 优势归一化
4. 多轮（epochs）更新：
   a. 用当前 Actor 算新 log_prob
   b. 算概率比 ratio = exp(new - old)
   c. 算 clip 策略损失
   d. 算 Critic 的 MSE 损失
   e. 算熵
   f. 总损失反向传播
5. 清空数据，回到第 1 步
```

---

# 第三章：LLM 微调

## 3.1 SFT（监督微调）

### 学习目标

**让模型学会"给定 prompt，输出 response"**，最大化 response 每个 token 的对数概率。

### 数据构造

```
input_ids = [prompt tokens] + [response tokens] + [EOS]
labels    = [-100] * len(prompt) + [response tokens] + [EOS]
```

> prompt 部分 labels 设为 -100，因为 prompt 是输入，不是目标。
>
> -100 是 PyTorch `cross_entropy` 的 `ignore_index`，会跳过这些位置。

### 损失函数

$$
L^{\text{SFT}} = -\frac{1}{N}\sum_{t=1}^{N} \log \pi_\theta(y_t \mid y_{\lt t}, x)
$$

> $x$：prompt。
>
> $y_t$：response 的第 $t$ 个 token。
>
> $y_{\lt t}$：response 的前 $t-1$ 个 token。
>
> $N$：response 的 token 数。

### Teacher Forcing（小知识点）

> **什么是 Teacher Forcing**：训练时，把标准答案作为输入，让模型预测下一个 token。"Teacher"是标准答案，"Forcing"是强制把标准答案喂给模型，而不是用模型自己上一步生成的 token。
>
> **为什么需要**：语言模型是自回归的，第 $t$ 个 token 的预测依赖前 $t-1$ 个 token。
>
> **如果不用 Teacher Forcing**（用模型自己上一步的输出当下一步输入）：
>
> - 第 1 步错了，第 2 步输入也错，错误累积
> - 预测 N 个 token 要前向 N 次，串行，慢
>
> **Teacher Forcing 的做法**：一次前向，把整条标准答案喂进去，所有位置并行预测（上面的 `数据构造` 和后面的 `Shift` 小节就是在做这件事）。
>
> **好处**：
>
> - 一次前向，所有位置都预测（并行，快）
> - 训练不累积错误
> - 训练信号干净
>
> **为什么模型不能"抄答案"**：关键在**因果 mask（Causal Mask）**。自回归模型的注意力是因果的，每个位置只能看到自己和左边的 token。
>
> ```
> 位置 4（"?"）能看到：位置 0~4（1, +, 1, =, ?）
> 看不到位置 5（"2"）  ← 被 mask 挡住
> ```
>
> 所以位置 4 要预测 "2"，但它的输入里没有 "2"，必须根据 `"1+1=?"` 自己算出。
>
> **训练 vs 推理**：
>
> | | 训练（Teacher Forcing） | 推理（自回归生成） |
> |---|---|---|
> | 输入 | 标准答案 | 只有 prompt |
> | 生成方式 | 一次前向，并行 | 逐 token 串行 |
> | 前向次数 | 1 次 | N 次 |
> | 错误 | 不累积 | 会累积 |
>
> **类比**：用 Teacher Forcing = 老师手把手纠正，每一步都从正确的起点开始；不用 = 孩子写错字继续往下写，错误越滚越大。

### Shift

语言模型是"用前 $t$ 个 token 预测第 $t+1$ 个"，所以：

```python
shift_logits = logits[:, :-1, :]     # 位置 0~T-2 的预测
shift_labels = labels[:, 1:]         # 位置 1~T-1 的标签
```

> **差一位**，容易错。

### 算法流程

```
1. 加载模型和 tokenizer
2. 构造数据：prompt + response，labels 里 prompt 部分设 -100
3. 训练循环：
   a. 前向：logits = model(input_ids)
   b. Shift：logits[:, :-1] 对 labels[:, 1:]
   c. 算损失：cross_entropy(logits, labels, ignore_index=-100)
   d. 反向传播 + 更新
4. 保存模型
```

### 训练配置

| 参数 | 典型值 |
|---|---|
| 学习率 | 2e-5（全参）/ 2e-4（LoRA） |
| epochs | 1~3 |
| batch_size | 1~8 |
| warmup | 总步数的 3%~10% |
| 梯度裁剪 | 1.0 |
| 调度器 | Warmup + Cosine |

> **Warmup + Cosine**：学习率先从 0 线性升到最大（warmup），再从最大余弦降到 0。

---

## 3.2 LoRA（低秩适配）

### 学习目标

**用少量可训练参数，逼近全参数微调的效果。**

> **核心假设**：微调时权重变化 $\Delta W$ 是低秩的，可以用两个小矩阵近似。

### 公式

$$
y = Wx + \frac{\alpha}{r} BAx
$$

> $W$：原始权重 $(out, in)$，**冻结不更新**。
>
> $A$：降维矩阵 $(r, in)$，**训练**。
>
> $B$：升维矩阵 $(out, r)$，**训练**。
>
> $r$：秩，通常 8。
>
> $\alpha$：缩放因子，通常 $2r$。

**PyTorch 行向量形式**（`nn.Linear` 内部是 `x @ W.T`）：

$$
y = x @ W^T + \text{scaling} \cdot x @ B @ A
$$

> **维度验证**：
>
> ```
> x:      (B, in)
> B:      (in, r)
> A:      (r, out)
>
> x @ B @ A: (B, out)   ← 和 x @ W.T 一致
> ```

### 初始化

- **A**：小随机数（$N(0, 0.01)$），打破对称性
- **B**：**全 0**，保证初始时 $BA = 0$

> **为什么 B 全 0**：初始时 $BA = 0$，$y = Wx$，和原模型一致。
>
> **为什么 A 用随机数**：如果 A 也全 0，梯度是 0，学不动。

### 参数量

$$
\text{比例} = \frac{r(in+out)}{in \cdot out}
$$

> **例**：$in = out = 896$，$r = 8$ → 1.8%。

### 合并

$$
W_{\text{new}} = W + \frac{\alpha}{r} BA
$$

> 合并后 LoRA 结构消失，变成普通 `nn.Linear`，推理不需要额外计算。

### 关键超参数

| 参数 | 典型值 |
|---|---|
| `r` | 8 |
| `alpha` | 16 |
| `dropout` | 0.05 |
| `target_modules` | q_proj, k_proj, v_proj, o_proj |

> **target_modules 选择**：
>
> - `q_proj, v_proj`：最少参数
> - `q, k, v, o`：推荐
> - 加上 MLP：接近全参

---

## 3.3 QLoRA（4-bit 量化 LoRA）

### 学习目标

**和 LoRA 一样**，但 base 权重从 bf16 量化到 4-bit，省 75% 显存。

### 三个关键技术

**1. NF4（4-bit NormalFloat）**

> 普通 int4 是均匀量化，浪费了正态分布集中区域的精度。
>
> NF4 按正态分布的分位数取 16 个值，精度比 int4 高。

**2. 双重量化**

> 每个 block 的 scale 也量化到 8-bit，每参数额外开销从 0.5 bit 降到 0.127 bit。

**3. 分页优化器**

> 显存不够时把优化器状态换到 CPU 内存。

### 反量化

**量化**：

$$
q = \text{round}(x / s)
$$

**反量化**：

$$
\hat{x} = q \times s
$$

> **QLoRA 前向**：
>
> 1. 读 4-bit 权重 $W_q$
> 2. 反量化到 bf16：$W = W_q \times s$
> 3. 算矩阵乘法
> 4. 丢弃 W
>
> 每次前向都要反量化，所以比 LoRA 慢 20~40%。

### 和 LoRA 的对比

| | LoRA | QLoRA |
|---|---|---|
| base 精度 | bf16 | 4-bit |
| 显存（7B） | 16~18 GB | 6~8 GB |
| 速度 | 快 | 慢 20~40% |
| 效果 | 好 | 略低 1~2% |

### 合并

> **QLoRA 不能直接合并**（4-bit 和 bf16 不匹配）。正确流程：
>
> 1. 加载 bf16 base（不是 4-bit）
> 2. 注入 LoRA 结构
> 3. 加载 LoRA 参数
> 4. merge_lora
> 5. 保存为 bf16

---

## 3.4 PPO（LLM 版）

### 学习目标

**在 SFT 模型的基础上，用 RM 的分数优化策略，同时用 KL 惩罚防止偏离 SFT 太远。**

$$
\max_\pi \mathbb{E}_{x, y \sim \pi}\left[r_{\text{RM}}(x,y) - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})\right]
$$

> $r_{\text{RM}}$：RM 对回答的打分。
>
> $\pi_{\text{ref}}$：参考模型（SFT）。
>
> $\beta$：KL 系数，控制偏离程度。

### 和传统 PPO 的关系

**数学核心完全相同**：TD 误差、GAE、Clip 损失、优势归一化。

| 维度 | 传统 PPO | LLM PPO |
|---|---|---|
| 状态 | 4 维向量 | prompt + 已生成 token |
| 动作 | 2 个 | 词表大小（15 万） |
| 奖励 | 每步环境给 | 末尾 RM + 中间 KL |
| 数据形状 | $(N,)$ | $(B, R)$ |
| 参考模型 | 无 | **有** |
| 折扣 γ | 0.99 | **1.0** |
| 学习率 | 3e-4 | **1e-6 ~ 2e-4** |
| epochs | 10 | **1~4** |

### 四个模型

| 模型 | 作用 | 训练 |
|---|---|---|
| **Actor** | 生成回答 | LoRA |
| **Critic** | 估计 V(s) | value_head |
| **Reference** | 算 KL | 冻结 |
| **Reward Model** | 打分 | 冻结 |

**结构**：

```
共享 base（SFT 模型，冻结）
    ├── Actor：base + LoRA（训练）
    ├── Critic：base + value_head（训练）
    └── Reference：base（冻结）
```

> Critic 前向时**禁用 LoRA**，只训练 `value_head`（约 900 参数）。
>
> Reference 直接用 `base_model`，不额外复制。

### KL 惩罚

$$
\text{KL}_t = \log \pi_\theta(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t)
$$

$$
r_t^{\text{KL}} = -\beta \cdot \text{KL}_t
$$

### 奖励设计

$$
r_t = \begin{cases} -\beta \cdot \text{KL}_t & t \lt R \\ -\beta \cdot \text{KL}_t + r^{\text{RM}} & t = R \end{cases}
$$

> **中间 token**：只有 KL 惩罚。
>
> **末尾**：KL + RM 分数。
>
> **为什么**：RM 只能给整段打分；KL 每 token 都有，防止偏离；GAE 把末尾分数"分配"到前面。

### GAE（token 级）

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$
A_t = \delta_t + \gamma\lambda A_{t+1}
$$

$$
R_t = A_t + V(s_t)
$$

### PPO 更新

$$
L^{\text{policy}} = -\mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 1\pm\epsilon) A_t\right)\right]
$$

$$
L^{\text{value}} = \mathbb{E}\left[(V(s_t) - R_t)^2\right]
$$

$$
L = L^{\text{policy}} + 0.5 L^{\text{value}}
$$

> **LLM 不加熵**：词表太大（15 万），算熵太贵。

### 算法流程

```
1. 采样：从 DataLoader 取 prompt，Actor 生成回答，记录 log_prob
2. 算 KL：Actor 和 Reference 各前向一次
3. 算奖励：中间 r_t = -β·KL_t，末尾 + RM
4. 算优势（GAE）：反向累积
5. PPO 更新：重新算 new_log_prob（带梯度），算 ratio，clip 损失 + value 损失
6. 清空数据，回到第 1 步
```

> **为什么 LLM PPO 更新轮数少**：
>
> - 前向是 LLM，几十毫秒，太慢
> - On-policy，多轮会偏离 old 策略太远

### 显存优化

| 优化 | 省显存 |
|---|---|
| Actor / Critic 共享 backbone | 1 GB |
| Reference 用 base_model（不 deepcopy） | 1 GB |
| batch_size=1, max_length=256 | 关键 |
| 梯度检查点 | 0.5 GB |
| 规则奖励代替 RM | 1 GB |

---

## 3.5 DPO（直接偏好优化）

### 学习目标

**在 SFT 模型基础上，用偏好对直接优化策略，跳过 RM 和 RL 循环。**

$$
\max_\pi \mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

> 形式和 RLHF 目标一致，但**数据是固定的偏好对**。

### 核心洞见：解析解

**RLHF 目标的最优策略有闭式解**：

$$
\pi^{\ast}(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{r_{\text{RM}}(x,y)}{\beta}\right)
$$

> $Z(x)$：归一化常数（对所有 $y$ 求和）。
>
> **含义**：最优策略是"参考策略"乘上"奖励的指数"。

**反解出奖励**：

$$
r_{\text{RM}}(x,y) = \beta \log \frac{\pi^{\ast}(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

> **含义**：**奖励 = β × 策略相对参考的对数概率比 + 常数**
>
> **这一步是 DPO 的核心洞见**：奖励不需要单独学，它已经"藏在"策略里了。

**代入 Bradley-Terry 后 $\log Z(x)$ 被差分消掉**：

$$
r(x,y_w) - r(x,y_l) = \beta \log \frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

> **关键**：偏好模型只需要"两个回答的奖励差"，不需要奖励本身，所以 $\log Z(x)$ 消掉。

### 损失函数

$$
L_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

**逐项理解**：

| 项 | 含义 |
|---|---|
| $\pi_\theta$ | 当前策略（Policy，要训练） |
| $\pi_{\text{ref}}$ | 参考模型（Reference，冻结） |
| $y_w$ | chosen（更好的回答） |
| $y_l$ | rejected（更差的回答） |
| $\beta$ | KL 系数，通常 0.1~0.5 |
| $\sigma$ | sigmoid |

**sigmoid 内的内容**：

$$
\underbrace{\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}}_{\text{chosen 的隐式奖励}} - \underbrace{\beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}}_{\text{rejected 的隐式奖励}}
$$

**含义**：让"chosen 相对参考的偏好程度"比"rejected 相对参考的偏好程度"更大。

> **为什么用"相对参考"而不是绝对概率**：
>
> - 直接用 $\log \pi_\theta(y_w)$ 和 $\log \pi_\theta(y_l)$，训练会让所有回答的概率都变大或变小，不稳定
> - 用比值 $\log \frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)}$：含义是"相比 SFT 模型，当前策略在 $y$ 上提高了多少"，是相对提升
> - 参考模型起"锚点"作用，类似 PPO 的 KL 惩罚

### 和 PPO 的关系

**数学上等价**：DPO 是 RLHF 目标的解析解。

**工程上不同**：

| 维度 | PPO | DPO |
|---|---|---|
| 模型数 | 4 | 2 |
| 采样 | ✅ | ❌ |
| 每步前向 | 多次（生成 + KL + Critic） | **4 次** |
| 损失 | clip + value | sigmoid |
| 训练速度 | 慢 | 快 |
| 稳定性 | 难调 | 稳定 |

> **关键区别：数据来源**
>
> - PPO 优化的是"策略能生成的所有回答"（在线采样）
> - DPO 优化的是"数据里出现的偏好对"（离线数据）
> - **两者目标函数一样，但数据来源不同，所以最优解不同**
>
> **DPO 在"已知偏好"上做得好；PPO 能"发现新的偏好"。**
>
> 只有当数据是"上帝视角"（覆盖所有可能），两者才等价。现实中数据永远不完美，所以两者效果不同，互补。

### 数据结构

每条偏好对：

```json
{
  "prompt": "1+1=?",
  "chosen": "1+1=2，所以答案是 2。#### 2",
  "rejected": "1+1=2，所以答案是 3。#### 3"
}
```

**返回 batch**：

```python
{
    "input_ids":               (B, P),
    "input_attention_mask":    (B, P),
    "chosen_ids":              (B, R1),
    "chosen_attention_mask":   (B, R1),
    "rejected_ids":            (B, R2),
    "rejected_attention_mask": (B, R2),
}
```

> **注意**：prompt 和 response **分开存**，训练时再拼接。

### 核心实现：`_calc_log_prob`

**功能**：对已有的 response 算一次前向的 log 概率（给已有回答打分，不生成）。

**关键点**：

> **`gather` 的作用**：从"整个词表的 log 概率 `(B, R, V)`"里，挑出"实际那个 token"的 log 概率 `(B, R)`。
>
> **为什么用 `prompt_len-1:-1`**：语言模型是"用位置 $t$ 预测位置 $t+1$"，所以差一位。
>
> - 位置 `P-1` 的 logits 预测 response 第 1 个 token
> - 位置 `P+R-2` 的 logits 预测 response 第 R 个 token
>
> **为什么整段求和**：整段 response 的 log 概率 = 每个 token log 概率之和。
>
> **为什么用 `response_attention_mask`**：padding 位置的 log_prob 是"假的"，要置零。

### 算法流程

```
1. 从数据集取一个偏好对 (prompt, chosen, rejected)
2. Policy 前向 2 次：算 logp_chosen, logp_rejected（带梯度）
3. Reference 前向 2 次：算 ref_chosen, ref_rejected（no_grad）
4. 算隐式奖励差：
     logits = β × ((logp_chosen - ref_chosen) - (logp_rejected - ref_rejected))
5. 算损失：
     loss = -logsigmoid(logits).mean()
6. 反向传播 + 更新 Policy
7. 下一个 batch
```

**4 次前向 + 1 次反向，和 SFT 一样简单。**

> **DPO 里没有"生成"**：response 是数据集给的，模型只负责"给已有 response 打分"。
>
> 对比 PPO：
>
> - PPO 采样时：Actor 生成 response（R 次前向）
> - DPO：不生成，直接对数据里的 chosen / rejected 做一次前向

### 训练配置

| 参数 | 典型值 |
|---|---|
| `learning_rate` | 5e-7 ~ 5e-6 |
| `beta` | 0.1 ~ 0.5 |
| `batch_size` | 1 ~ 8 |
| `n_epoch` | 1 ~ 3 |
| `max_length` | 512 ~ 1024 |
| `max_prompt_length` | 256 |
| `grad_clip_eps` | 1.0 |

> **beta 的作用**：
>
> | β | 效果 |
> |---|---|
> | 0.01 | 偏离参考大，可能过拟合 |
> | **0.1** | **标准** |
> | 0.5 | 保守，学得慢 |
> | 1.0 | 太保守，几乎不动 |
>
> beta 越大，越"紧贴"参考模型。

### 评估指标

**准确率**：`chosen_reward > rejected_reward` 的比例。

```python
acc = ((policy_chosen_logp - ref_chosen_logp)
       > (policy_rejected_logp - ref_rejected_logp)).float().mean()
```

**训练成功标志**：`acc` 从 ~0.5 升到 ~0.9，`loss` 从 ~0.69 降到 ~0.3。

### 显存占用（0.5B + 6GB）

| 组件 | 显存 |
|---|---|
| Policy（bf16 + LoRA） | 1 GB |
| Reference（bf16，冻结） | 1 GB |
| 激活值 | 1~2 GB |
| **总计** | **约 3~4 GB** ✅ |

**比 PPO 省 1~2 GB**（少 Critic 和 RM）。

---

## 3.6 四者关系

### 算法演进

```
SFT（监督学习）
    ↓ 教模型"怎么说话"
    ↓ 数据：(prompt, response)
    ↓ 损失：交叉熵
    ↓ 模型：1 个
    ↓

RM（奖励模型）
    ↓ 教模型"什么是好回答"
    ↓ 数据：(prompt, chosen, rejected)
    ↓ 损失：Bradley-Terry
    ↓

PPO / DPO
    ↓ 让模型"回答更好"
    ├── PPO：在线采样 + 4 个模型
    └── DPO：离线偏好 + 2 个模型
```

### 对比总表

| 维度 | SFT | PPO | DPO |
|---|---|---|---|
| 数据 | (prompt, response) | prompt | (prompt, chosen, rejected) |
| 学习方式 | 监督 | 强化 | 监督（隐式强化） |
| 损失 | 交叉熵 | clip + value | sigmoid |
| 需要 RM | ❌ | ✅ | ❌ |
| 需要 Critic | ❌ | ✅ | ❌ |
| 需要采样 | ❌ | ✅ | ❌ |
| 需要 Ref | ❌ | ✅ | ✅ |
| 模型数 | 1 | 4 | 2 |
| 显存 | 低 | 高 | 中 |
| 稳定性 | 稳定 | 需调参 | 稳定 |

### 阶段速查（学习目标 + 核心公式）

| 阶段 | 学习目标 | 核心公式 |
|---|---|---|
| SFT | 最大化 response 每个 token 的对数概率 | $L = -\sum_t \log \pi_\theta(y_t \mid y_{\lt t})$ |
| LoRA | 用低秩增量逼近全参微调的效果 | $y = Wx + \frac{\alpha}{r}BAx$ |
| QLoRA | 同 LoRA，但省显存 | 同 LoRA，但 base 是 4-bit |
| PPO | 最大化 RM 分数，同时不偏离 SFT 太远 | $L = L^{\text{policy}} + 0.5 L^{\text{value}}$ |
| DPO | 让 chosen 的隐式奖励 > rejected 的隐式奖励 | $L = -\log \sigma(\beta \log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})$ |

### PPO vs DPO 的哲学区别

**PPO**：

- 优化"策略能生成的所有回答"
- 在线采样，能探索数据外的好回答
- 上限高，但调参难、成本高

**DPO**：

- 优化"数据里出现的偏好对"
- 离线数据，学不到数据外的好回答
- 稳定、简单、成本低

> **类比**：
>
> - DPO = 从题库学，能学会题库里的偏好
> - PPO = 自己做题 + 打分，能探索题库外
>
> **两者目标函数一样，但数据来源不同，所以最优解不同，互补。**

### 完整链路

**方案 A：SFT → RM → PPO**

```
1. SFT
   - 输入：原始模型 + (prompt, response)
   - 输出：SFT 模型
        ↓
2. 训练 RM
   - 输入：SFT 模型主干 + (prompt, chosen, rejected)
   - 输出：RM
        ↓
3. PPO
   - Actor：SFT 模型 + 新 LoRA
   - Critic：SFT 模型 + value_head
   - Reference：SFT 模型（冻结）
   - RM：冻结
   - 输出：PPO 模型
```

**方案 B：SFT → DPO**

```
1. SFT
   - 输入：原始模型 + (prompt, response)
   - 输出：SFT 模型
        ↓
2. DPO
   - Policy：SFT 模型 + 新 LoRA
   - Reference：SFT 模型（冻结）
   - 数据：(prompt, chosen, rejected)
   - 输出：DPO 模型
```

**方案 C：SFT → DPO → PPO（推荐）**

```
SFT → DPO（暖启动）→ PPO（精调）
```

> **为什么 PPO 前先做 DPO**：
>
> 1. **暖启动**：给 PPO 一个更好的起点，不需要从"完全不懂偏好"开始
> 2. **稳定性**：减少 PPO 震荡、发散
> 3. **抗刷分**：DPO 已经学过"什么格式是好的"，不容易被 RM 骗
> 4. **成本低**：DPO 便宜，PPO 贵，先用便宜的把模型拉起来
>
> **类比**：SFT 是小学，DPO 是中学，PPO 是大学。从小学直接跳大学容易崩。

### LoRA / QLoRA 在其中的位置

> **LoRA 是一种"参数高效微调方法"，可以用于 SFT、PPO、DPO 的任何阶段**：
>
> - SFT 阶段：用 LoRA 微调，省显存
> - PPO 阶段：Actor 用新 LoRA，Critic 用 value_head
> - DPO 阶段：Policy 用新 LoRA，Reference 冻结
>
> **QLoRA 是 LoRA 的显存优化版**：
>
> - 只在"base 太大放不下"时使用（7B+）
> - 小模型（0.5B）用普通 LoRA 即可

### 数据复用

| 阶段 | 可用数据 |
|---|---|
| SFT | (prompt, response) |
| PPO | prompt（自己采样） |
| DPO | (prompt, chosen, rejected) |

**GSM8K 场景**：

- SFT：题目 + 标准答案
- PPO：题目（自己生成答案，用规则打分）
- DPO：题目 + 正确答案 + 错误答案

### 选择建议

| 场景 | 推荐 |
|---|---|
| 入门，显存紧张 | SFT（LoRA）→ DPO |
| 追求效果，资源充足 | SFT → RM → PPO |
| 可验证任务（数学、代码） | SFT → PPO + 规则奖励 |
| 单卡 6GB | SFT（LoRA）→ DPO |
| 想要简单稳定 | SFT → DPO |
| 想要上限高 | SFT → DPO → PPO |


---

## 3.7 GRPO（组相对策略优化）

### 学习目标

**用同一 prompt 下多个回答的相对排名代替 Critic，省显存、更稳定。**

形式化目标：

$$
\max_\pi \mathbb{E}_{x, \{y_i\} \sim \pi}\left[r_{\text{RM}}(x, y_i) - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})\right]
$$

> $x$：prompt
>
> $\{y_i\}$：从策略采样的 G 个回答
>
> $r_{\text{RM}}$：RM 打分
>
> $\pi_{\text{ref}}$：参考模型（SFT）
>
> $\beta$：KL 系数

### 核心洞见：不需要 Critic

**PPO 的问题**：Critic 和 Actor 一样大，显存翻倍。

**GRPO 的思路**：

> 对同一个 prompt 采样 G 个回答，用**组内相对分数**作为优势，**不需要 Critic 估计 V(s)**。

### 组内归一化

对同一个 prompt 的 G 个回答：

$$
\{y_1, y_2, \ldots, y_G\} \sim \pi_\theta(\cdot | x)
$$

用 RM 打分：

$$
r_i = r_{\text{RM}}(x, y_i), \quad i = 1, \ldots, G
$$

**组内归一化得到优势**：

$$
A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}
$$

> $\text{mean}(r)$：这 G 个回答的平均分。
>
> $\text{std}(r)$：标准差。
>
> **含义**：回答 $i$ 比“同组平均”好多少，用标准差归一化。

**举例**：
> prompt = "1+1=?"  
> 采 4 个回答：  
> y_1 = "2" → r_1 = 1.0  
> y_2 = "3" → r_2 = -1.0  
> y_3 = "2" → r_3 = 1.0  
> y_4 = "4" → r_4 = -1.0  

> mean = 0, std = 1  
> A_1 = 1.0 ← 好回答，正优势  
> A_2 = -1.0 ← 坏回答，负优势  


**直觉**：

- 好回答：$A > 0$ → 提高概率
- 坏回答：$A < 0$ → 降低概率
- **不需要 Critic 估 V(s)，平均分天然是基线**

### 策略损失（和 PPO 一样）

$$
L^{\text{policy}} = -\mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 1\pm\epsilon) A_t\right)\right]
$$

> $r_t = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$：概率比。
>
> $A_t$：**组内归一化后的优势**（PPO 是 GAE 算的）。
>
> $\epsilon$：clip 范围，通常 0.2。
>
> **形式和 PPO 完全一样**，只是优势的来源不同。

### 总损失

$$
L = L^{\text{policy}} + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

> **没有 value loss**（因为没 Critic）。
>
> KL 惩罚可以加进损失，也可以只用于监控。

### 和 PPO 的对比

| 维度 | PPO | GRPO |
|---|---|---|
| Critic | ✅ 需要 | ❌ **不需要** |
| 优势估计 | GAE（需 V(s)） | **组内归一化** |
| 每个 prompt 采样 | 1 个回答 | **G 个回答** |
| 模型数 | 4 | **3** |
| 显存 | 高 | **省 1 个 Critic** |
| 策略损失 | clip | **完全一样** |
| KL 惩罚 | ✅ | ✅ 一样 |
| On-policy | ✅ | ✅ 一样 |
| 训练速度 | 慢 | 快 |
| 稳定性 | 需调参 | 更稳 |
| 代表 | InstructGPT | DeepSeek-R1 |

**关键区别**：**GRPO 去掉了 Critic，用组内相对分数代替 GAE。**

### 和传统 RL 的对应

> **Critic 被“组内平均分”替代**：
>
> $$
> V(s) \approx \text{mean}(r_1, \ldots, r_G)
> $$
>
> 组内平均分就是“这个 prompt 的平均表现”，作为基线。优势就是“比平均好多少”。
>
> **这是 GRPO 的核心思想**：**用蒙特卡洛采样估计基线，不训练 Critic。**

### 优势粒度的区别

**PPO**：

- 优势是 **token 级**：每个 token 位置一个 $A_t$（GAE 算）
- 形状 `(B, R)`

**GRPO**：

- 优势是 **回答级**：每个 response 一个 $A_i$
- 形状 `(B, G)`
- **广播到所有 token**：同一 response 的所有 token 共享这个优势

$$
A_{b,g,t} = A_{b,g}, \quad \forall t
$$

**含义**：这个 response 好，它里面所有 token 都值得鼓励；坏，所有 token 都值得抑制。

### 算法流程
```
1. 对每个 prompt：
a. 用 Actor 采样 G 个回答
b. 用 RM 给每个回答打分 → (B, G)
c. 组内归一化得到优势：A_i = (r_i - mean(r)) / std(r)

2. 算 KL 惩罚（可选）：
对每个回答的每个 token：KL_t = log π_actor - log π_ref

3. 策略更新：

展平 (B, G, P+R) → (B*G, P+R)

重新前向算 new_logp（带梯度）

优势广播到每 token：A (BG,) → (BG, R)

算 ratio = exp(new - old)

clip 损失

反向传播

4. 清空数据，回到第 1 步
```

### 适用场景

| 场景 | 推荐 |
|---|---|
| 可验证任务（数学、代码） | GRPO + 规则奖励 |
| 通用对话对齐 | PPO 或 DPO |
| 显存紧张 | GRPO |
| 追求极致效果 | PPO |
| 单卡 6GB | GRPO |

GRPO 最适合：

- 可验证奖励：规则打分（数学、代码）
- 显存有限：省掉 Critic
- 需要在线采样：比 DPO 探索能力强

DeepSeek-R1 就是用 GRPO + 规则奖励，训练出了强大的推理能力。

### 为什么 GRPO 有效

### 1. 组内归一化是一个好的基线

优势的定义：

$$
A(s,a)=Q(s,a)-V(s)
$$

PPO：用 Critic 估计 $V(s)$。

GRPO：用同组回答的平均分作为基线：

$$
A_i = r_i - \mathrm{mean}(r)
$$

为什么有效：

- 同一 prompt 的多个回答，难度相同
- 平均分是一个天然的基线
- 不需要 Critic 学 $V(s)$

### 2. 去掉了 Critic 的误差

Critic 的问题：

- 需要单独训练
- 估计有误差
- 显存翻倍
- 训练不稳定

GRPO 没有 Critic，直接比较同组回答，更稳定、更省显存。

### 3. 适合可验证任务

GSM8K、代码、数学：答案对错明确，RM 可以很简单（规则），组内归一化非常有效。

### 一句话总结

GRPO = PPO − Critic + 组内归一化。

核心公式：

$$
A_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)}
$$

- 策略损失、KL 惩罚、clip 都和 PPO 完全一样
- 唯一区别是优势的计算方式：去掉 Critic，用同组回答的相对排名代替

GRPO 是显存有限时的最佳选择，尤其适合可验证任务（数学、代码）。

---

# 第四章：Agentic RL

## 4.1 两者的定位关系

Agentic RL 不是 LLM RL 的替代，而是**在 LLM RL 基础上增加了"环境交互"这一维度**。

```
┌──────────────────────────────────────────────────┐
│                RL 后训练（Post-Training）          │
│                                                  │
│  ┌────────────────┐   ┌────────────────────┐     │
│  │     LLM RL     │   │     Agentic RL     │     │
│  │                │   │                    │     │
│  │  - RLHF        │   │  - Search-R1       │     │
│  │  - DPO         │   │  - Tool RL         │     │
│  │  - PPO/GRPO    │   │  - Code RL         │     │
│  │  （单轮）       │   │  - GUI Agent       │     │
│  │                │   │  （多轮 + 工具）     │     │
│  └────────────────┘   └────────────────────┘     │
│              ↑                   ↑               │
│              └─────── 共享 ──────┘               │
│              算法内核（PPO/GRPO）                 │
│              SFT 后起点                          │
└──────────────────────────────────────────────────┘
```

**一句话**：Agentic RL = LLM RL + 环境交互 + 多轮轨迹。

---

## 4.2 LLM RL 流程

### 标准流程

```
Prompt → LLM 生成 → Response → Reward → 更新
        └────────── 单轮 ──────────┘
```

**特点**：

- 一次生成，一次打分
- 序列是 `[prompt | response]` 两段
- 奖励来自 RM 或规则
- 算法：PPO / GRPO / DPO / RLHF

### 关键组件

| 组件 | 职责 |
|---|---|
| Actor | 策略模型（生成 response） |
| Critic | 价值估计（PPO 用，GRPO 无） |
| Reference | KL 约束（可选） |
| Reward Model | 打分 |
| Rollout | 单轮生成 |

### 典型算法

| 算法 | 奖励 | 信用分配 |
|---|---|---|
| PPO | RM + 逐 token KL | Token 级 GAE |
| GRPO | 结果奖励 | 轨迹级（组内归一化） |
| DPO | 偏好对 | — |
| RLHF | RM | Token 级 |

---

## 4.3 Agentic RL 流程

### 标准流程

```
Prompt
   ↓
┌──────────────────────────────────────┐
│             多轮循环                   │
│  LLM 生成 → 解析动作 → 工具执行         │
│    ↑                        ↓        │
│    └────── 拼 observation ←─┘        │
│        直到 done 或 max_turns          │
└──────────────────────────────────────┘
   ↓
完整轨迹 → Reward → 更新
```

### 一条轨迹的结构

```
[prompt | gen1 | info1 | gen2 | info2 | gen3]
   0       1       0       1       0       1
   ↑       ↑       ↑       ↑       ↑       ↑
数据集     LLM     工具     LLM     工具     LLM
          生成     返回     生成     返回     生成

loss_mask: 0=不参与loss, 1=参与loss
```

### 关键组件

| 组件 | 职责 | 相比 LLM RL 的差异 |
|---|---|---|
| Actor | 策略模型 | 生成多段，非一次性 |
| Rollout Engine | 多轮采样 | **新增**：调工具、拼 observation |
| Environment | 环境交互 | **新增** |
| ToolRouter | 解析动作 | **新增** |
| Reward | 打分 | 更复杂（多维） |

### 完整训练循环

```python
for sample in batch:
    # 1. 采样 G 条轨迹
    trajectories = [engine.rollout(sample) for _ in range(G)]

    # 2. 组内归一化算 advantage
    rewards = [t.reward for t in trajectories]
    advantages = (rewards - mean) / std

    # 3. 逐条算 loss
    for traj, adv in zip(trajectories, advantages):
        new_logp = actor(traj.input_ids)
        ratio = exp(new_logp - traj.old_log_probs)
        policy_loss = -min(ratio * adv, clip(ratio) * adv)
        loss = (policy_loss * traj.loss_mask).sum() / traj.loss_mask.sum()

    # 4. 更新
    loss.backward()
    optimizer.step()
```

---

## 4.4 核心差异对比

### 数据流差异

| 维度 | LLM RL | Agentic RL |
|---|---|---|
| 序列结构 | `[prompt \| response]` | `[prompt \| gen \| info \| gen \| ...]` |
| 段数 | 2 段 | 2N+1 段（N=工具调用次数） |
| 长度 | 固定或可预测 | 变长，不可预测 |
| loss_mask | 只有 prompt 段为 0 | prompt + 所有 info 段为 0 |
| 生成次数 | 1 次 | N 次（多轮） |

### 算法差异

| 维度 | LLM RL | Agentic RL |
|---|---|---|
| 算法内核 | PPO / GRPO / DPO | 相同 |
| advantage 粒度 | Token 级（PPO）或轨迹级（GRPO） | 轨迹级为主 |
| 归一化分母 | response_mask.sum() | loss_mask.sum()（排除 info 段） |
| 批量前向 | 可（等长 pad） | 通常不行（变长） |

### 奖励差异

| 维度 | LLM RL | Agentic RL |
|---|---|---|
| 奖励来源 | RM 为主 | 可验证奖励为主 |
| 奖励粒度 | 轨迹级 | 轨迹级 / 步骤级 |
| 奖励设计 | 相对成熟 | 核心难点 |
| reward hacking | 存在 | 更严重（可钻工具空子） |

### 工程差异

| 维度 | LLM RL | Agentic RL |
|---|---|---|
| Rollout | 简单，一次生成 | 复杂，多轮 + 工具调用 |
| 推理引擎 | HF generate 够用 | vLLM / SGLang 更常见 |
| 权重同步 | 不需要（同模型） | vLLM 场景需要 |
| 环境交互 | 无 | 网络调用 / 沙箱 / 浏览器 |
| 异步训练 | 少见 | 常见（rollout 慢） |

---

## 4.5 位置关系

### 训练阶段中的位置

```
预训练（Pretraining）
    ↓
SFT（指令微调）
    ↓
┌───────────────────────────────────┐
│       RL 后训练（并列的多种）        │
│  ├─ RLHF（通用偏好）               │
│  ├─ DPO（偏好对）                  │
│  ├─ LLM RL（单轮推理）             │
│  └─ Agentic RL（多轮工具）         │
└───────────────────────────────────┘
    ↓
部署
```

关键：Agentic RL 和 RLHF 是并列关系——都是 RL，都在 SFT 之后，区别在奖励来源和优化目标。

### 实践顺序

```
1. 基座模型（预训练）
2. Chat Template（tokenizer 配置）
3. 通用 SFT + RLHF（模型发布方做）
   → 发布：Qwen2.5-Instruct
4. 工具调用 SFT（应用方做，可选，冷启动）
5. Agentic RL（应用方做，核心）
```

### 和 FC 能力的关系

| 阶段 | 解决什么 |
|---|---|
| Chat Template | token 层面支持 `<tool_call>` |
| SFT | 格式 + 粗决策 |
| RLHF | 通用偏好 |
| Agentic RL | 特定工具集下的最优策略 |

Agentic RL 是在已具备 FC 能力的模型上，针对具体工具集做进一步优化。

---

## 4.6 共享的算法内核

### GRPO 在两种场景下完全一致

```python
# 公式完全相同
ratio = exp(new_logp - old_logp)
surr1 = ratio * advantage
surr2 = clamp(ratio, 1-eps, 1+eps) * advantage
policy_loss = -min(surr1, surr2)
kl = new_logp - ref_logp
loss = policy_loss + beta * kl
```

### 唯一差异：序列构造

| 项 | LLM RL | Agentic RL |
|---|---|---|
| input_ids | `[prompt \| response]` | `[prompt \| gen \| info \| gen \| ...]` |
| loss_mask | prompt 段 0，response 段 1 | prompt + info 段 0，gen 段 1 |
| old_log_probs | response 段的 logprob | 所有 gen 段的 logprob，info 段填 0 |

### Agentic RL 的三个工程要点

1. **mask 正确**：info 段必须 mask=0，否则模型学会"复述检索结果"
2. **偏移对齐**：logprob、loss_mask 都要偏移一位
3. **变长处理**：无法批量 stack，通常逐条 forward

---

## 4.7 奖励与信用分配

### 奖励的三个层次

```
┌──────────────────────────────────────┐
│ 结果奖励（Outcome Reward）             │
│  最终答案对不对（EM / 测试通过率）      │
│  最稀疏，最难学，但最可靠              │
├──────────────────────────────────────┤
│ 过程奖励（Process Reward）            │
│  中间步骤对不对（推理链、工具调用）      │
│  稠密，好学好，但需要标注或 PRM        │
├──────────────────────────────────────┤
│ 格式奖励（Format Reward）             │
│  有没有用对标签、有没有遵守协议         │
│  最稠密，塑形，但不该主导              │
└──────────────────────────────────────┘
```

权重设计：结果奖励 1.0，过程/格式 0.1~0.3，惩罚 < 0.1。

### 信用分配的四种方案

| 方案 | 粒度 | 适用 |
|---|---|---|
| 轨迹级（GRPO） | 整条轨迹一个标量 | 短轨迹（< 5 轮） |
| Token 级（PPO/GAE） | 每个 token | 有 token 级奖励 |
| 步骤级（GiGPO） | 每次工具调用一步 | 长轨迹（> 10 轮） |
| Turn 级 | 每轮对话 | 多轮 Agent |

### 子 Agent 场景

当 Harness 有子 Agent 时，问题升级为多智能体协作信用分配：

| 方法 | 核心思想 |
|---|---|
| 反事实信用分配 | 对比"移除某 Agent"前后的结果差异 |
| Shapley 值 | 公平分配，但复杂度指数级 |
| 树状信用分配 | 将交互建模为树，逆向传播优势 |
| 分层 RL | Manager 定目标，Worker 执行 |

---

## 4.8 工程架构差异

### LLM RL 架构

```
┌────────────────────────────────┐
│  Trainer                       │
│  - Dataset                     │
│  - Actor                       │
│  - Reward Model                │
│  - Rollout（单轮）              │
│  - 训练循环                    │
└────────────────────────────────┘
```

### Agentic RL 架构

```
┌───────────────────────────────────────────┐
│  Trainer                                  │
│  - 训练循环                                │
│  - advantage 计算                         │
├───────────────────────────────────────────┤
│  Rollout Engine                           │
│  - 多轮循环                                │
│  - loss_mask 维护                         │
│  - old_log_probs 收集                     │
├───────────────────────────────────────────┤
│  Environment                              │
│  - 工具执行                                │
│  - 状态维护                                │
├───────────────────────────────────────────┤
│  ToolRouter                               │
│  - 解析 LLM 输出                           │
│  - 路由到工具                              │
├───────────────────────────────────────────┤
│  RewardContextBuilder + RewardFn          │
│  - 从轨迹构建 reward 所需信息              │
└───────────────────────────────────────────┘
```

### 关键抽象

| 抽象 | 职责 |
|---|---|
| Environment | 工具执行 + 状态维护，纯执行 |
| ToolRouter | 文本 → Action 解析 |
| Tool | 参数解析 + 执行 + observation 格式化 |
| RewardContextBuilder | 从 trace 构建 reward ctx |
| InferenceBackend | 生成 + 权重同步（HF / vLLM） |
| RolloutEngine | 编排多轮循环 |

### 解耦原则

1. **Environment 不知道 reward**：只返回 (obs, done)
2. **Engine 不知道 ctx 字段**：只填 RolloutTrace
3. **Router 不做执行**：只做识别
4. **Tool 不做路由**：只管执行
5. **算法变化不改 Engine**：改 Trainer 的 advantage_fn / loss_fn

---

## 4.9 常见坑

### 数据与 mask

| 坑 | 后果 | 解法 |
|---|---|---|
| info 段参与 loss | 模型学"复述检索结果"，训练崩 | loss_mask 屏蔽 info 段 |
| logprob 偏移没对齐 | ratio 全错 | logp[:-1] 对 input_ids[1:] |
| 混用 attention_mask 和 loss_mask | loss 计算错误 | 严格区分 |
| decode 再 encode 不一致 | old_log_probs 对不上 | 维护 token 序列，不重建字符串 |

### 显存

| 坑 | 解法 |
|---|---|
| 没开梯度检查点 | gradient_checkpointing_enable() |
| LoRA 场景忘加 enable_input_require_grads | 显式加 |
| batch 太大 | 逐 group backward + 梯度累积 |
| 序列太长 | 收紧 max_turns / max_new_tokens |
| 参考模型占用 | kl_beta=0 时不加载 |

### 训练稳定性

| 坑 | 解法 |
|---|---|
| 全对全错，advantage=0 | 加格式奖励、课程学习 |
| Reward hacking | KL 惩罚、对抗样本、人工审核 |
| 多轮梯度爆炸 | 过滤"无效轮次"（SimpleTIR 思路） |
| 冷启动太难 | 先做轻量 SFT，再 RL |

---

## 4.10 实践路线建议

### 从 LLM RL 迁移到 Agentic RL

```
1. 复用算法内核（GRPO / PPO）
2. 改造 Rollout：加入多轮循环
3. 新增 Environment / ToolRouter / Tool
4. 维护 loss_mask（info 段为 0）
5. 奖励设计：先做结果奖励，再加格式/过程
6. 工程优化：梯度检查点、批量 loss、vLLM
```

### 推荐顺序

| 阶段 | 目标 | 关键点 |
|---|---|---|
| 1 | 跑通单工具（Search-R1） | 理解多轮 + mask |
| 2 | 加格式奖励塑形 | reward 权重设计 |
| 3 | 扩展到多工具（Tool RL） | 路由 + 工具集 |
| 4 | 加过程奖励 | 信用分配 |
| 5 | 多 Agent / 子 Agent | 协作信用分配 |
| 6 | 数据飞轮 | 自动化闭环 |

### 参考论文

| 论文 | 贡献 |
|---|---|
| Toolformer | 自监督工具学习（思想奠基） |
| ToolRL | 系统研究奖励设计 |
| ReTool | 冷启动 SFT + 结果导向 RL |
| SimpleTIR | 多轮训练稳定性 |
| Search-R1 | 多轮搜索 + 检索 token masking |
| GiGPO | 步骤级信用分配 |

---

## 4.11 一句话总结

| 维度 | 结论 |
|---|---|
| 关系 | Agentic RL 是 LLM RL 的扩展，不是替代 |
| 共享 | 算法内核（PPO / GRPO）完全一致 |
| 差异 | 序列结构、loss_mask、奖励、工程架构 |
| 核心难点 | 奖励设计 + 信用分配 |
| 工程核心 | 多轮 rollout + mask 维护 + 权重同步 |
| 判断标准 | 是否涉及"环境交互"和"多轮决策" |

一句话：Agentic RL = LLM RL + 环境交互。算法内核共享，工程架构扩展，奖励与信用分配是新的核心挑战。

---

# 第五章：Agentic RL：Tool RL

## 5.1 背景

ToolRL 将工具集成推理（TIR）建模为多步决策过程。模型在每一步生成 `<think>`、`<tool_call>`、`<response>`，环境执行工具调用并返回 `<obs>`。训练使用 GRPO，奖励设计是核心。

奖励设计的目标：既奖励格式正确，也奖励工具调用内容正确，并且随着训练推进，逐渐把重点从格式转向正确性。

信用分配的目标：把最终奖励归因到每一步的工具调用决策上，不依赖 critic，而是用**步骤级密集奖励 + GRPO 组内相对优势**。

---

## 5.2 奖励设计

### 总体结构

最终奖励由两部分组成：

$$
R_{\mathrm{final}} = R_{\mathrm{format}} + R_{\mathrm{correct}}
$$

- $R_{\mathrm{format}}$：格式奖励，检查输出结构是否正确。
- $R_{\mathrm{correct}}$：正确性奖励，细粒度评估工具调用。

经过动态缩放后，两部分会被映射到不同的目标区间，最终相加。

---

### 格式奖励

格式奖励 $R_{\mathrm{format}} \in \{0,1\}$。

检查模型输出是否按正确顺序包含：

1. `<think>`
2. `<tool_call>` 或 `<response>` 至少一个
3. 若两者都有，`<tool_call>` 必须在 `<response>` 之前

用状态机检查，顺序非法得 0，全部通过得 1。

**例子**：

```text
<think>查询北京天气</think>
<tool_call>
{"name": "get_weather", "parameters": {"city": "Beijing", "date": "2026-10-12"}}
</tool_call>
<response>正在查询北京天气</response>
```

顺序正确，$R_{\mathrm{format}} = 1$。

---

### 正确性奖励

正确性奖励 $R_{\mathrm{correct}} \in [-3,3]$，由三部分组成：

$$
r_{\mathrm{match}} = r_{\mathrm{name}} + r_{\mathrm{param}} + r_{\mathrm{value}}
$$

#### 工具名称匹配

$$
r_{\mathrm{name}} = \frac{|N_G \cap N_P|}{|N_G \cup N_P|} \in [0,1]
$$

- $N_G$：真实工具名集合
- $N_P$：预测工具名集合
- 本质是 Jaccard 相似度

**例子**：
- 真实：`{get_weather}`
- 预测：`{get_weather}` → $r_{\mathrm{name}} = 1$
- 预测：`{search}` → $r_{\mathrm{name}} = 0$
- 真实：`{get_weather, get_air_quality}`，预测命中一个 → $r_{\mathrm{name}} = 1/2 = 0.5$

#### 参数名称匹配

$$
r_{\mathrm{param}} = \sum_{G_j \in G} \frac{|\mathrm{keys}(P_G) \cap \mathrm{keys}(P_P)|}{|\mathrm{keys}(P_G) \cup \mathrm{keys}(P_P)|} \in [0,|G|]
$$

对每个真实工具调用，计算参数名集合的 Jaccard，再求和。

**例子**：
- 真实 $G_1$ 参数名：`{city, date}`
- 预测 $P_1$ 参数名：`{city, date}` → 贡献 $2/2 = 1$
- 预测：`{city}` → 贡献 $1/2 = 0.5$
- 预测：`{query}` → 贡献 $0/2 = 0$

#### 参数内容匹配

$$
r_{\mathrm{value}} = \sum_{G_j \in G} \sum_{k \in \mathrm{keys}(G_j)} \mathbb{1}[P_G[k] = P_P[k]]
$$

逐参数比较预测值与真实值，相等得 1，不等得 0。

**例子**：
- 真实：`{city: "Beijing", date: "2026-10-12"}`
- 预测：`{city: "Beijing", date: "2026-10-12"}` → $r_{\mathrm{value}} = 2$
- 预测：`{city: "Shanghai", date: "2026-10-12"}` → $r_{\mathrm{value}} = 1$

比较时忽略类型差异：`5.0` 与 `"5"` 视为相等。推荐用 `Decimal` 统一归一化。

#### 最优匹配与归一化

多工具时，预测与真实之间要找**最优配对**，使 $r_{\mathrm{param}} + r_{\mathrm{value}}$ 之和最大。工具名匹配是集合层面的 Jaccard，不依赖配对。

最大可能分数：

$$
S_{\max} = 1 + |G| + \sum_{G_j \in G} |\mathrm{keys}(G_j)|
$$

归一化：

$$
R_{\mathrm{correct}} = 6 \cdot \frac{r_{\mathrm{match}}}{S_{\max}} - 3 \in [-3,3]
$$

- 完全正确：$r_{\mathrm{match}} = S_{\max}$ → $R_{\mathrm{correct}} = 3$
- 完全错误：$r_{\mathrm{match}} = 0$ → $R_{\mathrm{correct}} = -3$

---

### 动态奖励尺度

设 $p = S_{\mathrm{current}} / S_{\mathrm{total}} \in [0,1]$，其中 $S_{\mathrm{current}}$ 是当前优化步数，$S_{\mathrm{total}}$ 是总优化步数。

格式奖励目标区间：

$$
[L_f, U_f] = [-2 + p,\ 2 - p]
$$

映射：

$$
R_{\mathrm{format}}' = L_f + R_{\mathrm{format}} \cdot (U_f - L_f)
$$

正确性奖励目标区间：

$$
[L_c, U_c] = [-2 - p,\ 2 + p]
$$

映射：

$$
R_{\mathrm{correct}}' = L_c + \frac{R_{\mathrm{correct}} + 3}{6} \cdot (U_c - L_c)
$$

**平滑过渡效果**：

| $p$ | 格式区间 | 正确性区间 | 格式正确奖励 | 完全正确奖励 |
|---|---|---|---|---|
| 0.0 | $[-2, 2]$ | $[-2, 2]$ | 2 | 2 |
| 0.5 | $[-1.5, 1.5]$ | $[-2.5, 2.5]$ | 1.5 | 2.5 |
| 1.0 | $[-1, 1]$ | $[-3, 3]$ | 1 | 3 |

格式奖励的相对价值逐渐下降，正确性奖励的相对价值逐渐上升，没有跳变点。

---

## 5.3 信用分配

### 步骤级奖励

ToolRL 的多步轨迹被拆成**单步实例**。每个训练样本对应轨迹中的一步，`user_prompt` 包含之前的历史上下文，`outputs` 是该步的真实工具调用。

因此，每一步的奖励直接由该步的预测与 ground truth 计算得到，不需要等待整条轨迹结束，也不需要 critic 做跨时间反向传播。

Ground truth 来自 ToolACE、Hammer、xLAM 数据集，不是人工额外标注。

---

### GRPO 组内相对优势

对每个查询 $Q$，模型生成一组响应：

$$
G_Q = \{A, (s_1, r_1), (s_2, r_2), \ldots, (s_n, r_n)\}
$$

其中 $r_i = R_{\mathrm{format}}'(s_i) + R_{\mathrm{correct}}'(s_i)$。

计算组内均值和标准差：

$$
\mu_Q = \frac{1}{n} \sum_{i=1}^n r_i,\quad
\sigma_Q = \sqrt{\frac{1}{n} \sum_{i=1}^n (r_i - \mu_Q)^2}
$$

归一化优势：

$$
A_i(s_i|Q) = \frac{r_i - \mu_Q}{\sigma_Q + \eta}
$$

高于组内平均的响应获得正优势，低于平均的获得负优势。

---

### 策略更新

GRPO 目标函数：

$$
J_{\mathrm{GRPO}}(\theta) = \mathbb{E}_{Q\sim\mathcal{D}} \mathbb{E}_{s_i\sim\pi_\theta}
\left[
\min\left(
\frac{\pi_\theta(s_i|Q)}{\pi_{\mathrm{old}}(s_i|Q)} A_i(s_i|Q),\ 
\mathrm{clip}\left(\frac{\pi_\theta(s_i|Q)}{\pi_{\mathrm{old}}(s_i|Q)}, 1-\epsilon, 1+\epsilon\right) A_i(s_i|Q)
\right)
\right]
$$

- 正优势 → 提高该响应概率
- 负优势 → 降低该响应概率
- clip 限制更新幅度
- 不使用 KL 惩罚项

---

## 5.4 完整例子

### 场景设定

查询 $Q$：查询北京 2026-10-12 的天气。

真实工具调用 $G$：

```json
[
  {
    "name": "get_weather",
    "parameters": {"city": "Beijing", "date": "2026-10-12"}
  }
]
```

$|G| = 1$，参数总数 = 2，$S_{\max} = 1 + 1 + 2 = 4$。

模型对同一查询生成 4 个响应。

---

### 四个响应

**响应 1：完全正确**

```text
<think>用户想知道北京天气</think>
<tool_call>
{"name": "get_weather", "parameters": {"city": "Beijing", "date": "2026-10-12"}}
</tool_call>
<response>正在查询北京天气</response>
```

**响应 2：工具名对，城市错**

```text
<think>用户想知道天气</think>
<tool_call>
{"name": "get_weather", "parameters": {"city": "Shanghai", "date": "2026-10-12"}}
</tool_call>
<response>正在查询上海天气</response>
```

**响应 3：工具名错**

```text
<think>搜索北京天气</think>
<tool_call>
{"name": "search", "parameters": {"query": "Beijing weather"}}
</tool_call>
<response>正在搜索</response>
```

**响应 4：格式错误**

```text
<tool_call>
{"name": "calculator", "parameters": {"expression": "1+1"}}
</tool_call>
```

---

### 奖励计算

假设当前训练进度 $p = 0.5$，动态缩放后。

| 响应 | $R_{\mathrm{format}}$ | $r_{\mathrm{name}}$ | $r_{\mathrm{param}}$ | $r_{\mathrm{value}}$ | $r_{\mathrm{match}}$ | $R_{\mathrm{correct}}$ | 缩放后格式 | 缩放后正确性 | $R_{\mathrm{final}}$ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 2 | 4 | $6\cdot4/4-3=3$ | 1.5 | 2.5 | **4.0** |
| 2 | 1 | 1 | 1 | 1 | 3 | $6\cdot3/4-3=1.5$ | 1.5 | 1.25 | **2.75** |
| 3 | 1 | 0 | 0 | 0 | 0 | $-3$ | 1.5 | -2.5 | **-1.0** |
| 4 | 0 | 0 | 0 | 0 | 0 | $-3$ | -1.5 | -2.5 | **-4.0** |

计算细节：

- 响应 1：$r_{\mathrm{name}}=1/1=1$，$r_{\mathrm{param}}=2/2=1$，$r_{\mathrm{value}}=1+1=2$，$r_{\mathrm{match}}=4$，$R_{\mathrm{correct}}=3$。
- 响应 2：$r_{\mathrm{value}}=0+1=1$，$r_{\mathrm{match}}=3$，$R_{\mathrm{correct}}=1.5$。
- 响应 3：工具名不匹配，参数名不匹配，$r_{\mathrm{match}}=0$，$R_{\mathrm{correct}}=-3$。
- 响应 4：格式错误，$R_{\mathrm{format}}=0$，工具名和参数都不匹配，$R_{\mathrm{correct}}=-3$。

缩放：$p=0.5$ 时，格式区间 $[-1.5, 1.5]$，正确性区间 $[-2.5, 2.5]$。
- 格式缩放：$R_{\mathrm{format}}' = -1.5 + R_{\mathrm{format}} \cdot 3$。$R_{\mathrm{format}}=1$ → $1.5$；$R_{\mathrm{format}}=0$ → $-1.5$。
- 正确性缩放：$R_{\mathrm{correct}}' = -2.5 + (R_{\mathrm{correct}}+3) \cdot 5 / 6$。
  - $R_{\mathrm{correct}}=3$ → $-2.5 + 6 \cdot 5/6 = 2.5$
  - $R_{\mathrm{correct}}=1.5$ → $-2.5 + 4.5 \cdot 5/6 = 1.25$
  - $R_{\mathrm{correct}}=-3$ → $-2.5 + 0 = -2.5$

---

### 优势计算与更新

四个响应的最终奖励：

$$
r = [4.0,\ 2.75,\ -1.0,\ -4.0]
$$

组内均值：

$$
\mu_Q = \frac{4.0 + 2.75 - 1.0 - 4.0}{4} = 0.4375
$$

标准差近似：

$$
\sigma_Q \approx 3.09
$$

归一化优势：

| 响应 | $r_i$ | $A_i$ |
|---|---|---|
| 1 | 4.0 | $(4.0-0.4375)/3.09 \approx 1.15$ |
| 2 | 2.75 | $(2.75-0.4375)/3.09 \approx 0.75$ |
| 3 | -1.0 | $(-1.0-0.4375)/3.09 \approx -0.47$ |
| 4 | -4.0 | $(-4.0-0.4375)/3.09 \approx -1.44$ |

含义：

- 响应 1 和 2 高于组内平均，获得正优势，被鼓励。
- 响应 3 和 4 低于平均，获得负优势，被抑制。
- 响应 1 的优势最大，提升幅度最大。

策略更新时，GRPO 按照优势加权，提高响应 1 和 2 的生成概率，降低响应 3 和 4 的生成概率。

---

## 5.5 实现要点

1. **格式奖励用状态机**：按 `<think>` → `<tool_call>` → `<response>` 顺序检查，非法顺序直接 0。
2. **正确性奖励共享最优配对**：参数名和参数值用同一个配对，最大化 $r_{\mathrm{param}} + r_{\mathrm{value}}$。
3. **参数值比较忽略类型**：用 `Decimal` 统一数字和数字字符串，`5.0` 与 `"5"` 相等。
4. **动态缩放用训练步数**：$p = S_{\mathrm{current}} / S_{\mathrm{total}}$，$S_{\mathrm{current}}$ 是优化步，不是 epoch。
5. **信用分配不依赖 critic**：步骤级奖励直接归因，GRPO 组内相对优势完成信用分配。
6. **Ground truth 来自数据集**：ToolACE、Hammer、xLAM 已包含每步真实工具调用。
7. **多工具并行调用**：`<tool_call>` 内可能包含多个 JSON 对象或数组，解析时要兼容。
8. **不需要 SFT 初始化**：GRPO 冷启动通常优于 SFT 初始化，SFT 容易过拟合。

---

## 5.6 总结

ToolRL 的奖励设计是：

$$
R_{\mathrm{final}} = R_{\mathrm{format}}' + R_{\mathrm{correct}}'
$$

其中 $R_{\mathrm{correct}}$ 细粒度分解为工具名、参数名、参数值，用最优匹配归一化到 $[-3,3]$，再通过动态尺度让格式奖励逐渐退场、正确性奖励逐渐主导。

信用分配依靠：

1. **步骤级奖励**：多步轨迹拆成单步实例，每步用数据集提供的真实工具调用作为 ground truth。
2. **GRPO 组内归一化优势**：同一查询生成多个响应，按组内相对好坏计算优势。
3. **策略梯度更新**：用优势加权，提高好响应的概率，降低差响应的概率。

整个流程不需要 critic，也不需要人工逐步标注，依赖的是**密集步骤奖励 + 组内相对比较**。