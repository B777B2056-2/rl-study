# 强化学习与 LLM 微调学习笔记

> **格式约定**：`##/###` 标题为主干内容，`>` 引用块为补充说明、小知识点、类比、注意事项。

---

# 目录

- [第一章：强化学习基本概念](#第一章强化学习基本概念)
- [第二章：传统强化学习](#第二章传统强化学习)
- [第三章：LLM 微调](#第三章llm-微调)

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

**目标**：找到策略 $\pi^*$，使期望累计折扣奖励最大：

$$
\pi^* = \arg\max_\pi \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1}\right]
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
V^*(s) = \max_a Q^*(s,a)
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
Q^*(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s',a') \mid s,a\right]
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
Q^*(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s',a')\right]
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
L^{\text{SFT}} = -\frac{1}{N}\sum_{t=1}^{N} \log \pi_\theta(y_t \mid y_{<t}, x)
$$

> $x$：prompt。
>
> $y_t$：response 的第 $t$ 个 token。
>
> $y_{<t}$：response 的前 $t-1$ 个 token。
>
> $N$：response 的 token 数。

### Teacher Forcing（小知识点）

> **什么是 Teacher Forcing**：训练时，把标准答案作为输入，让模型预测下一个 token。"Teacher"是标准答案，"Forcing"是强制把标准答案喂给模型，而不是让模型用自己生成的 token 作为下一步的输入。
>
> **为什么需要**：语言模型是自回归的，第 $t$ 个 token 的预测依赖前 $t-1$ 个 token。
>
> 如果不用 Teacher Forcing：
>
> ```
> 第 1 步：输入 [BOS]，模型预测 → "我"
> 第 2 步：输入 [BOS, 我]，模型预测 → "爱"      ← 用模型自己上一步的输出
> 第 3 步：输入 [BOS, 我, 爱]，模型预测 → "学习"
> ```
>
> **问题**：
>
> - 第 1 步错了，第 2 步输入也错，错误累积
> - 要预测 N 个 token，就要前向 N 次，串行，慢
>
> **具体流程**：
>
> **第 1 步：构造输入**
>
> ```
> prompt_ids   = [1, +, 1, =, ?]            # 5 个 token
> response_ids = [2, EOS]                   # 2 个 token
>
> input_ids = [1, +, 1, =, ?, 2, EOS]       # 7 个 token
> ```
>
> **第 2 步：构造标签**
>
> ```
> labels = [-100, -100, -100, -100, -100, 2, EOS]
>          ←────── prompt 部分忽略 ──────→ ←─ response ─→
> ```
>
> **第 3 步：一次前向，得到每个位置的预测**
>
> ```
> 输入: [1, +, 1, =, ?, 2, EOS]
> 位置:  0  1  2  3  4  5  6
>
> 位置 4 的 logits: 预测下一个 token（应该是 "2"）      ← response 第 1 个
> 位置 5 的 logits: 预测下一个 token（应该是 "EOS"）    ← response 第 2 个
> ```
>
> **第 4 步：Shift 对齐**
>
> ```
> shift_logits = 位置 0 1 2 3 4 5 的 logits
> shift_labels = 位置 1 2 3 4 5 6 的标签 = [+, 1, =, ?, 2, EOS]
> ```
>
> **第 5 步：交叉熵**
>
> ```
> loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
> ```
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
> 位置 4（"?"）能看到：
> 位置 0: 1
> 位置 1: +
> 位置 2: 1
> 位置 3: =
> 位置 4: ?
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
> **类比**：
>
> - **用 Teacher Forcing**：老师手把手纠正，每一步都从正确的起点开始，孩子只需要学"给定正确前缀，下一个字写什么"。
> - **不用 Teacher Forcing**：孩子写错一个字，老师不管，继续往下写，错误越滚越大。

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
r_t = \begin{cases} -\beta \cdot \text{KL}_t & t < R \\ -\beta \cdot \text{KL}_t + r^{\text{RM}} & t = R \end{cases}
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

## 3.5 三者关系

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

PPO（强化学习）
    ↓ 让模型"回答更好"
    ↓ 数据：prompt（自己采样）
    ↓ 损失：clip + value
    ↓ 模型：4 个
```

### 对比总表

| 维度 | SFT | PPO |
|---|---|---|
| 数据 | (prompt, response) | prompt |
| 学习方式 | 监督 | 强化 |
| 损失 | 交叉熵 | clip + value |
| 需要 RM | ❌ | ✅ |
| 需要 Ref | ❌ | ✅ |
| 需要采样 | ❌ | ✅ |
| 模型数 | 1 | 4 |

### 学习目标对照

| 阶段 | 学习目标 |
|---|---|
| SFT | 最大化 response 每个 token 的对数概率 |
| LoRA | 用低秩增量逼近全参微调的效果 |
| QLoRA | 同 LoRA，但省显存 |
| PPO | 最大化 RM 分数，同时不偏离 SFT 太远 |

### 从 SFT 到 PPO 的完整链路

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

### LoRA / QLoRA 在其中的位置

> **LoRA 是一种"参数高效微调方法"，可以用于 SFT、PPO 的任何阶段**：
>
> - SFT 阶段：用 LoRA 微调，省显存
> - PPO 阶段：Actor 用新 LoRA，Critic 用 value_head
>
> **QLoRA 是 LoRA 的显存优化版**：
>
> - 只在"base 太大放不下"时使用（7B+）
> - 小模型（0.5B）用普通 LoRA 即可

### 选择建议

| 场景 | 推荐 |
|---|---|
| 入门，显存紧张 | SFT（LoRA） |
| 追求效果，资源充足 | SFT → RM → PPO |
| 可验证任务（数学、代码） | SFT → PPO + 规则奖励 |
| 单卡 6GB | SFT（LoRA）+ PPO（共享 backbone） |

### 核心公式对照

| 阶段 | 核心公式 |
|---|---|
| SFT | $L = -\sum_t \log \pi_\theta(y_t \mid y_{<t})$ |
| LoRA | $y = Wx + \frac{\alpha}{r}BAx$ |
| QLoRA | 同 LoRA，但 base 是 4-bit |
| PPO | $L = L^{\text{policy}} + 0.5 L^{\text{value}}$ |
| PPO 的 KL | $r_t = -\beta(\log\pi_\theta - \log\pi_{\text{ref}})$ |
| PPO 的 GAE | $A_t = \delta_t + \gamma\lambda A_{t+1}$ |

---
