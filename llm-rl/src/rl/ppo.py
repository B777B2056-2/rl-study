from dataclasses import dataclass
import torch
from copy import deepcopy
from src.plotter import Plotter
from src.models import LoraConfig, inject_lora, disabled_lora
from src.data import RLDataset
from typing import Callable
from tqdm import tqdm
from src.ckpt import *

@dataclass
class PPOConfig:
    dataset: RLDataset
    ckpt_config: CkptConfig
    max_new_tokens: int
    temperature: float
    top_p: float
    n_episode: int
    learning_rate: float
    enable_actor_critic_weight_share: bool  # 是否开启actor和critic的主干权重共享
    actor_lora_config: LoraConfig
    kl_beta: float  # KL惩罚系数
    reward_fn: Callable # 奖励函数，可以是奖励模型的输出、规则奖励结果，不关心具体实现
    n_epoch: int # 策略模型、价值模型训练轮次
    gamma: float
    device: str
    lambda_val: float = 0.95
    ratio_clip_eps: float = 0.2 # 新旧logp裁剪
    grad_clip_eps: float = 1.0  # 梯度裁剪

class Actor(torch.nn.Module):
    """
        策略模型：输出动作概率，选动作
        输入：当前状态，即 当前已生成的Tokens
        输出：各状态对应选取的动作概率，即 词表中各Token为下一Token的概率
        做法：SFT 模型（冻结） + PPO LoRA（训练）；若SFT本身为Lora来的，则需把SFT-Lora权重合并到原模型，再进行PPO
    """
    def __init__(self, base_model, lora_config: LoraConfig):
        super(Actor, self).__init__()
        # 冻结SFT模型
        for p in base_model.parameters():
            p.requires_grad = False

        # 注入 LoRA
        inject_lora(base_model, lora_config)
        self._actor = base_model

    def forward(self, input_ids, attention_mask):
        # 获取actor模型前向传播的输出
        logits = self._actor(input_ids, attention_mask=attention_mask).logits
        return logits

    def generate(self, *args, **kwargs):
        # 生成一次完整回答
        return self._actor.generate(*args, **kwargs)

    def model(self):
        return self._base

    def lora_config(self):
        return self._lora_config

class Critic(torch.nn.Module):
    """
        价值模型：估计状态价值，算优势
        输入：当前状态，即 当前已生成的Tokens
        输出：各状态对应的状态价值，即 词表中各Token的状态价值估计值
    """
    def __init__(self, base_model):
        super(Critic, self).__init__()

        self._base_model = base_model

        # 冻结LLM主干权重
        for p in self._base_model.parameters():
            p.requires_grad = False

        # 使用value head，用于下一个Token的状态价值估计
        hidden_size = self._base_model.config.hidden_size
        self._value_head = torch.nn.Linear(
            in_features=hidden_size, 
            out_features=1,
            dtype = next(self._base_model.parameters()).dtype
        )   # 输出维度：(batch_size, seq_len, 1)，下一个Token的状态价值估计
    
    def forward(self, input_ids, attention_mask=None):
        with disabled_lora(self._base_model): # 若共享权重，则在Critic前向时冻结Actor的Lora层权重，防止Critic更新了Actor的Lora
            # 主干前向，跳过 lm_head（其实没有真正跳过，还是会计算，只是不用这个计算结果；想要真正跳过，需要使用XXXModel类，如Qwen2Model类）
            outputs = self._base_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]            # (batch_size, seq_len, hidden_size)

        # 计算状态价值估计
        values = self._value_head(hidden)             # (batch_size, seq_len, 1)
        return values.squeeze(-1)                     # (batch_size, seq_len)

class ActorCriticWeightSharedAdapter(object):
    """Actor与Critic权重共享适配器"""
    def __init__(self, base_model, actor_lora_config: LoraConfig):
        self._shared_model = deepcopy(base_model)

        # 冻结主干权重
        for p in self._shared_model.parameters():
            p.requires_grad = False

        self._actor = Actor(base_model=self._shared_model, lora_config=actor_lora_config)
        self._critic = Critic(base_model=self._shared_model)

    def actor(self):
        return self._actor

    def critic(self):
        return self._critic

class Reference(torch.nn.Module):
    """
        参考模型：冻结权重，不更新，主要目的为计算KL惩罚，防止PPO微调结果与SFT的结果有较大差距
    """
    def __init__(self, base_model):
        super().__init__()
        self._ref_model = base_model

        # 冻结
        for p in self._ref_model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self._ref_model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )
        return outputs.logits


class PPOTrainer(CheckpointableTrainer):
    """
        Actor（策略网络）：输入状态，输出动作概率，负责选动作。
        Critic（价值网络）：输入状态，输出状态价值 V(s)，负责评估状态好坏。

        状态 s
            ├→ Actor → 动作分布 π(a|s) → 采样动作 a
            └→ Critic → V(s)
                            ↓
                    算优势 A(s,a)
                            ↓
                Actor 和 Critic 更新

        状态空间：LLM为当前已生成的Token
        动作空间：在LLM里，动作为生成下一个token，因此为词表大小
    """
    def __init__(self, base_model, tokenizer, config: PPOConfig):
        self._plotter = Plotter()
        ckpt_manager = CheckpointManager(config=config.ckpt_config)
        super().__init__(ckpt_manager, self._plotter)

        self._config = config
        self._tokenizer = tokenizer

        if config.enable_actor_critic_weight_share:
            adapter = ActorCriticWeightSharedAdapter(base_model=base_model, actor_lora_config=config.actor_lora_config)
            self._actor = adapter.actor().to(config.device)
            self._critic = adapter.critic().to(config.device)
        else:
            self._actor = Actor(
                base_model=deepcopy(base_model),
                lora_config=config.actor_lora_config,
            ).to(config.device)

            self._critic = Critic(
                base_model=deepcopy(base_model),
            ).to(config.device)

        self._ref_model = Reference(base_model=base_model) # 参考模型

        self._trainable_parameters = [p for p in self._actor.parameters() if p.requires_grad] + [p for p in self._critic.parameters() if p.requires_grad]
        self._optimizer = torch.optim.Adam(
            self._trainable_parameters, 
            lr=config.learning_rate,
        )

    def _actor_generate_full_replay(self, input_ids, attention_mask):
        """actor生成一次完整回答"""
        output_ids = self._actor.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self._config.max_new_tokens,
            do_sample=True,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )   # (batch_size, len(prompt + response))

        output_attention_mask = torch.ones_like(output_ids)
        for i in range(output_ids.size(0)):
            eos_pos = (output_ids[i] == self._tokenizer.eos_token_id).nonzero()
            if len(eos_pos) > 0:
                first_eos = eos_pos[0].item()
                output_attention_mask[i, first_eos+1:] = 0
        return output_ids, output_attention_mask

    @classmethod
    def _calc_log_probs(cls, prompt_len, output_ids, model, output_attention_mask):
        """针对Actor或Reference，计算对数概率、助手回复的token ids"""
        logits = model(output_ids, output_attention_mask)      # (batch_size, len(prompt + response), vocab_size)
        response_logits = logits[:, prompt_len-1:-1, :]        # (batch_size, len(response), vocab_size)
        log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
        response_ids = output_ids[:, prompt_len:]                       # (batch_size, len(response), vocab_size)
        log_probs = log_probs.gather(-1, response_ids.unsqueeze(-1))    # (batch_size, len(response), vocab_size)
        log_probs = log_probs.squeeze(-1)                               # (batch_size, len(response))
        return log_probs, response_ids

    @torch.no_grad()
    def _collect_traces(self, input_ids, attention_mask, ground_truth):
        """
            单batch采样数据轨迹
            方法：用当前策略和环境交互，收集的一批 (s, a, r, s', done, log_prob, V) 数据；不进行梯度更新
        """
        
        # actor完整生成一次回答
        output_ids, output_attention_mask = self._actor_generate_full_replay(input_ids=input_ids, attention_mask=attention_mask)
        
        # 输入策略模型，获取决策出来的动作（即 下一Token预测概率）
        prompt_len = input_ids.size(1)
        actor_log_probs, actor_response_ids = PPOTrainer._calc_log_probs(prompt_len, output_ids, self._actor, output_attention_mask)
        
        # 输入价值模型，计算该状态对应的价值（即下一Token对应的状态价值）
        values = self._critic(output_ids, output_attention_mask)     # (batch_size, len(prompt + response))
        values = values[:, prompt_len:]                              # (batch_size, len(response))

        # 算奖励分数
        responses = self._tokenizer.batch_decode(actor_response_ids, skip_special_tokens=True)
        rm_scores = torch.tensor(
            [self._config.reward_fn(r, gt) for r, gt in zip(responses, ground_truth)],
            dtype=torch.float32,
            device=self._config.device,
        )

        # 输入参考模型，计算KL惩罚
        ref_log_probs, _ = PPOTrainer._calc_log_probs(prompt_len, output_ids, self._ref_model, output_attention_mask)
        kl = actor_log_probs - ref_log_probs        # (batch_size, len(response))
        rewards = -self._config.kl_beta * kl
        rewards[:, -1] += rm_scores                 # 末尾加 RM

        return {
            "input_ids":      output_ids,
            "attention_mask": output_attention_mask,
            "prompt_len":     prompt_len,
            "actions":        actor_response_ids,
            "log_probs":      actor_log_probs,
            "values":         values,
            "rewards":        rewards,
        }

    def _compute_gae(self, traceInfoDict):
        """
            广义优势估计：用多步 TD 误差（时序差分误差）的加权和来估计优势，平衡偏差和方差。
            优势 > 0：这个动作比平均好 → 提高它的概率
            优势 < 0：这个动作比平均差 → 降低它的概率
        """
        # 1. 获取奖励、价值
        rewards = traceInfoDict["rewards"]  # (batch_size, len(response))
        values = traceInfoDict["values"]    # (batch_size, len(response))

        response_len = rewards.shape[1]

        # 2. 反向遍历：因t时刻的GAE值依赖t+1时刻的GAE值，因此只能反向遍历
        advantages = torch.zeros_like(rewards)
        for t in reversed(range(response_len)):
            # 最后一个token时，预测结束，价值为0，优势估计也为0（此时没有未来了）
            if t == response_len - 1:
                next_values = 0
                next_advantages = 0
            else:
                next_values = values[:, t+1]
                next_advantages = advantages[:, t+1]

            # 1. 计算当前t时刻TD误差（时序差分误差）
            td_error = rewards[:, t] + self._config.gamma * next_values - values[:, t]
            # 2. 计算计算当前t时刻的优势估计值
            advantages[:, t] = td_error + self._config.gamma * self._config.lambda_val * next_advantages

        # 3. 计算回报：t时刻的回报 = t时刻的优势 + t时刻的价值
        returns = advantages + values
        return advantages, returns

    def _clip(self, logp_ratio):
        """裁剪：限制概率比的范围，每次策略网络训练只走一小步，保持训练稳定"""
        return torch.clamp(logp_ratio, 1 - self._config.ratio_clip_eps, 1 + self._config.ratio_clip_eps)

    def _update_batch(self, traceInfoDict, advantages: torch.Tensor, returns: torch.Tensor):
        """
            迭代训练一个batch
        """
        prompt_len = traceInfoDict["prompt_len"]
        input_ids = traceInfoDict["input_ids"]                 # (batch_size, len(prompt)+len(response))
        attention_mask = traceInfoDict["attention_mask"]       # (batch_size, len(prompt)+len(response))
        old_actor_log_probs = traceInfoDict["log_probs"]       # (batch_size, len(response))

        # 2. batch迭代
        total_loss = 0.0        # 总损失
        total_policy_loss = 0.0 # 总策略损失
        total_value_loss = 0.0  # 总价值损失
        for _ in range(self._config.n_epoch):
            # 策略网络前向传播，获取新动作、新动作对应的新概率分布、熵
            new_actor_log_probs, _ = PPOTrainer._calc_log_probs(prompt_len, input_ids, self._actor, attention_mask)
            # 计算概率比：新概率 / 旧概率，使用exp保证数值稳定性，避免极小概率相除导致溢出
            ratio = torch.exp(new_actor_log_probs - old_actor_log_probs)
            # 策略损失：让"好动作"的概率变大，让"坏动作"的概率变小
            policy_loss = -torch.min(ratio * advantages,  self._clip(ratio) * advantages).mean()
            total_policy_loss += policy_loss.item()
            # 价值损失：价值网络估计的 V(s) 和实际回报 R 的均方误差。
            values = self._critic(input_ids, attention_mask)     # (batch_size, len(prompt + response))
            values = values[:, prompt_len:]                      # (batch_size, len(response))
            value_loss = torch.nn.functional.mse_loss(values, returns)
            total_value_loss += value_loss.item()
            # 总损失 = 策略损失 + 价值损失，LLM不算熵（熵太多，算不过来）
            loss = policy_loss + 0.5 * value_loss
            total_loss += loss.item()

            # 梯度更新
            self._optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self._trainable_parameters, self._config.grad_clip_eps)
            self._optimizer.step()

        return total_loss / self._config.n_epoch, total_policy_loss / self._config.n_epoch, total_value_loss / self._config.n_epoch

    def train(self) -> dict:
        """ppo训练"""
        # 从 checkpoint 恢复
        self.maybe_resume()

        train_data_loader = self._config.dataset.build_train_data_loader()
        for ep in range(self._epoch, self._config.n_episode):
            # 0. 遍历训练集
            step_cnt, loss, policy_loss, value_loss = 0, 0.0, 0.0, 0.0
            for batch in tqdm(train_data_loader, desc="Training"):
                input_ids = batch["input_ids"].to(self._config.device)
                attention_mask = batch["attention_mask"].to(self._config.device)
                ground_truth = batch["ground_truth"]

                # 1. 采样轨迹
                traceInfoDict = self._collect_traces(input_ids, attention_mask, ground_truth)

                # 2. 计算优势（GAE）
                advantages, returns = self._compute_gae(traceInfoDict)

                # 3. 优势归一化：减少梯度方差
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # 4. 多轮训练
                batch_loss, batch_policy_loss, batch_value_loss = self._update_batch(traceInfoDict, advantages, returns)
                loss, policy_loss, value_loss = loss+batch_loss, policy_loss+batch_policy_loss, value_loss+batch_value_loss
                step_cnt += 1

                # 5. 定期保存 checkpoint
                self.maybe_save()

            loss, policy_loss, value_loss = loss/step_cnt, policy_loss/step_cnt, value_loss/step_cnt
            self._plotter.log(name="train_loss", value=loss, step=ep)
            self._plotter.log(name="train_policy_loss", value=policy_loss, step=ep)
            self._plotter.log(name="train_value_loss", value=value_loss, step=ep)
        self._plotter.plot(names=['train_loss', 'train_policy_loss', 'train_value_loss'])

    @torch.no_grad()
    def eval(self) -> None:
        """
        测试集上评估
        返回:
            {
                "mean_reward": 平均奖励,
                "accuracy":    成功率,
                "mean_length": 平均回答长度,
                "num_samples": 样本数,
            }
        """
        self._actor.eval()

        test_data_loader = self._config.dataset.build_test_data_loader()

        rewards = []
        lengths = []

        for batch in tqdm(test_data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self._config.device)
            attention_mask = batch["attention_mask"].to(self._config.device)
            ground_truth = batch["ground_truth"]

            # 1. 生成回答
            output_ids, _ = self._actor_generate_full_replay(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # 2. 解码
            prompt_len = input_ids.size(1)
            response_ids = output_ids[:, prompt_len:]
            responses = self._tokenizer.batch_decode(
                response_ids, skip_special_tokens=True,
            )

            # 3. 逐样本算奖励
            for response, gt in zip(responses, ground_truth):
                rewards.append(self._config.reward_fn(response, gt))
                lengths.append(response_ids.size(1))    # 近似：整 batch 长度

        self._actor.train()

        # 记录到 Plotter（每个样本一个点）
        for i, (r, l) in enumerate(zip(rewards, lengths)):
            self._plotter.log("eval_reward", r, step=i)
            self._plotter.log("eval_length", l, step=i)

        # 画评估曲线
        self._plotter.plot(names=["eval_reward", "eval_length"])

    def actor(self):
        """返回训练好的 Actor"""
        return self._actor

    def _state_dict(self) -> dict:
        return {
            "actor_lora": {
                n: p.data.cpu()
                for n, p in self._actor.named_parameters()
                if p.requires_grad
            },
            "critic_value_head": self._critic._value_head.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "rng": get_rng_state(),
        }

    def _load_state_dict(self, state: dict):
        # Actor LoRA
        current_actor = dict(self._actor.named_parameters())
        for n, p in state["actor_lora"].items():
            current_actor[n].data.copy_(p.to(current_actor[n].device))

        # Critic value_head
        self._critic._value_head.load_state_dict(state["critic_value_head"])

        # Optimizer
        self._optimizer.load_state_dict(state["optimizer"])

        # RNG
        if "rng" in state:
            set_rng_state(state["rng"])


if __name__ == "__main__":
    from src.models import Qwen2_5, LoraConfig
    from src.data import GSM8kDatasetAdapter
    import re

    # ============================================================
    # 1. 奖励函数（GSM8K 规则奖励）
    # ============================================================
    def extract_answer(text: str):
        """从回答里提取 #### 后面的数字"""
        if "####" in text:
            return text.split("####")[-1].strip()
        nums = re.findall(r"-?\d+\.?\d*", text)
        return nums[-1] if nums else None

    def gsm8k_reward_fn(response: str, ground_truth: str) -> float:
        """
        规则奖励：
            答案对 +1
            答案错 / 没答案 -1
        """
        pred = extract_answer(response)
        if pred is None:
            return -1.0
        return 1.0 if pred == ground_truth else -1.0

    # ============================================================
    # 2. PPO训练
    # ============================================================
    qwen2_5 = Qwen2_5.from_pretrained(save_dir='./outputs/gsm8k-sft-qwen2_5-0_5B')
    tokenizer = qwen2_5.tokenizer()
    base_model = qwen2_5.model()

    gsm8k_adapter = GSM8kDatasetAdapter()
    gsm8k_rl_dataset = RLDataset(tokenizer=tokenizer, adapter=gsm8k_adapter, batch_size=1)

    ppo_trainer = PPOTrainer(
        base_model=base_model,
        tokenizer=tokenizer,
        config=PPOConfig(
            dataset=gsm8k_rl_dataset,
            ckpt_config=CkptConfig(
                ckpt_dir="outputs/gsm8k-ppo/ckpt",
                save_steps = 1,
                keep_last = 3,
            ),
            max_new_tokens=256,             # 生成回答的最大长度
            temperature=1.0,                # 采样温度
            top_p=1.0,                      # 核采样，1.0 表示不限制
            n_episode=3,                    # 外层训练轮数
            learning_rate=1e-5,             # PPO 学习率，比 SFT 小
            enable_actor_critic_weight_share=True,  # 6GB 显存共享主干
            actor_lora_config=LoraConfig(   # PPO 的新 LoRA
                alpha=16,
                rank=8,
                dropout=0.05,
                use_qlora=False,            # 这里不用 QLoRA，base 已经是 SFT 模型
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
            kl_beta=0.01,                   # KL 惩罚系数
            reward_fn=gsm8k_reward_fn,      # 规则奖励
            n_epoch=1,                      # PPO 内循环，LLM 通常 1 轮
            gamma=1.0,                      # LLM 的回报不需要折扣
            device="cuda",
            lambda_val=0.95,
            ratio_clip_eps=0.2,
            grad_clip_eps=1.0,
        ),
    )

    ppo_trainer.train()
    ppo_trainer.eval()

    actor = ppo_trainer.actor()
    qwen_ppo = Qwen2_5.from_model(
        model=actor.model(),
        lora_config=actor.lora_config(),
        tokenizer=tokenizer,
    )
    qwen_ppo.save(save_dir='./outputs/gsm8k-ppo-qwen2_5-0_5B')
