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
class GRPOConfig:
    dataset: RLDataset
    ckpt_config: CkptConfig
    max_new_tokens: int
    temperature: float
    top_p: float
    learning_rate: float
    actor_lora_config: LoraConfig
    kl_beta: float  # KL惩罚系数
    n_sample_group: int # 组内归一化时，一组采样多少个回答
    reward_fn: Callable # 奖励函数，可以是奖励模型的输出、规则奖励结果，不关心具体实现
    n_epoch: int # 策略模型训练轮次
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
        做法：SFT 模型（冻结） + GRPO LoRA（训练）；若SFT本身为Lora来的，则需把SFT-Lora权重合并到原模型，再进行PPO
    """
    def __init__(self, base_model, lora_config: LoraConfig):
        super(Actor, self).__init__()
        # 冻结SFT模型
        for p in base_model.parameters():
            p.requires_grad = False

        if lora_config is not None:
            # 注入 LoRA
            inject_lora(base_model, lora_config)

        self._actor = base_model
        self._lora_config = lora_config

    def forward(self, input_ids, attention_mask):
        # 获取actor模型前向传播的输出
        logits = self._actor(input_ids, attention_mask=attention_mask).logits
        return logits

    def generate(self, *args, **kwargs):
        # 生成一次完整回答
        return self._actor.generate(*args, **kwargs)

    def model(self):
        return self._actor

    def lora_config(self):
        return self._lora_config

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


class GRPOTrainer(CheckpointableTrainer):
    """
        Actor（策略网络）：输入状态，输出动作概率，负责选动作。
        Critic（价值网络）：被省略，采样同一prompt的一组输出回答，通过RM为一组回答给出奖励，把一组奖励的平均值作为价值

        状态 s
            ├→ Actor → 动作分布 π(a|s) → 采样动作 a
            └→ 采样一组回答 → 组内归一化 → V(s)
                            ↓
                    算优势 A(s,a)
                            ↓
                        Actor 更新

        状态空间：LLM为当前已生成的Token
        动作空间：在LLM里，动作为生成下一个token，因此为词表大小
    """
    def __init__(self, base_model, tokenizer, config: GRPOConfig):
        self._plotter = Plotter()
        ckpt_manager = CheckpointManager(config=config.ckpt_config)
        super().__init__(ckpt_manager, self._plotter)

        self._config = config
        self._tokenizer = tokenizer

        self._actor = Actor(
            base_model=deepcopy(base_model),
            lora_config=config.actor_lora_config,
        ).to(config.device)

        self._ref_model = Reference(base_model=base_model) # 参考模型

        self._trainable_parameters = [p for p in self._actor.parameters() if p.requires_grad]
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
        logits = model(output_ids, output_attention_mask)      # (batch_size * n_group, len(prompt + response), vocab_size)
        response_logits = logits[:, prompt_len-1:-1, :]        # (batch_size * n_group, len(response), vocab_size)
        log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
        response_ids = output_ids[:, prompt_len:]                       # (batch_size * n_group, len(response), vocab_size)
        log_probs = log_probs.gather(-1, response_ids.unsqueeze(-1))    # (batch_size * n_group, len(response), vocab_size)
        log_probs = log_probs.squeeze(-1)                               # (batch_size * n_group, len(response))
        return log_probs, response_ids

    @torch.no_grad()
    def _collect_traces(self, input_ids, attention_mask, ground_truth):
        """
            单batch采样数据轨迹
            方法：用当前策略和环境交互，收集的一批 (s, a, r, s', done, log_prob, V) 数据；不进行梯度更新
        """

         # 1. 采样一组完整输出
        group_responses = []
        group_responses_mask = []
        for _ in range(self._config.n_sample_group):
            # actor完整生成一次回答
            output_ids, output_attention_mask = self._actor_generate_full_replay(input_ids=input_ids, attention_mask=attention_mask)
            group_responses.append(output_ids)
            group_responses_mask.append(output_attention_mask)
        # 拼接为(batch_size, n_group, len(prompt)+len(response))
        output_ids = torch.stack(group_responses, dim=1)
        output_attention_mask = torch.stack(group_responses_mask, dim=1)

        # 输入策略模型，获取决策出来的动作（即 下一Token预测概率）
        batch_size = input_ids.size(0)
        prompt_len = input_ids.size(1)
        # 展平为(batch_size * n_group, len(prompt)+len(response))，进行一次前向
        flat_ids = output_ids.reshape(batch_size * self._config.n_sample_group, -1)
        flat_mask = output_attention_mask.reshape(batch_size * self._config.n_sample_group, -1)
        actor_log_probs, actor_response_ids = GRPOTrainer._calc_log_probs(prompt_len, flat_ids, self._actor, flat_mask) # (batch_size * n_group, len(response))

        # 算奖励分数
        rm_scores_list = []
        group_response_ids = actor_response_ids.reshape(batch_size, self._config.n_sample_group, -1)
        for n in range(self._config.n_sample_group):
            response_ids = group_response_ids[:, n, :]
            responses = self._tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            rm_scores = torch.tensor(
                [self._config.reward_fn(r, gt) for r, gt in zip(responses, ground_truth)],
                dtype=torch.float32,
                device=self._config.device,
            )
            rm_scores_list.append(rm_scores)
        rm_scores = torch.stack(rm_scores_list, dim=1)  # 拼接为(batch_size, n_group)

        #  组内归一化
        mean_r = rm_scores.mean(dim=1, keepdim=True)       # (batch_size, 1)
        std_r = rm_scores.std(dim=1, keepdim=True)         # (batch_size, 1)
        advantages = (rm_scores - mean_r) / (std_r + 1e-8) # (batch_size, n_group)

        # 输入参考模型，计算KL惩罚
        ref_log_probs, _ = GRPOTrainer._calc_log_probs(prompt_len, flat_ids, self._ref_model, flat_mask)
        kl = actor_log_probs - ref_log_probs        # (batch_size * n_group, len(response))
        rewards = -self._config.kl_beta * kl
        rewards[:, -1] += rm_scores.reshape(-1) # 末尾加 RM

        rewards = rewards.reshape(batch_size, self._config.n_sample_group, -1)
        actor_log_probs = actor_log_probs.reshape(batch_size, self._config.n_sample_group, -1)
        
        return {
            "batch_size":     batch_size,
            "prompt_len":     prompt_len,
            "response_len":   actor_response_ids.shape[1],
            "input_ids":      output_ids,                  # (batch_size, n_group, len(prompt)+len(response))
            "attention_mask": output_attention_mask,       # (batch_size, n_group, len(prompt)+len(response))
            "log_probs":      actor_log_probs,             # (batch_size, n_group, len(response))  旧策略每 token log_prob
            "advantages":     advantages,                  # (batch_size, n_group)
            "rewards":        rewards,                     # (batch_size, n_group, len(response))
        }

    def _clip(self, logp_ratio):
        """裁剪：限制概率比的范围，每次策略网络训练只走一小步，保持训练稳定"""
        return torch.clamp(logp_ratio, 1 - self._config.ratio_clip_eps, 1 + self._config.ratio_clip_eps)

    def _update_batch(self, traceInfoDict):
        """
            迭代训练一个batch
        """
        batch_size = traceInfoDict["batch_size"]
        response_len = traceInfoDict["response_len"]
        prompt_len = traceInfoDict["prompt_len"]
        input_ids = traceInfoDict["input_ids"]                 # (batch_size, n_group, len(prompt)+len(response))
        attention_mask = traceInfoDict["attention_mask"]       # (batch_size, n_group, len(prompt)+len(response))
        old_actor_log_probs = traceInfoDict["log_probs"].reshape(batch_size * self._config.n_sample_group, -1)       # (batch_size, n_group, len(response))
        advantages = traceInfoDict["advantages"]               # (batch_size, n_group)
        advantages = advantages.reshape(batch_size * self._config.n_sample_group, 1).expand(-1, response_len)       # 扩展到所有回答维度，变为(batch_size, n_group, len(response))

        # 2. batch迭代
        # 展平为(batch_size * n_group, len(prompt)+len(response))，进行一次前向
        flat_ids = input_ids.reshape(batch_size * self._config.n_sample_group, -1)
        flat_mask = attention_mask.reshape(batch_size * self._config.n_sample_group, -1)
        # 策略网络前向传播，获取新动作、新动作对应的新概率分布
        new_actor_log_probs, _ = GRPOTrainer._calc_log_probs(prompt_len, flat_ids, self._actor, flat_mask)
        # 计算概率比：新概率 / 旧概率，使用exp保证数值稳定性，避免极小概率相除导致溢出
        ratio = torch.exp(new_actor_log_probs - old_actor_log_probs)
        # 策略损失：让"好动作"的概率变大，让"坏动作"的概率变小
        policy_loss = -torch.min(ratio * advantages,  self._clip(ratio) * advantages).mean()
        # 总损失 = 策略损失
        loss = policy_loss

        # 梯度更新
        self._optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self._trainable_parameters, self._config.grad_clip_eps)
        self._optimizer.step()

        return loss.item()

    def train(self) -> None:
        """ppo训练"""
        # 从 checkpoint 恢复
        self.maybe_resume()

        train_data_loader = self._config.dataset.build_train_data_loader()
        for ep in range(self._epoch, self._config.n_epoch):
            self.set_epoch(ep)
            # 0. 遍历训练集
            step_cnt, loss = 0, 0.0
            for batch in tqdm(train_data_loader, desc="Training"):
                input_ids = batch["input_ids"].to(self._config.device)
                attention_mask = batch["attention_mask"].to(self._config.device)
                ground_truth = batch["ground_truth"]

                # 1. 采样轨迹 + 组内归一化
                traceInfoDict = self._collect_traces(input_ids, attention_mask, ground_truth)

                # 2. 多轮训练
                batch_loss = self._update_batch(traceInfoDict)
                loss += batch_loss
                step_cnt += 1

                # 3. 定期保存 checkpoint
                self.maybe_save()

            self._plotter.log(name="train_loss", value=loss/step_cnt, step=ep)
        self._plotter.plot(names=['train_loss'])

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
            "optimizer": self._optimizer.state_dict(),
            "rng": get_rng_state(),
        }

    def _load_state_dict(self, state: dict):
        current_actor = dict(self._actor.named_parameters())
        for n, p in state["actor_lora"].items():
            current_actor[n].data.copy_(p.to(current_actor[n].device))
        self._optimizer.load_state_dict(state["optimizer"])
        if "rng" in state:
            set_rng_state(state["rng"])


if __name__ == "__main__":
    import re
    from src.models import Qwen2_5, LoraConfig
    from src.data import GSM8kDatasetAdapter, RLDataset
    from src.ckpt import CkptConfig
    from src.rl.grpo import GRPOTrainer, GRPOConfig

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
    # 2. 加载 SFT 模型
    # ============================================================
    qwen2_5 = Qwen2_5.from_pretrained(
        save_dir="./outputs/gsm8k-sft-qwen2_5-0_5B",
    )
    tokenizer = qwen2_5.tokenizer()
    base_model = qwen2_5.model()
    print(f"SFT 模型已加载, 设备: {qwen2_5.device}")

    # ============================================================
    # 3. RL 数据集（只有 prompt + ground_truth）
    # ============================================================
    gsm8k_adapter = GSM8kDatasetAdapter()
    gsm8k_rl_dataset = RLDataset(
        tokenizer=tokenizer,
        adapter=gsm8k_adapter,
        batch_size=1,
        max_length=256,
    )

    train_loader = gsm8k_rl_dataset.build_train_data_loader()
    print(f"训练集大小: {len(train_loader.dataset)}")

    # ============================================================
    # 4. GRPO 训练器
    # ============================================================
    grpo_trainer = GRPOTrainer(
        base_model=base_model,
        tokenizer=tokenizer,
        config=GRPOConfig(
            dataset=gsm8k_rl_dataset,

            ckpt_config=CkptConfig(
                ckpt_dir="outputs/gsm8k-grpo/ckpt",
                save_steps=500,
                keep_last=3,
            ),

            # 采样
            max_new_tokens=256,
            temperature=1.0,
            top_p=1.0,
            n_sample_group=4,

            # 训练
            n_epoch=3,
            learning_rate=1e-6,
            kl_beta=0.01,
            gamma=1.0,
            lambda_val=0.95,
            ratio_clip_eps=0.2,
            grad_clip_eps=1.0,

            # LoRA
            actor_lora_config=LoraConfig(
                alpha=16,
                rank=8,
                dropout=0.05,
                use_qlora=False,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),

            # 奖励
            reward_fn=gsm8k_reward_fn,

            device="cuda",
        ),
    )

    # ============================================================
    # 5. 训练
    # ============================================================
    grpo_trainer.train()

    # ============================================================
    # 6. 评估
    # ============================================================
    grpo_trainer.eval()

    # ============================================================
    # 7. 保存 Actor
    # ============================================================
    actor = grpo_trainer.actor()
    qwen_grpo = Qwen2_5.from_model(
        model=actor.model(),
        lora_config=actor.lora_config(),
        tokenizer=tokenizer,
    )
    qwen_grpo.save(save_dir="./outputs/gsm8k-grpo-qwen2_5-0_5B")
