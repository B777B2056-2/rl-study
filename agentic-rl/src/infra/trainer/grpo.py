import math
import os
from dataclasses import dataclass
import torch
import numpy as np
from copy import deepcopy
from typing import List, Optional
from tqdm import tqdm

from src.infra.plotter import Plotter
from src.infra.models.lora import LoraConfig, inject_lora
from src.infra.trainer.ckpt import *
from src.infra.dataset.dataset import BaseSample, BaseDataset
from src.infra.rollout.backends import *
from src.infra.rollout.engine import Trajectory, RolloutEngine
from src.infra.environment.env import Environment
from src.infra.reward.reward import RewardFn


@dataclass
class GRPOConfig:
    ckpt_config: CkptConfig
    learning_rate: float
    actor_lora_config: LoraConfig
    kl_beta: float  # KL惩罚系数
    n_sample_group: int # 组内归一化时，一组采样多少个回答
    n_epoch: int # 策略模型训练轮次
    inference_backend_config: InferenceBackendConfig    # 采样引擎推理配置
    device: str
    ratio_clip_eps: float = 0.2 # 新旧logp裁剪
    grad_clip_eps: float = 1.0  # 梯度裁剪
    system_prompt: Optional[str] = None  # 采样时注入的 system prompt（None 表示不注入）

class TrajectoryGroup:
    "一个组，包含多条轨迹"
    def __init__(self):
        self._trajectories: List[Trajectory] = []       # 轨迹集合
        self._advantages: Optional[torch.Tensor] = None  # 组内轨迹归一化后的优势估计，从轨迹集合计算而来
        self._device = None

    def append_trajectory(self, trajectory: Trajectory) -> None:
        """添加一条轨迹到组内"""
        self._trajectories.append(trajectory)

        if self._device is None:
            self._device = trajectory.old_log_probs.device

    def calc_advantage(self):
        """计算组的优势估计，即：组内归一化"""
        if len(self._trajectories) == 0:
            return

        # 组内归一化
        rewards = np.array([t.reward for t in self._trajectories], dtype=np.float32)
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        advantages = (rewards - mean_r) / (std_r + 1e-8)
        self._advantages = torch.from_numpy(advantages).to(self._device)

    @property
    def trajectories(self) -> List[Trajectory]:
        return self._trajectories

    @property
    def advantages(self) -> torch.Tensor:
        assert self._advantages is not None
        return self._advantages

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
    def __init__(self, base_model, tokenizer, dataset: BaseDataset, env: Environment,
                    reward_fn: RewardFn, config: GRPOConfig):
        self._plotter = Plotter(storage_path=os.path.join(config.ckpt_config.ckpt_dir, "metrics.jsonl"))
        ckpt_manager = CheckpointManager(config=config.ckpt_config)
        super().__init__(ckpt_manager, self._plotter)

        self._config = config
        self._dataset = dataset
        self._tokenizer = tokenizer

        self._actor = Actor(
            base_model=deepcopy(base_model),
            lora_config=config.actor_lora_config,
        ).to(config.device)

        # 参考模型
        if math.isclose(config.kl_beta, 0, abs_tol=1e-8):
            self._ref_model = None  # kl系数为0，不需要参考模型
        else:
            self._ref_model = Reference(base_model=base_model)

        self._trainable_parameters = [p for p in self._actor.parameters() if p.requires_grad]
        self._optimizer = torch.optim.Adam(
            self._trainable_parameters, 
            lr=config.learning_rate,
        )

        # 构建采样引擎
        self._rollout_engine = self._build_rollout_engine(env=env, reward_fn=reward_fn)
        # 首次同步初始权重
        self._rollout_engine.sync_weights(self._actor.state_dict())

    def _build_rollout_engine(self, env: Environment, reward_fn: RewardFn) -> RolloutEngine:
        """
        构建采样引擎：根据配置 + 训练侧资源，构造 Backend。
        """
        cfg = self._config.inference_backend_config
        if isinstance(cfg.backend, HFBackendConfig):
            # HF 后端：从 Actor 拿模型对象
            rollout_inference_backend = HFBackend(
                model=self._actor,
                tokenizer=self._tokenizer,
                params=cfg.params,
            )
        else:
            raise RuntimeError(f"Unsupported backend: {type(cfg.backend)}")

        return RolloutEngine(
            tokenizer=self._tokenizer,
            env=env,
            reward_fn=reward_fn,
            inference_backend=rollout_inference_backend,
            device=self._config.device,
            system_prompt=self._config.system_prompt,
        )

    @classmethod
    def _calc_log_probs(cls, model, input_ids, attn_mask):
        """针对Actor或Reference，计算对数概率、助手回复的token ids"""
        logits = model(input_ids=input_ids, attention_mask=attn_mask)

        # 偏移一位：位置 t-1 预测位置 t 的 token
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)    # (1, L-1, V)
        target_ids = input_ids[:, 1:]                          # (1, L-1)
        token_logp = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)   # (1, L-1)
        
        return token_logp, target_ids

    @torch.no_grad()
    def _rollout(self, sample: BaseSample) -> TrajectoryGroup:
        """
            单batch采样数据轨迹
        """
        
        group_trajectories = TrajectoryGroup()
        # 生成一组轨迹
        for _ in range(self._config.n_sample_group):
            # actor完整生成一次轨迹
            trajectory = self._rollout_engine.rollout(sample=sample)
            group_trajectories.append_trajectory(trajectory)

        # 组内归一化估计优势
        group_trajectories.calc_advantage()
        return group_trajectories

    def _clip(self, logp_ratio):
        """裁剪：限制概率比的范围，每次策略网络训练只走一小步，保持训练稳定"""
        return torch.clamp(logp_ratio, 1 - self._config.ratio_clip_eps, 1 + self._config.ratio_clip_eps)

    def _compute_batch_loss(self, groups: List[TrajectoryGroup]) -> torch.Tensor:
        """计算 batch group 的 loss（单group内所有轨迹累加）。"""
        # 1. 对多个组内的多个轨迹的old_log_probs、loss_mask，找出最长的，然后进行padding + stack
        batch_size = len(groups)
        max_traj_len = 0
        for group in groups:
            for traj, adv in zip(group.trajectories, group.advantages):
                max_traj_len = max(max_traj_len, traj.input_ids.shape[1])
        # padding + cat
        batch_input_ids = None
        batch_attn_mask = None
        batch_loss_mask = None
        batch_old_logp = None
        batch_adv = None
        for group in groups:
            for traj, adv in zip(group.trajectories, group.advantages):
                # padding
                cur_len = traj.input_ids.shape[1]
                pad_input_ids = torch.nn.functional.pad(traj.input_ids, (0, max_traj_len - cur_len), value=self._tokenizer.pad_token_id)
                pad_attn_mask = torch.nn.functional.pad(torch.ones_like(traj.input_ids), (0, max_traj_len - cur_len), value=0)
                pad_loss_mask = torch.nn.functional.pad(traj.loss_mask, (0, max_traj_len - cur_len), value=0)
                pad_old_logp = torch.nn.functional.pad(traj.old_log_probs, (0, max_traj_len - cur_len), value=0)
                # cat
                batch_input_ids = torch.cat((batch_input_ids, pad_input_ids), dim=0) if batch_input_ids is not None else pad_input_ids
                batch_attn_mask = torch.cat((batch_attn_mask, pad_attn_mask), dim=0) if batch_attn_mask is not None else pad_attn_mask
                batch_loss_mask = torch.cat((batch_loss_mask, pad_loss_mask), dim=0) if batch_loss_mask is not None else pad_loss_mask
                batch_old_logp = torch.cat((batch_old_logp, pad_old_logp), dim=0) if batch_old_logp is not None else pad_old_logp
            # adv无需padding，直接stack
            batch_adv = torch.cat((batch_adv, group.advantages), dim=0) if batch_adv is not None else group.advantages

        assert batch_input_ids is not None
        assert batch_attn_mask is not None
        assert batch_loss_mask is not None
        assert batch_old_logp is not None
        assert batch_adv is not None

        # 2. 一次策略网络前向传播，获取新动作、新动作对应的新概率分布
        new_log_probs, _ = GRPOTrainer._calc_log_probs(
            model=self._actor, 
            input_ids=batch_input_ids, 
            attn_mask=batch_attn_mask,
        )   # (batch_size, L_max-1)

        # 计算概率比：新概率 / 旧概率，使用exp保证数值稳定性，避免极小概率相除导致溢出
        ratio = torch.exp(new_log_probs - batch_old_logp)   # (batch_size, L_max-1)
    
        # 策略损失：让"好动作"的概率变大，让"坏动作"的概率变小
        adv = batch_adv.unsqueeze(1)    # advantage 广播，(batch_size, 1)
        policy_loss = -torch.min(ratio * adv,  self._clip(ratio) * adv)  # (batch_size, 1)

        if self._ref_model is not None:
            # 算 KL
            with torch.no_grad():
                ref_log_probs, _ = GRPOTrainer._calc_log_probs(
                    model=self._ref_model, 
                    input_ids=batch_input_ids, 
                    attn_mask=batch_attn_mask,
                )
            kl = new_log_probs - ref_log_probs

            # 按token的损失 = 策略损失 + beta * kl
            per_token_loss = policy_loss + self._config.kl_beta * kl
        else:
            # 按token的损失 = 策略损失
            per_token_loss = policy_loss

        # 4. 用 mask 筛
        per_token_loss = per_token_loss * batch_loss_mask
        total_loss = per_token_loss.sum()   # 求和获取总loss
        total_tokens = batch_loss_mask.sum().clamp(min=1.0)

        return total_loss / max(total_tokens, 1)

    def train(self, save=None, show=True) -> None:
        """grpo训练"""
        # 从 checkpoint 恢复
        self.maybe_resume()

        train_data_loader = self._dataset.train_data_loader()
        for ep in range(self._epoch, self._config.n_epoch):
            self.set_epoch(ep)
            step_cnt, total_loss = 0, 0.0

            for batch in tqdm(train_data_loader, desc="Training"):
                # 一个batch
                self._actor.eval()   # 采样阶段用 eval：关闭 dropout/梯度检查点，generate 在 no_grad 下
                groups = []
                for sample in batch:
                    # 1. 采样轨迹 + 组内归一化
                    group = self._rollout(sample)
                    groups.append(group)

                self._actor.train()  # 更新阶段用 train：开启 dropout 和梯度检查点
                # 2. 计算loss
                batch_group_loss = self._compute_batch_loss(groups)

                # 绘图相关
                total_loss += batch_group_loss.item()
                step_cnt += 1

                # 3. 梯度下降
                self._optimizer.zero_grad()
                batch_group_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._trainable_parameters, self._config.grad_clip_eps)  # 梯度裁剪
                self._optimizer.step()

                # 4. 权重同步到推理后端
                self._rollout_engine.sync_weights(self._actor.state_dict())

                # 5. 定期保存 checkpoint
                self.maybe_save()

            tqdm.write(f"""epoch: {ep}, loss={total_loss/step_cnt}""")
            self._plotter.log(name="train_loss", value=total_loss/step_cnt, step=ep)
        self._plotter.plot(names=['train_loss'], save=save, show=show)

    @torch.no_grad()
    def eval(self, save=None, show=True) -> None:
        """测试集上评估：逐样本 rollout 一次，用 reward（EM）统计准确率"""
        self._actor.eval()

        test_data_loader = self._dataset.test_data_loader()
        rewards: list[float] = []

        for batch in tqdm(test_data_loader, desc="Evaluating"):
            for sample in batch:
                # 每个样本采样 1 条轨迹，traj.reward 已是 SearchR1Reward 打的 0/1 分
                traj = self._rollout_engine.rollout(sample)
                rewards.append(traj.reward)

        acc = float(np.mean(rewards)) if rewards else 0.0
        self._plotter.log("eval_acc", acc, step=self.global_step)
        self._actor.train()
        self._plotter.plot(names=["eval_acc"], save=save, show=show)
        tqdm.write(f"eval acc (EM) = {acc:.4f}")


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
