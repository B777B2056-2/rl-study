import math
from dataclasses import dataclass
import torch
import numpy as np
from copy import deepcopy
from llm_rl.plotter import Plotter
from llm_rl.models import LoraConfig, inject_lora
from typing import List, Tuple, Optional
from tqdm import tqdm
from llm_rl.ckpt import *

from src.search_r1.dataset import NqAndHotpotqaSample, NqAndHotpotqaDataset
from src.search_r1.environment import Environment
from src.search_r1.reward import SearchR1Reward


@dataclass
class GRPOConfig:
    dataset: NqAndHotpotqaDataset
    ckpt_config: CkptConfig
    max_new_tokens: int
    temperature: float
    top_p: float
    learning_rate: float
    actor_lora_config: LoraConfig
    kl_beta: float  # KL惩罚系数
    n_sample_group: int # 组内归一化时，一组采样多少个回答
    n_epoch: int # 策略模型训练轮次
    device: str
    ratio_clip_eps: float = 0.2 # 新旧logp裁剪
    grad_clip_eps: float = 1.0  # 梯度裁剪

@dataclass
class Trajectory:
    """单个轨迹"""
    input_ids: torch.LongTensor     # 输入的完整query，包含刚开始的prompt，以及中间动作的观察
    old_log_probs: torch.Tensor     # 轨迹的完整输出对数概率，(1, L-1)，和 loss_mask 对齐；非模型位置的值无意义，靠 mask 屏蔽
    loss_mask: torch.LongTensor     # 损失的掩码，为1的位置参与损失计算，0的位置不参与
    reward: float                   # 该轨迹对应的奖励值

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
    def advantages(self) -> Optional[torch.Tensor]:
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
    def __init__(self, base_model, tokenizer, env: Environment, config: GRPOConfig):
        self._plotter = Plotter()
        ckpt_manager = CheckpointManager(config=config.ckpt_config)
        super().__init__(ckpt_manager, self._plotter)

        self._env = env
        self._config = config
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

    def _actor_generate_full_reply(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """actor生成一次完整回答，返回：生成的文本、output_ids、output_attention_mask"""
        # 1. 生成完整序列
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

        # 2. 构造 output_attention_mask（遇到第一个 eos 后置 0）
        output_attention_mask = torch.ones_like(output_ids)
        for i in range(output_ids.size(0)):
            eos_pos = (output_ids[i] == self._tokenizer.eos_token_id).nonzero()
            if len(eos_pos) > 0:
                first_eos = eos_pos[0].item()
                output_attention_mask[i, first_eos+1:] = 0

        # 3. 解码新生成的文本
        prompt_len = input_ids.shape[1]
        generated_ids = output_ids[0, prompt_len:]          # 只取新生成的 token
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text, output_ids, output_attention_mask

    @classmethod
    def _calc_log_probs(cls, model, input_ids, attn_mask):
        """针对Actor或Reference，计算对数概率、助手回复的token ids"""
        logits = model(input_ids=input_ids, attention_mask=attn_mask)

        # 偏移一位：位置 t-1 预测位置 t 的 token
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)    # (1, L-1, V)
        target_ids = input_ids[:, 1:]                          # (1, L-1)
        token_logp = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)   # (1, L-1)
        
        return token_logp, target_ids

    def _generate_one_full_trajectory(self, prompt: str, golden_answers: List[str]) -> Trajectory:
        """生成一条完整轨迹（prompt -> reply -> action -> observation -> action -> ... -> final_answer)"""        
        # 1. 重置环境
        self._env.reset()

        # 2. 循环生成决策，直到结束（输出最终回答 或 非法命令）
        # 第一轮：走 chat template
        prompt_text = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = torch.tensor([self._tokenizer.encode(prompt_text, add_special_tokens=False)], dtype=torch.long, device=self._config.device)

        done = False
        delta_input_text = ""                            # 增量输入的文本，向量化后拼在full_ids后面输入，自然带上了之前的历史
        full_ids = prompt_ids                            # [prompt | gen1 | info1 | gen2 | info2 | gen3]
        loss_mask = torch.zeros_like(input=prompt_ids)   # 遮住所有非模型生成的位置，这些位置不参与后续前向与loss计算

        while not done:
            # 1. 编码增量文本，拼接到full_ids上
            if delta_input_text != "":
                # 编码
                delta_input_ids = self._tokenizer.encode(delta_input_text, add_special_tokens=False)
                delta_input_ids = torch.tensor([delta_input_ids], dtype=torch.long, device=self._actor.device)
                # 拼接
                full_ids = torch.cat([full_ids, delta_input_ids], dim=1)
                loss_mask = torch.cat([loss_mask, torch.zeros(size=delta_input_ids.size(), dtype=loss_mask.dtype, device=loss_mask.device)], dim=1)

            # 2. 生成一次回答，获取决策命令；full_ids无padding，故attn mask全1
            command, output_ids, _ = self._actor_generate_full_reply(input_ids=full_ids, attention_mask=torch.ones_like(full_ids))
            
            # 拼接
            prompt_len = full_ids.shape[1]
            generated_ids = output_ids[:, prompt_len:]  # output_ids包含此时的full_ids+真正生成的tokens，需要切一下
            full_ids = torch.cat([full_ids, generated_ids], dim=1)
            loss_mask = torch.cat([loss_mask, torch.ones(size=generated_ids.size(), dtype=loss_mask.dtype, device=loss_mask.device)], dim=1)

            # 3. 触发环境进行一次行动，获取行动观察结果、结束标志
            obs, done = self._env.action(cmd=command)
            delta_input_text = obs

        # 3. actor前向一次，获取轨迹的输出完整对数概率
        with torch.no_grad():
            actor_log_probs, _ = GRPOTrainer._calc_log_probs(
                model=self._actor, 
                input_ids=full_ids, 
                attn_mask=torch.ones_like(full_ids),
            )   # full_ids无padding，故attn mask全1

        # 4. 根据金标准，计算单轨迹奖励
        reward = SearchR1Reward.calc(final_answer=self._env.get_final_answer(), golden_answers=golden_answers)

        return Trajectory(
            input_ids=full_ids,             # (1, L)
            old_log_probs=actor_log_probs,  # (1, L-1)，对应 input_ids[:, 1:]
            loss_mask=loss_mask[:, 1:],     # (1, L-1)，与old_log_probs对齐
            reward=reward,
        )

    @torch.no_grad()
    def _rollout(self, sample: NqAndHotpotqaSample) -> TrajectoryGroup:
        """
            单batch采样数据轨迹
        """
        
        group_trajectories = TrajectoryGroup()
        # 生成一组轨迹
        for _ in range(self._config.n_sample_group):
            # actor完整生成一次轨迹
            trajectory = self._generate_one_full_trajectory(prompt=sample.question, golden_answers=sample.golden_answers)
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

    def train(self) -> None:
        """grpo训练"""
        # 从 checkpoint 恢复
        self.maybe_resume()

        train_data_loader = self._config.dataset.train_data_loader()
        for ep in range(self._epoch, self._config.n_epoch):
            self.set_epoch(ep)
            step_cnt, total_loss = 0, 0.0

            for batch in tqdm(train_data_loader, desc="Training"):
                # 一个batch
                groups = []
                for sample in batch:
                    # 1. 采样轨迹 + 组内归一化
                    group = self._rollout(sample)
                    groups.append(group)

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

                # 4. 定期保存 checkpoint
                self.maybe_save()

            tqdm.write(f"""epoch: {ep}, loss={total_loss/step_cnt}""")
            self._plotter.log(name="train_loss", value=total_loss/step_cnt, step=ep)
        self._plotter.plot(names=['train_loss'])

    @torch.no_grad()
    def eval(self) -> None:
        """
        测试集上评估
        """
        self._actor.eval()

        test_data_loader = self._config.dataset.test_data_loader()

        for batch in tqdm(test_data_loader, desc="Evaluating"):
            for sample in batch:
                # 采样轨迹 + 组内归一化
                group = self._rollout(sample)

                self._plotter.log("eval_advantage", group.advantages.mean().item())

        self._actor.train()
        self._plotter.plot(names=["eval_advantage"])

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
    from llm_rl.models import Qwen2_5, LoraConfig
    from llm_rl.ckpt import CkptConfig

    from src.search_r1.dataset import NqAndHotpotqaDataset
    from src.search_r1.search_tool import DuckDuckGoSearchTool
    from src.search_r1.environment import WebSearchEnvironment

    # ============================================================
    # 1. 加载模型
    # ============================================================
    qwen2_5 = Qwen2_5()
    tokenizer = qwen2_5.tokenizer()
    base_model = qwen2_5.model()
    print(f"模型已加载, 设备: {qwen2_5.device}")

    # ============================================================
    # 2. Search-R1环境
    # ============================================================
    env = WebSearchEnvironment(max_turns=8, search_tool=DuckDuckGoSearchTool(top_k=2))

    # ============================================================
    # 3. Search-R1数据集
    # ============================================================
    dataset = NqAndHotpotqaDataset()

    # ============================================================
    # 4. GRPO 训练器
    # ============================================================
    grpo_trainer = GRPOTrainer(
        base_model=base_model,
        tokenizer=tokenizer,
        env=env,
        config=GRPOConfig(
            dataset=dataset,

            ckpt_config=CkptConfig(
                ckpt_dir="outputs/search-r1/ckpt",
                save_steps=5,
                keep_last=3,
            ),

            # 采样
            max_new_tokens=256,
            temperature=1.0,
            top_p=1.0,
            n_sample_group=4,

            # 训练
            n_epoch=1,
            learning_rate=1e-6,
            kl_beta=0,
            ratio_clip_eps=0.2,
            grad_clip_eps=1.0,

            # LoRA
            actor_lora_config=LoraConfig(
                alpha=16,
                rank=8,
                dropout=0.05,
                use_qlora=True,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),

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
    qwen_grpo.save(save_dir="./outputs/search-r1/grpo-qwen2_5-0_5B")
