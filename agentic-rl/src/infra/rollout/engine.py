from dataclasses import dataclass
from typing import Optional
import torch

from src.infra.log import get_logger
from src.infra.dataset.dataset import BaseSample
from src.infra.environment.env import Environment
from src.infra.reward.reward import AgentTraceType, AgentTraceItem, RewardCalcContext, RewardFn
from src.infra.rollout.backends import InferenceBackend


logger = get_logger(__name__)


@dataclass
class Trajectory:
    """单个轨迹"""
    input_ids: torch.Tensor     # 输入的完整query，包含刚开始的prompt，以及中间动作的观察
    old_log_probs: torch.Tensor # 轨迹的完整输出对数概率，(1, L-1)，和 loss_mask 对齐；非模型位置的值无意义，靠 mask 屏蔽
    loss_mask: torch.Tensor     # 损失的掩码，为1的位置参与损失计算，0的位置不参与
    reward: float               # 该轨迹对应的奖励值


class RolloutEngine(object):
    """采样引擎，进行一次完整轨迹采样"""
    def __init__(self,
                 tokenizer,
                 env: Environment,
                 reward_fn: RewardFn,
                 inference_backend: InferenceBackend,
                 device: str,
                ):
        super().__init__()

        self._tokenizer = tokenizer
        self._env = env
        self._reward_fn = reward_fn
        self._inference_backend = inference_backend
        self._device = device
    
    def sync_weights(self, state_dict: dict) -> None:
        """从训练侧同步权重"""
        self._inference_backend.sync_weights(state_dict=state_dict)

    def rollout(self, current_step: int, sample: BaseSample) -> Trajectory:
        """生成一条完整轨迹（prompt -> reply -> action -> observation -> action -> ... -> final_answer)"""
        reward_calc_ctx = RewardCalcContext(current_step=current_step) 
        reward_calc_ctx.agent_full_trace.append(AgentTraceItem(a_type=AgentTraceType.INPUT, content=sample.user_prompt))
        # 1. 重置环境
        self._env.reset()

        # 2. 循环生成决策，直到结束（输出最终回答 或 非法命令）
        # 第一轮：走 chat template；可选注入外部自定义 system prompt
        messages = []
        if sample.system_prompt:
            messages.append({"role": "system", "content": sample.system_prompt})
        messages.append({"role": "user", "content": sample.user_prompt})
        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        prompt_ids = torch.tensor([self._tokenizer.encode(prompt_text, add_special_tokens=False)], dtype=torch.long, device=self._device)

        done = False
        delta_input_text = ""                            # 增量输入的文本，向量化后拼在full_ids后面输入，自然带上了之前的历史
        full_ids = prompt_ids                            # [prompt | gen1 | info1 | gen2 | info2 | gen3]
        loss_mask = torch.zeros_like(input=prompt_ids)   # 遮住所有非模型生成的位置，这些位置不参与后续前向与loss计算
        old_log_probs = torch.zeros(
            1, prompt_ids.size(1) - 1, 
            device=self._device,
        ) # 初始：prompt 段的前向 logprob，长度 L_prompt - 1，全 0

        while not done:
            # 1. 编码增量文本，拼接到full_ids上
            if delta_input_text != "":
                # 编码
                delta_input_ids = self._tokenizer.encode(delta_input_text, add_special_tokens=False)
                delta_input_ids = torch.tensor([delta_input_ids], dtype=torch.long, device=self._device)
                # 拼接
                full_ids = torch.cat([full_ids, delta_input_ids], dim=1)
                loss_mask = torch.cat([loss_mask, torch.zeros(size=delta_input_ids.size(), dtype=loss_mask.dtype, device=loss_mask.device)], dim=1)
                # 补 obs 的占位：长度和 full_ids 同步，值全 0（会被 loss_mask 盖掉）
                old_log_probs = torch.cat([old_log_probs, torch.zeros(delta_input_ids.size(), dtype=old_log_probs.dtype, device=old_log_probs.device)], dim=1)

            # 2. 生成一次回答，获取决策命令；full_ids无padding，故attn mask全1
            gen_ids, gen_logps = self._inference_backend.generate(input_ids=full_ids, attention_mask=torch.ones_like(full_ids))
            
            # 拼接
            full_ids = torch.cat([full_ids, gen_ids], dim=1)
            loss_mask = torch.cat([loss_mask, torch.ones(size=gen_ids.size(), dtype=loss_mask.dtype, device=loss_mask.device)], dim=1)
            old_log_probs = torch.cat([old_log_probs, gen_logps], dim=1)

            # 3. 触发环境进行一次行动，获取行动观察结果、结束标志
            command = self._tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            obs, done = self._env.action(cmd=command)
            if not done:
                reward_calc_ctx.agent_full_trace.append(AgentTraceItem(a_type=AgentTraceType.ACTION, content=command))
            else:
                reward_calc_ctx.agent_full_trace.append(AgentTraceItem(a_type=AgentTraceType.END, content=command))
            delta_input_text = obs

        # 4. 计算单轨迹奖励
        logger.info(f"[RolloutEngine] 轨迹： {str(reward_calc_ctx)}")
        reward = self._reward_fn.score(ctx=reward_calc_ctx, sample=sample)

        return Trajectory(
            input_ids=full_ids,             # (1, L)
            old_log_probs=old_log_probs,    # (1, L-1)，对应 input_ids[:, 1:]
            loss_mask=loss_mask[:, 1:],     # (1, L-1)，与old_log_probs对齐
            reward=reward,
        )
