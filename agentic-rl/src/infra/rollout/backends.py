from dataclasses import dataclass
from typing import Protocol, Union, Any
import torch



@dataclass
class SamplingParams:
    max_new_tokens: int
    max_turns: int
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class HFBackendConfig:
    pass


@dataclass
class InferenceBackendConfig:
    # 共性参数
    params: "SamplingParams"
    
    # 后端特有（单字段，Union 类型）
    backend: Union[HFBackendConfig, Any]
    
    def __post_init__(self):
        if self.backend is None:
            raise ValueError("必须指定一个后端配置")
    
    @property
    def kind(self) -> str:
        return type(self.backend).__name__
    
    @property
    def hf(self):
        return self.backend if isinstance(self.backend, HFBackendConfig) else None


class InferenceBackend(Protocol):
    """推理后端。只负责"生成一段"和"同步权重"。"""
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        生成一段。返回 (gen_ids, gen_logprobs)，都是 (1, L_gen)。
        """
        ...
    
    def sync_weights(self, state_dict) -> None:
        """从训练侧同步权重。"""
        ...


class HFBackend:
    def __init__(self, model, tokenizer, params: SamplingParams):
        self.model = model
        self.tokenizer = tokenizer
        self.config = params
        self.model.eval()
    
    def generate(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        gen_ids = outputs.sequences[:, input_ids.size(1):]
        logps = []
        for t, score in enumerate(outputs.scores):
            logp = torch.log_softmax(score, dim=-1)
            logps.append(logp.gather(1, gen_ids[:, t:t+1]))
        gen_logps = torch.cat(logps, dim=1)
        
        return gen_ids, gen_logps
    
    def sync_weights(self, state_dict):
        pass   # 同一模型，无需同步
