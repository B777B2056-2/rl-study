import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List
from src.infra.dataset.dataset import BaseSample


class AgentTraceType(Enum):
    INPUT = "input"
    ACTION = "action"
    END = "end"


@dataclass
class AgentTraceItem:
    a_type: AgentTraceType
    content: str

    @property
    def end(self) -> bool:
        return self.a_type == AgentTraceType.END

    def to_dict(self) -> dict:
        return {
            "type": self.a_type.value,
            "content": self.content,
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class RewardCalcContext:
    """rollout采样时，计算奖励的上下文"""
    agent_full_trace: List[AgentTraceItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent_full_trace": [
                item.to_dict() for item in self.agent_full_trace
            ]
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class RewardFn(ABC):
    """奖励函数接口。规则奖励只实现 score，RM 覆盖 score_batch。"""

    @abstractmethod
    def score(self, ctx: RewardCalcContext, sample: BaseSample) -> float:
        """逐条打分"""
        ...
    
    def batch_score(self, ctxs: list[RewardCalcContext], samples: List[BaseSample]) -> list[float]:
        """批量打分。默认串行实现，子类可覆盖它做批量前向。"""
        return [self.score(ctx, sample) for ctx, sample in zip(ctxs, samples)]