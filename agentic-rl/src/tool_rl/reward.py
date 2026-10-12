import re
import string

from src.infra.dataset.dataset import BaseSample
from src.infra.reward.reward import RewardCalcContext, RewardFn
from src.tool_rl.dataset import ToolRLSample


class ToolRLReward(RewardFn):
    """Tool-RL奖励"""

    def __init__(self) -> None:
        super().__init__()

    def score(self, ctx: RewardCalcContext, sample: BaseSample) -> float:
        """计算最终奖励"""
        assert isinstance(sample, ToolRLSample)
        raise NotImplementedError("奖励还未实现")