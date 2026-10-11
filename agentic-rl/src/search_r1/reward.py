from typing import List
import re
import string

from src.infra.dataset.dataset import BaseSample
from src.infra.reward.reward import RewardCalcContext, RewardFn
from src.search_r1.dataset import NqAndHotpotqaSample


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[%s]" % re.escape(string.punctuation))
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class SearchR1Reward(RewardFn):
    """Search-R1奖励"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def _normalize(cls, text: str) -> str:
        """把答案文本归一化，用于 Exact Match 比较。"""
        if not isinstance(text, str):
            return ""
        # 1. 转小写
        text = text.lower()
        # 2. 去标点
        text = _PUNCT_RE.sub("", text)
        # 3. 连续空白压成单个空格
        text = _WS_RE.sub(" ", text)
        # 4. 去首尾空白
        text = text.strip()
        return text

    @staticmethod
    def _extract_answer(text: str) -> str:
        """从模型原始输出中提取 <answer>...</answer> 的内容，未找到返回空串。"""
        m = _ANSWER_RE.search(text)
        return m.group(1).strip() if m else ""

    def score(self, ctx: RewardCalcContext, sample: BaseSample) -> float:
        """根据金标准，计算最终奖励"""
        assert isinstance(sample, NqAndHotpotqaSample)
        last = ctx.agent_full_trace[-1]
        final_answer = SearchR1Reward._extract_answer(last.content) if last.end else ''
        if not final_answer:
            return 0.0
        
        gold = {SearchR1Reward._normalize(g) for g in sample.golden_answers}
        return 1.0 if SearchR1Reward._normalize(final_answer) in gold else 0.0