from typing import List
import re
import string


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[%s]" % re.escape(string.punctuation))


class SearchR1Reward(object):
    """Search-R1奖励"""
    @classmethod
    def calc(self, final_answer: str, golden_answers: List[str]) -> float:
        """根据金标准，计算最终奖励"""
        if final_answer is None or final_answer == '':
            return 0.0
        def normalize(text: str) -> str:
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
        return 1.0 if normalize(final_answer) in {normalize(g) for g in golden_answers} else 0.0