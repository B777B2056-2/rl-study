from dataclasses import dataclass
from typing import Tuple, List
from enum import Enum
import re
from src.search_r1.search_tool import SearchResult, SearchTool
from abc import ABC, abstractmethod


class WebSearchActionType(Enum):
    SEARCH = "search"
    FINAL_ANSWER = "answer"
    INVALID = "invalid"

# 非贪婪 .*? + DOTALL 保证：匹配到最近的闭合标签，且内容可含换行
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_WS_RE = re.compile(r"\s+")

class Environment(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def action(self, cmd: str) -> Tuple[str, bool]:
        pass

    @abstractmethod
    def get_final_answer(self) -> str:
        pass

class WebSearchEnvironment(Environment):
    """环境类"""
    def __init__(self, max_turns: int, search_tool: SearchTool):
        super().__init__()
        self._max_turns = max_turns     # 最大轮次
        self._turns = 0                 # 已执行 action 次数
        self._final_answer = ''         # 最终结束时的回复
        self._search_tool = search_tool # 搜索工具

    def reset(self):
        """重置环境"""
        self._turns = 0
        self._final_answer = ''

    def _valid_and_parse_cmd(self, cmd: str) -> Tuple[WebSearchActionType, str]:
        """校验并解析指令"""
        # 优先级 1：最终答案
        answer_match = _ANSWER_RE.search(cmd)
        if answer_match:
            content = _WS_RE.sub(" ", answer_match.group(1)).strip()
            if content:
                return WebSearchActionType.ANSWER, content
            return WebSearchActionType.INVALID, ""

        # 优先级 2：搜索请求
        search_match = _SEARCH_RE.search(cmd)
        if search_match:
            content = _WS_RE.sub(" ", search_match.group(1)).strip()
            if content:
                return WebSearchActionType.SEARCH, content
            return WebSearchActionType.INVALID, ""

        # 优先级 3：无合法标签
        return WebSearchActionType.INVALID, ""

    def _format_for_llm(self, results: List[SearchResult]) -> str:
        """
        把搜索结果格式化为 <information>...</information> 内的文本。
        这是 LLM 最终看到的 observation 内容。
        """
        if not results:
            return "<information>No results found.</information>"
        
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r.title}\n{r.snippet}")
        body = "\n\n".join(parts)
        return f"<information>{body}</information>"

    def action(self, cmd: str) -> Tuple[str, bool]:
        """
            根据决策（即prompt），执行一次动作（比如调用搜索工具）
            输入：cmd - 决策指令
            输出：obs - 环境反馈的观察内容，done - 是否完成了指定任务
            done置为True的时机: cmd内含</answer>（即模型给出最终答案），cmd格式违规（即模型决策给出非法内容）
        """
        if self._turns >= self._max_turns:
            self._final_answer = ''
            return '', True

        # 1. 校验并解析指令
        action_type, extract_cmd = self._valid_and_parse_cmd(cmd)

        # 2. 根据指令，执行动作
        obs, done = '', False
        if action_type == WebSearchActionType.INVALID:
            # 非法指令，直接返回
            obs, done = '', True
            self._final_answer = ''
        elif action_type == WebSearchActionType.FINAL_ANSWER:
            # 模型给出最终答案，直接返回
            obs, done = '', True
            self._final_answer = cmd
        else:
            # 执行搜索命令
            raw_results = self._search_tool.search(query=extract_cmd)
            obs, done = self._format_for_llm(raw_results), False
            self._final_answer = cmd

        # 3. 环境内部状态更新
        self._turns += 1

        return obs, done

    def get_final_answer(self) -> str:
        return self._final_answer
