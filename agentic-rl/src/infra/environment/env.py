from typing import Tuple, List
from src.infra.environment.tools import Tool, ToolRouter


class Environment(object):
    """环境类"""
    def __init__(self, max_turns: int, tool_router: ToolRouter):
        self._max_turns = max_turns            # 最大轮次
        self._turns = 0                        # 已执行 action 次数
        self._register_tools: List[Tool] = []  # 已注册的工具调用
        self._tool_router = tool_router        # 工具路由，根据模型输出的cmd来决定调用哪个工具

    def register_tool(self, tool: Tool) -> None:
        """注册一个工具"""
        self._register_tools.append(tool)

    def reset(self):
        """重置环境"""
        self._turns = 0

    def action(self, cmd: str) -> Tuple[str, bool]:
        """
            根据决策（即prompt），执行一次动作（比如调用搜索工具）
            输入：cmd - 决策指令
            输出：obs - 环境反馈的观察内容，done - 是否完成了指定任务
            done置为True的时机: 由传入的done_action_kinds配置决定
        """
        # 0. 超过最大迭代轮次，强行终止
        if self._turns >= self._max_turns:
            return "", True

        # 1. 校验并解析指令
        action = self._tool_router.route(cmd)
        self._turns += 1

        # 2. 检查模型决策是否为终止回合
        if action.is_done:
            return "", True

        # 3. 未到终止回合，继续工具调用
        return self._tool_router.call_tools_by_action(action), False
