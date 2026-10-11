# ============================================================
# 【契约层 Contract Layer】
# 跨层传递的数据结构，被所有层引用。
# ============================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    """
    【契约层】工具执行结果。
    
    流向：Tool.run() 产出 → Environment 消费 → reward_fn 读 metadata
    """
    success: bool                                           # 执行是否成功
    content: str                                            # 主输出（给模型看的裸内容）
    error: Optional[str] = None                             # 失败原因
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加信息（给框架用，不给模型）


@dataclass
class Action:
    """
    【契约层】从 LLM 输出解析出的动作。
    
    流向：ToolRouter.route() 产出 → Environment 消费
    """
    kind: str           # 工具名
    raw_text: str = ""  # 解析来源的原始文本（调试用）


# ============================================================
# 【工具层 Tool Layer】
# 每个工具一个类，负责：参数解析 + 执行 + 格式化 observation。
# 不关心如何被路由到，不关心 episode 状态。
# ============================================================

class Tool(ABC):
    """
    【工具层】工具抽象基类。
    
    职责：被调用时执行 + 格式化 observation。
    不负责：如何被识别、如何被触发（那是 Router 的事）。
    
    注意：Tool 不含 tag / trigger 属性。触发方式是 Router 的配置，
          同一个 Tool 可以被不同 Router 以不同方式触发。
    """
    
    name: str   # 内部唯一标识，用于字典查找、日志、配置
    
    @abstractmethod
    def parse_args(self, raw_text: str) -> Dict[str, Any]:
        """把裸文本解析成结构化参数。失败时抛 ParseError。"""
        ...
    
    @abstractmethod
    def run(self, args: Dict[str, Any]) -> ToolResult:
        """执行工具，返回结构化结果。"""
        ...
    
    @abstractmethod
    def format_observation(self, result: ToolResult) -> str:
        """把 ToolResult 包成 observation 文本。"""
        ...


# ============================================================
# 【路由层 Routing Layer】
# 负责：从 LLM 输出文本中识别动作，路由到对应工具。
# 不负责：执行工具、参数校验、维护状态。
# ============================================================

class ToolRouter(ABC):
    """
    【路由层】路由策略抽象基类。
    
    职责：文本 → Action。只做识别和提取，不执行任何工具。
    
    可替换实现：
        - TagRouter: <search>query</search>
        - JSONRouter: {"tool": "search", "args": {...}}
        - FunctionCallRouter: OpenAI 风格 tool_call
        - HybridRouter: 多策略按优先级组合
    """
    
    def __init__(self, tools: List[Tool]):
        super().__init__()
        self._tools_by_name: Dict[str, Tool] = {t.name: t for t in tools}
    
    @abstractmethod
    def route(self, text: str) -> Action:
        """解析 LLM 输出，返回结构化 Action。"""
        ...

    def call_tool_by_action(self, action: Action) -> str:
        tool = self._tools_by_name[action.kind]
        args = tool.parse_args(action.raw_text)
        result = tool.run(args)
        return tool.format_observation(result)
