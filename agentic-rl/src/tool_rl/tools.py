import re
import json
from dataclasses import dataclass, field
from typing import List, Any, Dict, Type

from src.infra.environment.tools import Action, Tool, ToolCall, ToolResult, ToolRouter
from src.infra.log import get_logger


logger = get_logger(__name__)


@dataclass
class ToolDef:
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


class DatasetTool(Tool):
    """
    通用工具类，适配 ToolRL 数据集里的任意工具。
    """

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters  # 参数 schema，比如 {"city": "string", "date": "string"}

    def parse_args(self, raw_text: str) -> Dict[str, Any]:
        """
        raw_text 是 Router 传来的工具调用片段，通常是 JSON 字符串。
        例如：'{"name": "get_weather", "parameters": {"city": "Beijing"}}'
        """
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # 如果 Router 只传了 parameters 部分，也尝试解析
            try:
                data = {"parameters": json.loads(raw_text)}
            except json.JSONDecodeError:
                raise RuntimeError(f"无法解析工具调用 JSON: {raw_text}")

        args = data.get("parameters", {})
        if not isinstance(args, dict):
            raise RuntimeError("parameters 必须是字典")

        # 可选：根据 self.parameters 做基本校验和类型转换
        for key, expected_type in self.parameters.items():
            if key not in args:
                # 如果参数缺失，可以给默认值或报错，取决于你的策略
                continue
            # 简单类型检查（按需扩展）
            if expected_type == "string" and not isinstance(args[key], str):
                args[key] = str(args[key])
            elif expected_type == "int" and not isinstance(args[key], int):
                try:
                    args[key] = int(args[key])
                except ValueError:
                    raise RuntimeError(f"参数 {key} 需要 int，得到 {args[key]}")

        return args

    def run(self, args: Dict[str, Any]) -> ToolResult:
        """
        在 RL 训练中，通常不需要真正执行工具。
        因为奖励是基于 ground truth 工具调用计算的，不是基于执行结果。
        这里返回一个模拟结果即可。
        如果要做评估，可以在这里接真实 API。
        """
        # 模拟执行：把参数拼成一段文本作为 observation
        content = f"工具 {self.name} 已执行，参数：{json.dumps(args, ensure_ascii=False)}"
        return ToolResult(
            success=True,
            content=content,
            metadata={"tool_name": self.name, "args": args}
        )


class ToolRLRouter(ToolRouter):
    """
    ToolRL 数据集专用路由。

    识别格式：
        <tool_call>
        {"name": "...", "parameters": {...}}
        {"name": "...", "parameters": {...}}
        </tool_call>

    也兼容：
        - 单个 JSON 对象
        - JSON 数组
        - 多个并列 JSON 对象

    终止判断：
        - 出现 <response> 且没有 <tool_call>，视为终止
    """

    # ------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------
    def route(self, text: str) -> Action:
        # 1. 终止动作：只有 <response>，没有 <tool_call>
        if "<response>" in text and "<tool_call>" not in text:
            return Action(is_done=True, calls=[])

        # 2. 提取 <tool_call> 内容
        m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.S)
        if not m:
            # 没有任何工具调用，也没有明确的 <response>
            # 按数据集语义，视为直接响应，终止
            return Action(is_done=True, calls=[])

        raw_json = m.group(1).strip()
        calls_data = self._parse_calls(raw_json)

        if not calls_data:
            return Action(is_done=True, calls=[])

        # 3. 构造 ToolCall 列表
        #    每个 ToolCall 的 raw_text 只包含"它自己那一段" JSON，
        #    避免把混合文本传给单个 Tool.parse_args
        tool_calls: List[ToolCall] = []
        for data in calls_data:
            name = data.get("name")
            if not name:
                continue
            tool_calls.append(ToolCall(
                kind=name,
                raw_text=json.dumps(data, ensure_ascii=False),
            ))

        if not tool_calls:
            return Action(is_done=True, calls=[])

        return Action(is_done=False, calls=tool_calls)

    # ------------------------------------------------------------
    # 单个 ToolResult → observation 文本
    # ------------------------------------------------------------
    def format_observation_for_single_result(self, result: ToolResult) -> str:
        if result.success:
            return f"<obs>{result.content}</obs>"
        return f"<obs>工具执行失败：{result.error}</obs>"

    # ------------------------------------------------------------
    # 多个 ToolResult → observation 文本
    # ------------------------------------------------------------
    def format_observation_for_results(self, results: List[ToolResult]) -> str:
        """
        并行多工具时，把多个结果合并到一个 <obs> 里。
        这样在语义上明确表示：这些观察结果属于同一步。
        """
        lines = []
        for i, result in enumerate(results, start=1):
            if result.success:
                lines.append(f"[{i}] {result.content}")
            else:
                lines.append(f"[{i}] 失败：{result.error}")

        body = "\n".join(lines)
        return f"<obs>\n{body}\n</obs>"

    # ------------------------------------------------------------
    # 内部工具：解析 <tool_call> 里的 JSON
    # ------------------------------------------------------------
    def _parse_calls(self, raw_json: str) -> List[dict]:
        """
        兼容三种情况：
            1. 单个 JSON 对象：   {"name": ..., "parameters": {...}}
            2. JSON 数组：        [{"name": ...}, {"name": ...}]
            3. 多个并列对象：      {"name": ...}\n{"name": ...}
        """
        raw_json = raw_json.strip()

        # 情况 1 / 2：整体可以解析
        try:
            data = json.loads(raw_json)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
            if isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            pass

        # 情况 3：逐个提取平衡括号的 JSON 对象
        calls = []
        depth = 0
        start = None
        in_string = False
        escape = False

        for i, ch in enumerate(raw_json):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    cand = raw_json[start:i + 1]
                    try:
                        obj = json.loads(cand)
                        if isinstance(obj, dict):
                            calls.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = None

        return calls


class ToolRegistryBuilder:
    """
    从数据集的所有 instruction 中提取工具定义，去重，创建 DatasetTool，并注册到 Router。
    """

    def __init__(self):
        self.tools_by_name: Dict[str, ToolDef] = {}

    def add_instruction(self, instruction: str):
        """从单条 instruction 中提取工具定义并加入注册表。"""
        for td in self._extract_tools_from_instruction(instruction):
            existing = self.tools_by_name.get(td.name)

            # 首次出现，直接注册
            if existing is None:
                self.tools_by_name[td.name] = td
                continue

            # 已存在同名工具，比较参数名集合
            existing_keys = set(existing.parameters.keys())
            new_keys = set(td.parameters.keys())

            # 没有新增参数，跳过
            if not (new_keys - existing_keys):
                continue

            # 有新增参数，只补充缺失的，不覆盖已有定义
            for key in new_keys - existing_keys:
                existing.parameters[key] = td.parameters[key]

            # 可选：如果同名参数类型不同，记录冲突
            for key in existing_keys & new_keys:
                if existing.parameters[key] != td.parameters[key]:
                    # 这里可以打日志、存冲突列表，或按策略处理
                    pass

    def add_tools_from_dataset(self, dataset):
        """遍历整个数据集，提取所有 instruction 中的工具定义。"""
        for row in dataset:
            instruction = row.get("instruction", "")
            if instruction:
                self.add_instruction(instruction)

    def build_tools(self) -> List[DatasetTool]:
        """根据注册表创建所有 DatasetTool 实例。"""
        return [
            DatasetTool(
                name=td.name,
                description=td.description,
                parameters=td.parameters
            )
            for td in self.tools_by_name.values()
        ]

    def build_router(self) -> ToolRLRouter:
        """创建 Router，并把所有工具注册进去。"""
        tools = self.build_tools()
        return ToolRLRouter(tools)

    # ------------------------------------------------------------
    # 从 instruction 中提取工具定义
    # ------------------------------------------------------------
    def _extract_tools_from_instruction(self, instruction: str) -> List[ToolDef]:
        # 策略 1：尝试解析 JSON 工具定义
        tools = self._try_json(instruction)
        if tools:
            return tools

        # 策略 2：文本解析
        return self._parse_text(instruction)

    def _try_json(self, instruction: str) -> List[ToolDef]:
        """尝试从 instruction 中提取 JSON 格式的工具定义。"""
        # 找 JSON 数组
        for cand in re.findall(r'\[[\s\S]*?\]', instruction):
            try:
                data = json.loads(cand)
                if isinstance(data, list) and all(
                    isinstance(x, dict) and "name" in x for x in data
                ):
                    return [
                        ToolDef(
                            name=x["name"],
                            description=x.get("description", ""),
                            parameters=x.get("parameters", {})
                        )
                        for x in data
                    ]
            except Exception:
                pass

        # 找单个 JSON 对象（可能是 OpenAI function schema）
        for cand in re.findall(r'\{[\s\S]*?\}', instruction):
            try:
                data = json.loads(cand)
                if isinstance(data, dict):
                    # 情况 A：直接是工具定义
                    if "name" in data:
                        return [
                            ToolDef(
                                name=data["name"],
                                description=data.get("description", ""),
                                parameters=data.get("parameters", {})
                            )
                        ]
                    # 情况 B：OpenAI function 格式
                    if "function" in data and isinstance(data["function"], dict):
                        fn = data["function"]
                        if "name" in fn:
                            return [
                                ToolDef(
                                    name=fn["name"],
                                    description=fn.get("description", ""),
                                    parameters=fn.get("parameters", {})
                                )
                            ]
            except Exception:
                pass

        return []

    def _parse_text(self, instruction: str) -> List[ToolDef]:
        """
        文本解析示例。
        假设格式类似：
            1. get_weather: 查询天气。
               parameters:
                 - city (string): 城市名
                 - date (string): 日期
        """
        tools = []
        # 按 "数字. 工具名:" 切分
        pattern = re.compile(
            r'^\s*(\d+)\.\s*([A-Za-z_][\w]*)\s*:\s*(.*?)(?=^\s*\d+\.\s*[A-Za-z_]|\Z)',
            re.M | re.S
        )
        for m in pattern.finditer(instruction):
            name = m.group(2)
            body = m.group(3)

            # 提取描述
            if "parameters:" in body:
                desc = body.split("parameters:")[0].strip()
            else:
                desc = body.strip()

            # 提取参数
            params = {}
            param_section = re.search(r'parameters:\s*(.*)', body, re.S)
            if param_section:
                for line in param_section.group(1).splitlines():
                    line = line.strip()
                    if line.startswith("-"):
                        pm = re.match(r'-\s*(\w+)\s*\((\w+)\)', line)
                        if pm:
                            params[pm.group(1)] = pm.group(2)

            tools.append(ToolDef(name=name, description=desc, parameters=params))

        return tools
