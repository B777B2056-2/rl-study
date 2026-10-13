import re
import json
from enum import Enum, auto
from decimal import Decimal, InvalidOperation
from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple, Callable

from src.infra.dataset.dataset import BaseSample
from src.infra.reward.reward import AgentTraceType, RewardCalcContext, RewardFn
from src.tool_rl.dataset import ToolRLSample


# ==== 模型输出文本的格式检查状态机 ====
class _FormatState(Enum):
    START = auto()
    SEEN_THINK = auto()
    SEEN_TOOL_CALL = auto()
    SEEN_RESPONSE = auto()
    INVALID = auto()


# 需要识别的关键标记
_KEY_TOKENS = ("<think>", "<tool_call>", "<response>")


def _iter_key_tokens(content: str):
    i = 0
    n = len(content)
    while i < n:
        best: Optional[tuple[int, str]] = None
        for tok in _KEY_TOKENS:
            pos = content.find(tok, i)
            if pos != -1 and (best is None or pos < best[0]):
                best = (pos, tok)
        if best is None:
            return
        pos, tok = best
        yield tok
        i = pos + len(tok)


def _check_format(content: str) -> bool:
    """用状态机检查单个模型输出的格式。"""
    state = _FormatState.START

    for tok in _iter_key_tokens(content):
        if state is _FormatState.START:
            if tok == "<think>":
                state = _FormatState.SEEN_THINK
            else:
                return False

        elif state is _FormatState.SEEN_THINK:
            if tok == "<tool_call>":
                state = _FormatState.SEEN_TOOL_CALL
            elif tok == "<response>":
                state = _FormatState.SEEN_RESPONSE
            else:
                return False

        elif state is _FormatState.SEEN_TOOL_CALL:
            if tok == "<response>":
                state = _FormatState.SEEN_RESPONSE
            else:
                return False

        elif state is _FormatState.SEEN_RESPONSE:
            # 终止状态，后面再出现关键标记都算非法
            return False

        else:
            return False

    # 接受条件：必须至少到 SEEN_TOOL_CALL 或 SEEN_RESPONSE
    return state in (_FormatState.SEEN_TOOL_CALL, _FormatState.SEEN_RESPONSE)


# ==== 工具调用文本解析 ====

def _parse_json_objects(raw: str) -> List[Dict[str, Any]]:
    """解析一段文本里的所有 JSON 对象。"""
    raw = raw.strip()
    if not raw:
        return []

    # 整体解析
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # 逐个提取平衡括号的 JSON 对象
    calls = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(raw):
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
                cand = raw[start:i + 1]
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict):
                        calls.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return calls

def _extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """从文本中提取所有 <tool_call> 内的工具调用。"""
    calls = []
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.S):
        calls.extend(_parse_json_objects(m.group(1)))
    # 只保留有 name 字段的调用
    return [c for c in calls if "name" in c]

def _collect_gt_calls(sample: ToolRLSample) -> List[Dict[str, Any]]:
    """从样本 outputs 中提取 ground truth 工具调用。"""
    return _extract_tool_calls(sample.outputs or "")

def _collect_pred_calls(ctx: RewardCalcContext) -> List[Dict[str, Any]]:
    """从轨迹的 ACTION 项中提取预测工具调用。"""
    calls = []
    for item in ctx.agent_full_trace:
        if item.a_type == AgentTraceType.ACTION:
            calls.extend(_extract_tool_calls(item.content or ""))
    return calls

def _optimal_pairing(
    pred_calls: List[Dict[str, Any]],
    gt_calls: List[Dict[str, Any]],
    key_fn: Callable[[Dict[str, Any], Dict[str, Any]], float],
) -> List[Tuple[int, int]]:
    """
    找最优配对，使 sum(key_fn(pred_i, gt_j)) 最大。
    返回 [(pred_idx, gt_idx), ...]。
    key_fn 是 (pred, gt) -> float 或 int。
    """
    n_pred = len(pred_calls)
    n_gt = len(gt_calls)
    if n_pred == 0 or n_gt == 0:
        return []

    # 预计算分数矩阵
    score = [[key_fn(pred_calls[i], gt_calls[j])
              for j in range(n_gt)] for i in range(n_pred)]

    best_score = float("-inf")
    best_pairs: List[Tuple[int, int]] = []

    if n_pred >= n_gt:
        # 从 pred 中选 n_gt 个，与 gt 一一对应
        for perm in permutations(range(n_pred), n_gt):
            cur = sum(score[perm[j]][j] for j in range(n_gt))
            if cur > best_score:
                best_score = cur
                best_pairs = [(perm[j], j) for j in range(n_gt)]
    else:
        # 从 gt 中选 n_pred 个，与 pred 一一对应
        for perm in permutations(range(n_gt), n_pred):
            cur = sum(score[i][perm[i]] for i in range(n_pred))
            if cur > best_score:
                best_score = cur
                best_pairs = [(i, perm[i]) for i in range(n_pred)]

    return best_pairs

def _param_name_jaccard(pred: Dict, gt: Dict) -> float:
    """参数名 Jaccard。"""
    pred_keys = set((pred.get("parameters") or {}).keys())
    gt_keys = set((gt.get("parameters") or {}).keys())
    if not pred_keys and not gt_keys:
        return 1.0
    if not pred_keys or not gt_keys:
        return 0.0

    # 共同出现的参数名
    common_keys = pred_keys & gt_keys
    # 预测和真实里出现过的所有参数名
    all_keys = pred_keys | gt_keys

    # 雅可比相似度 = 交集大小 / 并集大小
    return len(common_keys) / len(all_keys)

def _values_equal(a: Any, b: Any) -> bool:
    """
    归一化json value后再比较，因json无严格数字类型，所以需防止模型输出为5，但目标值为字符串5，导致被惩罚

    规则：
        - bool 优先处理（避免 True 被当成 1）
        - int / float / 数字字符串 统一成 Decimal 比较
        - 其他类型 strip 后按字符串比较
    """
    def _norm(v: Any):
        # bool 必须最先判断
        if isinstance(v, bool):
            return "true" if v else "false"
        # 数字统一成 Decimal
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        # 字符串先 strip，再尝试转 Decimal
        if isinstance(v, str):
            s = v.strip()
            try:
                return Decimal(s)
            except InvalidOperation:
                return s
        return v

    na, nb = _norm(a), _norm(b)

    # 两边都是 Decimal 时用 == 精确比较（Decimal 的 == 无浮点误差）
    # 其他情况按类型比较，类型不同直接 False
    if type(na) is not type(nb):
        return False
    return na == nb

def _param_value_matches(pred: Dict, gt: Dict) -> int:
    """参数值精确匹配的个数。"""
    pred_params = pred.get("parameters") or {}
    gt_params = gt.get("parameters") or {}

    return sum(
        1 for k, v in gt_params.items()
        if k in pred_params and _values_equal(pred_params[k], v)
    )


class ToolRLReward(RewardFn):
    """Tool-RL奖励"""

    def __init__(self, total_steps: int) -> None:
        super().__init__()
        self._total_steps=total_steps

    def _format_reward(self, ctx: RewardCalcContext) -> float:
        """
            格式奖励
            检查模型输出是否按正确顺序包含：
                <think>
                <tool_call>
                <response>
            全部符合得 1，否则得 0。
        """
        # 仅对模型输出进行格式校验（即跳过INPUT），且需保证顺序：<think> -> <tool_call> -> <response>
        full_match: bool = True
        has_model_output: bool = False

        for item in ctx.agent_full_trace:
            if item.a_type not in (AgentTraceType.ACTION, AgentTraceType.END):
                continue

            has_model_output = True
            if not _check_format(item.content):
                full_match = False
                break

        if not has_model_output:
            return 0.0

        return 1.0 if full_match else 0.0

    def _tool_names_match_reward(self, gt_calls: List[Dict[str, Any]], pred_calls : List[Dict[str, Any]]) -> float:
        """
        工具名称匹配奖励（雅可比相似度）。

        公式：
            r_name = |N_G ∩ N_P| / |N_G ∪ N_P| ∈ [0, 1]

        符号含义：
            N_G : 真实工具调用的工具名集合，来自 sample.outputs 里的 <tool_call>
            N_P : 预测工具调用的工具名集合，来自 ctx.agent_full_trace 中
                ACTION / END 项里的 <tool_call>

        含义：
            1.0  表示预测工具名集合与真实工具名集合完全一致
            0.0  表示两边没有任何共同工具名
            0.5  表示真实有两个工具、预测命中一个，或预测多出一个无关工具

        对应论文公式 (3.3 节)：
            r_name = |N_G ∩ N_P| / |N_G ∪ N_P|
        """
        gt_names = {c["name"] for c in gt_calls}
        pred_names = {c["name"] for c in pred_calls}

        if not gt_names and not pred_names:
            return 1.0
        if not gt_names or not pred_names:
            return 0.0

        # 共同出现的工具名
        common_names = gt_names & pred_names
        # 真实和预测里出现过的所有工具名
        all_names = gt_names | pred_names

        # 雅可比相似度 = 交集大小 / 并集大小
        return len(common_names) / len(all_names)

    def _tool_arg_names_match_reward(
            self, 
            gt_calls: List[Dict[str, Any]], 
            pred_calls : List[Dict[str, Any]],
            pairs: List[Tuple[int, int]],
        ) -> float:
        """
        工具参数名称匹配奖励。

        公式：
            r_param = Σ_{G_j ∈ G} |keys(P_G) ∩ keys(P_P)| / |keys(P_G) ∪ keys(P_P)|
                    ∈ [0, |G|]

        符号含义：
            G              : 真实工具调用集合，G = {G_1, ..., G_n}
            |G|            : 真实工具调用的个数
            keys(P_G)      : 预测工具调用 P 的参数名集合
            keys(P_P)      : 真实工具调用 G_j 的参数名集合
            Σ_{G_j ∈ G}   : 对每个真实工具调用求和

        含义：
            每个真实工具调用贡献一个 [0, 1] 的参数名 Jaccard 分数，
            总奖励是这些分数的和，范围 [0, |G|]。

            例如：
            - 真实 G_1 参数名为 {city, date}，预测 P_1 参数名为 {city, date}
            → 该项贡献 2/2 = 1.0
            - 真实 G_1 参数名为 {city, date}，预测 P_1 参数名为 {city}
            → 该项贡献 1/2 = 0.5
            - 真实 G_1 参数名为 {city, date}，预测 P_1 参数名为 {query}
            → 该项贡献 0/2 = 0.0

        多工具时：
            用 _optimal_pairing 找预测与真实之间的最优配对，
            使 Σ 参数名 Jaccard 最大，再返回该最优配对下的总分。

        对应论文公式 (3.3 节)：
            r_param = Σ_{G_j ∈ G} |keys(P_G) ∩ keys(P_P)| / |keys(P_G) ∪ keys(P_P)|
        """
        if not gt_calls or not pred_calls:
            return 0.0

        return sum(
            _param_name_jaccard(pred_calls[i], gt_calls[j])
            for i, j in pairs
        )

    def _tool_arg_values_match_reward(
            self, 
            gt_calls: List[Dict[str, Any]], 
            pred_calls : List[Dict[str, Any]],
            pairs: List[Tuple[int, int]],
        ) -> float:
        """
        工具参数取值匹配奖励。

        公式：
            r_value = Σ_{G_j ∈ G} Σ_{k ∈ keys(G_j)}
                    I[P_G[k] = P_P[k]]
                    ∈ [0, Σ_{G_j ∈ G} |keys(G_j)|]

        符号含义：
            G                      : 真实工具调用集合，G = {G_1, ..., G_n}
            keys(G_j)              : 真实工具调用 G_j 的参数名集合
            k                      : 某个参数名
            P_G[k]                 : 预测工具调用中参数 k 的取值
            P_P[k]                 : 真实工具调用中参数 k 的取值
            I[P_G[k] = P_P[k]]     : 指示函数，取值相等为 1，否则为 0
            Σ_{G_j ∈ G}           : 对每个真实工具调用求和
            Σ_{k ∈ keys(G_j)}     : 对每个参数名求和

        含义：
            逐参数比较预测值与真实值，相等得 1 分，不等得 0 分。
            总奖励是“值完全匹配的参数个数”，
            范围 [0, Σ|keys(G_j)|]，即真实所有工具的参数总数。

            例如：
            - 真实 G_1 参数 {city: "Beijing", date: "2026-10-12"}
            预测 P_1 参数 {city: "Beijing", date: "2026-10-12"}
            → r_value = 1 + 1 = 2
            - 预测 {city: "Shanghai", date: "2026-10-12"}
            → r_value = 0 + 1 = 1
            - 预测 {city: "Shanghai", date: "2026-10-13"}
            → r_value = 0 + 0 = 0

        多工具时：
            用 _optimal_pairing 找预测与真实之间的最优配对，
            使 Σ 参数值匹配个数最大，再返回该最优配对下的总和。

        对应论文公式 (3.3 节)：
            r_value = Σ_{G_j ∈ G} Σ_{k ∈ keys(G_j)} I[P_G[k] = P_P[k]]
        """
        if not gt_calls or not pred_calls:
            return 0.0

        return float(sum(
            _param_value_matches(pred_calls[i], gt_calls[j])
            for i, j in pairs
        ))

    def _joint_pair_score(self, pred_call: Dict[str, Any], gt_call: Dict[str, Any]) -> float:
        """配对分数 = 参数名 Jaccard + 参数值匹配个数。"""
        return _param_name_jaccard(pred_call, gt_call) + _param_value_matches(pred_call, gt_call)

    def _correct_reward(self, ctx: RewardCalcContext, sample: ToolRLSample) -> float:
        """工具调用正确性奖励"""
        # 提取工具调用
        gt_calls = _collect_gt_calls(sample)    # ground truth工具调用，即目标工具调用轨迹
        pred_calls = _collect_pred_calls(ctx)   # 模型输出的工具调用轨迹

        if not gt_calls:
            # 真实没有工具调用时，正确性奖励没有意义
            return 0.0

        # 计算总和
        pairs = _optimal_pairing(pred_calls, gt_calls, self._joint_pair_score)  # 参数名和参数值共享同一个最优配对
        sum_tool_rewards = self._tool_names_match_reward(gt_calls=gt_calls, pred_calls=pred_calls) + \
                           self._tool_arg_names_match_reward(gt_calls=gt_calls, pred_calls=pred_calls, pairs=pairs) + \
                           self._tool_arg_values_match_reward(gt_calls=gt_calls, pred_calls=pred_calls, pairs=pairs)

        # 真实工具调用个数
        num_tools = len(gt_calls)

        # 真实所有工具的参数总个数
        total_param_keys = sum(
            len((call.get("parameters") or {}).keys())
            for call in gt_calls
        )

        # 归一化
        # 1. 统一尺度：不同样本工具数/参数数不同，原始匹配分不可比。
        # 2. 固定锚点：完全正确 = +3，完全错误 = -3，部分正确在中间。
        # 3. 权重可控：正确性跨度 6，格式奖励跨度 1，正确性优先。
        # 4. 训练稳定：避免奖励随样本复杂度漂移，GRPO 组内优势更可靠。
        s_max = 1 + num_tools + total_param_keys
        final_reward = 6 * (sum_tool_rewards / s_max) - 3
        return final_reward

    def _scale_rewards_dynamic(
        self,
        r_format: float,      # 原始格式奖励，0 或 1
        r_correct: float,     # 原始正确性奖励，[-3, 3]
        step_current: int,
    ) -> Tuple[float, float]:
        """
            动态细粒度缩放，按目标区间进行映射
            随着训练进行，格式奖励区间宽度↓，正确性奖励区间宽度↑
        """
        p = step_current / self._total_steps  # [0, 1]

        # 按目标区间进行映射：格式奖励目标范围
        L_f = -2 + p
        U_f = 2 - p
        r_format_scaled = L_f + r_format * (U_f - L_f)

        # 按目标区间进行映射：正确性奖励目标范围
        L_c = -2 - p
        U_c = 2 + p
        r_correct_scaled = L_c + (r_correct + 3) * (U_c - L_c) / 6

        return r_format_scaled, r_correct_scaled

    def score(self, ctx: RewardCalcContext, sample: BaseSample) -> float:
        """计算最终奖励"""
        assert isinstance(sample, ToolRLSample)

        # 格式奖励
        reward_format = self._format_reward(ctx)
        # 工具调用正确性奖励
        reward_correct = self._correct_reward(ctx, sample)

        # 平滑动态奖励：使得训练过程中，格式奖励权重逐渐下降，工具调用正确性奖励权重逐渐上升
        reward_format, reward_correct = self._scale_rewards_dynamic(
            r_format=reward_format, 
            r_correct=reward_correct, 
            step_current=ctx.current_step,
        )

        # 总奖励 = 格式奖励 + 工具调用正确性奖励
        return reward_format + reward_correct
