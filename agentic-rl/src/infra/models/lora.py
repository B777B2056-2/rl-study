from dataclasses import dataclass, asdict
import json
from typing import Tuple
import torch
from contextlib import contextmanager


@dataclass
class LoraConfig:
    alpha: float
    rank: int
    target_modules: Tuple[str, ...]
    use_qlora: bool = False
    dropout: float = 0.0

    def save(self, path: str):
        """保存到 JSON"""
        data = asdict(self)
        data["target_modules"] = list(data["target_modules"])   # tuple → list
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "LoraConfig":
        """从 JSON 加载"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["target_modules"] = tuple(data["target_modules"])   # list → tuple
        return cls(**data)

class LoraAdaptLinear(torch.nn.Module):
    """Lora适配层"""
    def __init__(self, original_linear, config: LoraConfig):
        super().__init__()
        self._enabled = True

        # 缩放因子，scaling = alpha / rank
        self._scaling = config.alpha / config.rank

        # 继承原模块的类型和设备
        dtype = original_linear.weight.dtype
        if not dtype.is_floating_point:
            dtype = torch.bfloat16              # QLoRA 兜底
        device = original_linear.weight.device

        """
            B初始化为全0，A以随机数（必须比较小的数）进行初始化
            维度计算：
                i = original_linear.in_features, o = original_linear.out_features
                设x为(n, i), self._original(x) = x @ w.T, 则w维度为(o, i)，w.T为(i, o)
                则lora层参数B @ A后也应为(i, o)；因此，B为(i, r)，A为(r, o)
        """
        self._B = torch.nn.Parameter(torch.zeros(original_linear.in_features, config.rank, dtype=dtype, device=device)) # i, r
        self._A = torch.nn.Parameter(torch.randn(config.rank, original_linear.out_features, dtype=dtype, device=device) * 0.01) # r, o

        # 防止过拟合小数据集（torch.nn.Identity()：直接透传数据，无任何变化）
        self._dropout = torch.nn.Dropout(config.dropout) if config.dropout > 0 else torch.nn.Identity()

        # 冻结原始权重
        self._original = original_linear
        for p in self._original.parameters():
            p.requires_grad = False

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    def forward(self, x):
        """
            原公式：y = Wx + scaling * B @ A @ dropout(x)
            由于self._original(x)实际为x @ w.T，因此公式变为：
            y = x @ w.T + scaling * dropout(x) @ B @ A
        """
        y_original = self._original(x)
        if not self._enabled:
            return y_original

        x_dropped = self._dropout(x).to(self._B.dtype)
        y_lora = self._scaling * (x_dropped @ self._B @ self._A)
        return y_original + y_lora

    def merge_weights(self):
        """
            把 LoRA 合并到原始权重：W_new = W + scaling * B @ A
            由于pytorch内self._original(x)实际为x @ w.T，因此合并权重时，也改为：
            W_new = W + scaling * (B @ A).T
        """
        with torch.no_grad():
            delta = self._scaling * (self._B @ self._A).T
            self._original.weight.data += delta
        return self._original

def inject_lora(model, config: LoraConfig):
    """把匹配的 Linear 替换成 LoRALinear"""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            inject_lora(module, config)
        elif isinstance(module, torch.nn.Linear) and any(t in name for t in config.target_modules):
            setattr(model, name, LoraAdaptLinear(module, config))

def merge_lora(model):
    """把所有 LoRALinear 合并回原始 Linear"""
    for name, module in model.named_modules():
        if isinstance(module, LoraAdaptLinear):
            merged = module.merge_weights()
            # 替换回父模块
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, merged)

def merge_qlora_to_bf16(model_name, qlora_model, lora_config):
    """
    把 QLoRA 训练的 LoRA 合并到 bf16 base

    参数:
        model_name:    base 模型名，如 "Qwen/Qwen2.5-0.5B"
        qlora_model:   QLoRA 训练后的模型（base 是 4-bit）
        lora_config:   LoRA 配置

    返回:
        merged_model:  合并后的 bf16 模型（在 CPU 上）
    """
    import torch
    from transformers import AutoModelForCausalLM

    # 1. 加载 bf16 base（不用 4-bit）
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )

    # 2. 冻结 + 注入 LoRA 结构
    for p in base_model.parameters():
        p.requires_grad = False
    inject_lora(base_model, lora_config)

    # 3. 从 QLoRA 模型里拷贝 LoRA 参数
    qlora_lora_state = {
        n: p.data.cpu()
        for n, p in qlora_model.named_parameters()
        if p.requires_grad
    }
    missing, unexpected = base_model.load_state_dict(
        qlora_lora_state, strict=False
    )
    if missing:
        print(f"[merge_qlora_to_bf16] 警告：缺少 {len(missing)} 个参数")
    if unexpected:
        print(f"[merge_qlora_to_bf16] 警告：多余 {len(unexpected)} 个参数")

    # 4. 合并 LoRA 到 bf16 base
    merge_lora(base_model)

    # 5. 搬到 CPU 返回
    base_model = base_model.cpu()

    return base_model

def save_lora(model, path):
    """只保存 LoRA 参数"""
    state = {n: p for n, p in model.named_parameters() if p.requires_grad}
    torch.save(state, path)

def load_lora(model, path):
    """加载 LoRA 参数"""
    state = torch.load(path)
    model.load_state_dict(state, strict=False)

@contextmanager
def disabled_lora(model):
    """临时禁用模型中所有 LoRA 层，退出代码块时恢复"""
    lora_modules = [
        m for m in model.modules() if isinstance(m, LoraAdaptLinear)
    ]
    for m in lora_modules:
        m.set_enabled(False)
    try:
        yield
    finally:
        for m in lora_modules:
            m.set_enabled(True)
