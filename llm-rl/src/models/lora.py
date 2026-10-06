from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass
class LoraConfig:
    alpha: float
    rank: float
    target_modules: Tuple[str]
    use_qlora: bool = False
    dropout: float = 0.0

class LoraAdaptLinear(torch.nn.Module):
    """Lora适配层"""
    def __init__(self, original_linear, config: LoraConfig):
        super().__init__()

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

    def forward(self, x):
        """
            原公式：y = Wx + scaling * B @ A @ dropout(x)
            由于self._original(x)实际为x @ w.T，因此公式变为：
            y = x @ w.T + scaling * dropout(x) @ B @ A
        """
        y_original = self._original(x)
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

def save_lora(model, path):
    """只保存 LoRA 参数"""
    state = {n: p for n, p in model.named_parameters() if p.requires_grad}
    torch.save(state, path)

def load_lora(model, path):
    """加载 LoRA 参数"""
    state = torch.load(path)
    model.load_state_dict(state, strict=False)


if __name__ == "__main__":
    """
    LoRA 单元测试
    运行: uv run python -m src.models.lora
    """
    import torch.nn as nn

    print("=" * 60)
    print("LoRA 单元测试")
    print("=" * 60)

    # ============================================================
    # 测试 1：前向形状
    # ============================================================
    print("\n[测试 1] 前向形状")
    linear = nn.Linear(8, 16)
    lora = LoraAdaptLinear(linear, LoraConfig(alpha=16, rank=4, target_modules=("q_proj",)))

    x = torch.randn(4, 8)
    y = lora(x)
    print(f"  输入: {tuple(x.shape)}, 输出: {tuple(y.shape)}")
    assert y.shape == (4, 16), f"期望 (4, 16)，实际 {y.shape}"
    print("  ✅ 通过")

    # ============================================================
    # 测试 2：初始时不改变模型（B=0）
    # ============================================================
    print("\n[测试 2] 初始状态不改变模型")
    linear = nn.Linear(8, 16)
    x = torch.randn(4, 8)
    y_original = linear(x)

    lora = LoraAdaptLinear(linear, LoraConfig(alpha=16, rank=4, target_modules=("q_proj",)))
    y_lora = lora(x)

    diff = (y_original - y_lora).abs().max().item()
    print(f"  最大差异: {diff:.8f}")
    assert diff < 1e-6, f"初始时应完全一致，实际差异 {diff}"
    print("  ✅ 通过（B 初始为 0，LoRA 分支无贡献）")

    # ============================================================
    # 测试 3：合并前后输出一致
    # ============================================================
    print("\n[测试 3] 合并前后一致")
    torch.manual_seed(42)
    linear = nn.Linear(8, 16)
    lora = LoraAdaptLinear(linear, LoraConfig(alpha=16, rank=4, target_modules=("q_proj",)))

    # 把 A、B 设成非零（否则合并前后都是 0）
    with torch.no_grad():
        lora._A.normal_(0, 0.1)
        lora._B.normal_(0, 0.1)

    x = torch.randn(4, 8)
    y1 = lora(x)                            # 合并前

    merged = lora.merge_weights()           # 合并
    y2 = merged(x)                          # 合并后

    diff = (y1 - y2).abs().max().item()
    print(f"  合并前后差异: {diff:.8f}")
    print(f"  merged.weight 形状: {tuple(merged.weight.shape)}")
    assert diff < 1e-5, f"合并前后应一致，实际差异 {diff}"
    assert merged.weight.shape == (16, 8)
    print("  ✅ 通过")

    # ============================================================
    # 测试 4：合并公式验证
    # ============================================================
    print("\n[测试 4] 合并公式验证")
    torch.manual_seed(42)
    linear = nn.Linear(8, 16)
    w_before = linear.weight.data.clone()

    lora = LoraAdaptLinear(linear, LoraConfig(alpha=16, rank=4, target_modules=("q_proj",)))
    with torch.no_grad():
        lora._A.normal_(0, 0.1)
        lora._B.normal_(0, 0.1)

    # 手动算 delta = scaling * (B @ A).T
    expected_delta = lora._scaling * (lora._B @ lora._A).T

    merged = lora.merge_weights()
    actual_delta = merged.weight.data - w_before

    diff = (expected_delta - actual_delta).abs().max().item()
    print(f"  手动 delta vs 实际 delta: {diff:.8f}")
    assert diff < 1e-6
    print("  ✅ 通过")

    # ============================================================
    # 测试 5：梯度流
    # ============================================================
    print("\n[测试 5] 梯度流")
    linear = nn.Linear(8, 16)
    lora = LoraAdaptLinear(linear, LoraConfig(alpha=16, rank=4, target_modules=("q_proj",)))

    x = torch.randn(4, 8)
    loss = lora(x).sum()
    loss.backward()

    print(f"  原始 weight.grad: {lora._original.weight.grad}")
    print(f"  A.grad: {'有' if lora._A.grad is not None else 'None'}")
    print(f"  B.grad: {'有' if lora._B.grad is not None else 'None'}")

    assert lora._original.weight.grad is None or lora._original.weight.grad.abs().sum() == 0
    assert lora._A.grad is not None
    assert lora._B.grad is not None
    print("  ✅ 通过（原权重冻结，只有 A、B 有梯度）")

    # ============================================================
    # 测试 6：注入到模型
    # ============================================================
    print("\n[测试 6] 注入模型")
    model = nn.ModuleDict({
        "q_proj": nn.Linear(8, 16),
        "v_proj": nn.Linear(8, 16),
        "mlp": nn.Linear(16, 4),
    })

    # 先冻结整个模型（模拟实际加载预训练模型的场景）
    for p in model.parameters():
        p.requires_grad = False

    # 注入 LoRA
    inject_lora(model, LoraConfig(alpha=16, rank=4, target_modules=("q_proj", "v_proj")))

    assert isinstance(model["q_proj"], LoraAdaptLinear)
    assert isinstance(model["v_proj"], LoraAdaptLinear)
    assert isinstance(model["mlp"], nn.Linear)

    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  可训练参数:")
    for n, num in trainable:
        print(f"    {n}: {num}")
    print(f"  总参数: {total}, 可训练: {trainable_num} ({trainable_num/total:.2%})")

    assert all("_A" in n or "_B" in n for n, _ in trainable), "只有 A、B 应可训练"
    print("  ✅ 通过")

    # ============================================================
    # 测试 7：模型级合并
    # ============================================================
    print("\n[测试 7] 模型级合并")
    torch.manual_seed(42)
    model = nn.ModuleDict({
        "q_proj": nn.Linear(8, 16),
    })

    x = torch.randn(4, 8)
    inject_lora(model, LoraConfig(alpha=16, rank=4, target_modules=("q_proj",)))

    # 设非零 A、B
    with torch.no_grad():
        model["q_proj"]._A.normal_(0, 0.1)
        model["q_proj"]._B.normal_(0, 0.1)

    y_with_lora = model["q_proj"](x)

    merge_lora(model)
    assert isinstance(model["q_proj"], nn.Linear), "合并后应该是普通 Linear"

    y_merged = model["q_proj"](x)

    diff = (y_with_lora - y_merged).abs().max().item()
    print(f"  合并前后差异: {diff:.8f}")
    assert diff < 1e-5
    print("  ✅ 通过")

    # ============================================================
    print("\n" + "=" * 60)
    print("✅ 全部 7 项测试通过")
    print("=" * 60)
