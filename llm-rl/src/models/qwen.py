import os
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .lora import (
    LoraConfig,
    inject_lora,
    merge_lora,
    save_lora,
    load_lora,
)


class Qwen2_5(object):
    """Qwen2.5 模型封装：加载 + 注入 LoRA + 保存 + 加载"""

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B",
                 lora_config: LoraConfig = None, device="cuda"):
        self._model_name = model_name
        self._lora_config = lora_config
        self._device = device

        self._init_tokenizer()
        self._init_model()

    # ============================================================
    # 初始化（新建模型）
    # ============================================================

    def _init_tokenizer(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _init_model(self):
        if self._lora_config is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name, torch_dtype=torch.bfloat16,
            ).to(self._device)
            return

        model = self._load_base(self._lora_config.use_qlora)

        for p in model.parameters():
            p.requires_grad = False
        inject_lora(model, self._lora_config)
        self._model = model

    def _load_base(self, use_qlora: bool):
        """加载 base 模型（普通或 4-bit）"""
        if use_qlora:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                quantization_config=bnb_config,
                device_map=self._device,
            )
            model = prepare_model_for_kbit_training(model)
            return model
        else:
            return AutoModelForCausalLM.from_pretrained(
                self._model_name, torch_dtype=torch.bfloat16,
            ).to(self._device)

    # ============================================================
    # 从保存目录加载（类方法）
    # ============================================================

    @classmethod
    def from_pretrained(cls, save_dir, lora_config: LoraConfig = None,
                        model_name="Qwen/Qwen2.5-0.5B", device="cuda"):
        """
        从 save_dir 加载模型

        自动识别三种结构：
            1. save_dir/merged/  → 合并后的完整模型
            2. save_dir/lora/    → base + LoRA adapter
            3. save_dir/         → 完整模型

        参数:
            save_dir:    保存目录
            lora_config: 加载 LoRA 时必须提供（用于重建 LoRA 结构）
            model_name:  base 模型名
            device:      设备
        """
        obj = cls.__new__(cls)
        obj._model_name = model_name
        obj._lora_config = lora_config
        obj._device = device
        obj._load_from(save_dir)
        return obj

    def _load_from(self, save_dir):
        merged_dir = os.path.join(save_dir, "merged")
        lora_dir = os.path.join(save_dir, "lora")

        # ============ 情况 1：合并后的完整模型 ============
        if os.path.isdir(merged_dir):
            self._load_tokenizer(merged_dir)
            self._model = AutoModelForCausalLM.from_pretrained(
                merged_dir, torch_dtype=torch.bfloat16,
            ).to(self._device)
            self._lora_config = None       # merged 不含 LoRA 结构
            print(f"[Qwen2_5] 已加载合并模型: {merged_dir}")
            return

        # ============ 情况 2：base + LoRA adapter ============
        if os.path.isdir(lora_dir):
            if self._lora_config is None:
                raise ValueError(
                    "加载 LoRA 权重需要提供 lora_config（用于重建 LoRA 结构）"
                )

            self._load_tokenizer(lora_dir)

            # 加载 base（考虑 QLoRA）
            model = self._load_base(self._lora_config.use_qlora)

            # 冻结 + 注入 LoRA 结构
            for p in model.parameters():
                p.requires_grad = False
            inject_lora(model, self._lora_config)

            # 加载 LoRA 参数
            load_lora(model, os.path.join(lora_dir, "lora.pt"))

            self._model = model
            print(f"[Qwen2_5] 已加载 LoRA: {lora_dir}")
            return

        # ============ 情况 3：save_dir 直接是完整模型 ============
        self._load_tokenizer(save_dir)
        self._model = AutoModelForCausalLM.from_pretrained(
            save_dir, torch_dtype=torch.bfloat16,
        ).to(self._device)
        self._lora_config = None
        print(f"[Qwen2_5] 已加载完整模型: {save_dir}")

    def _load_tokenizer(self, path):
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    # ============================================================
    # 访问器
    # ============================================================

    def model(self):
        return self._model

    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self):
        return self._device

    @property
    def model_name(self):
        return self._model_name

    @property
    def has_lora(self):
        return self._lora_config is not None

    @property
    def is_qlora(self):
        return self.has_lora and self._lora_config.use_qlora

    # ============================================================
    # 保存
    # ============================================================

    def save(self, save_dir: str):
        """
        保存模型

        - 无 LoRA：保存完整模型
        - 普通 LoRA：保存 lora/ + merged/
        - QLoRA：只保存 lora/
        """
        os.makedirs(save_dir, exist_ok=True)

        # 无 LoRA：直接保存
        if not self.has_lora:
            self._model.save_pretrained(save_dir)
            self._tokenizer.save_pretrained(save_dir)
            print(f"[Qwen2_5] 完整模型已保存到 {save_dir}")
            return

        # 保存 LoRA adapter
        lora_dir = os.path.join(save_dir, "lora")
        os.makedirs(lora_dir, exist_ok=True)
        save_lora(self._model, os.path.join(lora_dir, "lora.pt"))
        self._tokenizer.save_pretrained(lora_dir)
        print(f"[Qwen2_5] LoRA adapter 已保存到 {lora_dir}")

        # QLoRA 不 merge
        if self.is_qlora:
            print("[Qwen2_5] QLoRA：跳过 merge（4-bit base 不能合并）")
            return

        # 普通 LoRA：合并
        merged_dir = os.path.join(save_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)

        merged = copy.deepcopy(self._model).cpu()
        merge_lora(merged)
        merged.save_pretrained(merged_dir)
        self._tokenizer.save_pretrained(merged_dir)
        print(f"[Qwen2_5] 合并模型已保存到 {merged_dir}")

        del merged
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # 1. 加载模型
    qwen = Qwen2_5.from_pretrained(
        save_dir="./outputs/gsm8k-sft-qwen2_5-0_5B",
        lora_config=LoraConfig(
            alpha=16,
            rank=8,
            dropout=0.0,                      # 推理时 dropout=0
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            use_qlora=False,                  # bf16 加载 base（QLoRA 保存的也能这样加载）
        ),
    )

    model = qwen.model()
    tokenizer = qwen.tokenizer()
    device = qwen.device

    # 2. 推理
    questions = [
        "小明有5个苹果，吃了2个，还剩几个？",
        "一个班级有30个学生，其中60%是女生，男生有多少人？",
        "一辆车以每小时60公里的速度行驶了2.5小时，行驶了多少公里？",
    ]

    model.eval()

    for question in questions:
        # a. 构造 messages
        messages = [{"role": "user", "content": question}]

        # b. 应用 chat 模板，末尾加 assistant 开头
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # c. 编码
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # d. 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,              # 贪心解码
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # e. 只解码新生成的部分
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print("=" * 60)
        print(f"问题: {question}")
        print(f"回答: {response}")
        print()
