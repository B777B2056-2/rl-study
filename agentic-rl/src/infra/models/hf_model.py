import os
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from src.infra.models.lora import LoraConfig, inject_lora, load_lora, merge_lora, save_lora, merge_qlora_to_bf16
from src.infra.log import get_logger

logger = get_logger(__name__)


class HFModel:
    """HuggingFace CausalLM 封装"""
    
    DEFAULT_DTYPE = torch.bfloat16
    
    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = DEFAULT_DTYPE,
        lora_config: Optional[LoraConfig] = None,
        device: str = "cuda",
        tokenizer_kwargs: dict = {},
        model_kwargs: dict = {},
        use_qlora: bool = False,
    ):
        self._model_name = model_name
        self._lora_config = lora_config
        self._device = device
        self._dtype = dtype or self.DEFAULT_DTYPE
        self._tokenizer_kwargs = tokenizer_kwargs
        self._model_kwargs = model_kwargs
        self._use_qlora = use_qlora
        
        self._init_tokenizer()
        self._init_model()
    
    # ============================================================
    # 初始化
    # ============================================================
    
    def _init_tokenizer(self):
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **self._tokenizer_kwargs
        )
        self._setup_tokenizer()
    
    def _setup_tokenizer(self):
        """默认：pad_token 缺失时用 eos_token。子类可覆盖适配特殊模型。"""
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
    
    def _init_model(self):
        if self._lora_config is None:
            self._model = self._load_base(use_qlora=self._use_qlora)
            self._model.gradient_checkpointing_enable()
            return
        model = self._load_base(self._lora_config.use_qlora)
        for p in model.parameters():
            p.requires_grad = False
        inject_lora(model, self._lora_config)
        # 开启梯度检查点
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        self._model = model
    
    def _load_base(self, use_qlora: bool):
        if use_qlora:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training
            
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self._dtype,
                bnb_4bit_use_double_quant=True,
            )
            model = self._from_pretrained_with_fallback(
                quantization_config=bnb, device_map=self._device
            )
            return prepare_model_for_kbit_training(model)
        else:
            return self._from_pretrained_with_fallback(
                torch_dtype=self._dtype, device_map=self._device
            )

    def _load_tokenizer(self, path):
        self._tokenizer = AutoTokenizer.from_pretrained(
            path, **self._tokenizer_kwargs
        )
        self._setup_tokenizer()

    def _auto_merge_qlora(self, lora_dir: str):
        """QLoRA 合并：bf16 base + 注入 LoRA + load + merge。"""
        base = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self._model_name, 
            torch_dtype=self._dtype,
            device_map=self._device,
        )
        for p in base.parameters():
            p.requires_grad = False

        assert self._lora_config is not None
        inject_lora(base, self._lora_config)
        load_lora(base, os.path.join(lora_dir, "lora.pt"))
        merge_lora(base)
        return base

    def _load_from(self, save_dir):
        merged_dir = os.path.join(save_dir, "merged")
        lora_dir = os.path.join(save_dir, "lora")

        # ── 情况 1：合并后的完整模型 ──
        if os.path.isdir(merged_dir):
            self._load_tokenizer(merged_dir)
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=merged_dir, 
                torch_dtype=self._dtype,
                device_map=self._device,
            )
            self._lora_config = None
            logger.info(f"[HFModel] 已加载合并模型: {merged_dir}")
            return

        # ── 情况 2：base + LoRA adapter ──
        if os.path.isdir(lora_dir):
            config_path = os.path.join(lora_dir, "lora_config.json")
            if os.path.isfile(config_path):
                self._lora_config = LoraConfig.load(config_path)
                logger.info(f"[HFModel] 已从磁盘加载 LoraConfig: {config_path}")
            elif self._lora_config is None:
                raise ValueError(
                    f"未找到 {config_path}，且未传入 lora_config，无法重建 LoRA 结构"
                )

            self._load_tokenizer(lora_dir)

            # QLoRA：自动合并为 bf16
            if self._lora_config.use_qlora:
                logger.info(f"[HFModel] 检测到 QLoRA，自动合并为 bf16 模型...")
                self._model = self._auto_merge_qlora(lora_dir)
                self._lora_config = None
                logger.info(f"[HFModel] QLoRA 合并完成")
                return

            # 普通 LoRA：保持 LoRA 结构
            model = self._load_base(use_qlora=False)
            for p in model.parameters():
                p.requires_grad = False
            inject_lora(model, self._lora_config)
            load_lora(model, os.path.join(lora_dir, "lora.pt"))

            self._model = model
            logger.info(f"[HFModel] 已加载普通 LoRA: {lora_dir}")
            return

        # ── 情况 3：save_dir 直接是完整模型 ──
        self._load_tokenizer(save_dir)
        self._model = AutoModelForCausalLM.from_pretrained(
            save_dir, 
            torch_dtype=self._dtype, 
            device_map=self._device,
        )
        self._lora_config = None
        logger.info(f"[HFModel] 已加载完整模型: {save_dir}")
    
    def _from_pretrained_with_fallback(self, **kwargs):
        """统一：先本地缓存，失败再远程下载。"""
        merged = {**self._model_kwargs, **kwargs}
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self._model_name, local_files_only=True, **merged
            )
            logger.info(f"[HFModel] 已从本地缓存加载: {self._model_name}")
        except OSError:
            logger.info(f"[HFModel] 本地未找到缓存，正在从远程下载: {self._model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                self._model_name, **merged
            )
        return model
    
    # ============================================================
    # from_pretrained / from_model（其余逻辑同原代码，只改类名）
    # ============================================================
    
    @classmethod
    def from_pretrained(
        cls, save_dir, model_name: str, device="cuda",
        dtype=None, tokenizer_kwargs=None, model_kwargs=None,
    ):
        obj = cls.__new__(cls)
        obj._model_name = model_name
        obj._device = device
        obj._dtype = dtype or cls.DEFAULT_DTYPE
        obj._tokenizer_kwargs = tokenizer_kwargs or {}
        obj._model_kwargs = model_kwargs or {}
        obj._lora_config = None
        obj._model = None
        obj._tokenizer = None
        obj._load_from(save_dir)
        return obj
    
    # _load_from / _auto_merge_qlora / _load_tokenizer 逻辑不变
    # ...
    
    @classmethod
    def from_model(cls, model, lora_config, tokenizer, model_name: str, device="cuda"):
        obj = cls.__new__(cls)
        obj._model_name = model_name
        obj._lora_config = lora_config
        obj._device = device
        obj._dtype = cls.DEFAULT_DTYPE
        obj._tokenizer_kwargs = {}
        obj._model_kwargs = {}
        obj._model = model
        obj._tokenizer = tokenizer
        return obj
    
    # ============================================================
    # 访问器
    # ============================================================

    def model(self):
        return self._model

    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def has_lora(self) -> bool:
        return self._lora_config is not None

    @property
    def is_qlora(self) -> bool:
        return self.has_lora and self._lora_config is not None and self._lora_config.use_qlora

    # ============================================================
    # 保存
    # ============================================================

    def save(self, save_dir: str):
        """
        保存模型

        - 无 LoRA：保存完整模型
        - 普通 LoRA：保存 lora/ + merged/
        - QLoRA：保存 lora/ + merged/（用 bf16 base 合并）
        """
        os.makedirs(save_dir, exist_ok=True)

        # ============ 无 LoRA：直接保存 ============
        if not self.has_lora:
            self._model.save_pretrained(save_dir)
            self._tokenizer.save_pretrained(save_dir)
            logger.info(f"[HFModel] 完整模型已保存到 {save_dir}")
            return

        # ============ 有 LoRA：先保存 adapter ============
        lora_dir = os.path.join(save_dir, "lora")
        os.makedirs(lora_dir, exist_ok=True)
        save_lora(self._model, os.path.join(lora_dir, "lora.pt"))
        self._tokenizer.save_pretrained(lora_dir)
        assert self._lora_config is not None
        self._lora_config.save(os.path.join(lora_dir, "lora_config.json"))
        logger.info(f"[HFModel] LoRA adapter 已保存到 {lora_dir}")

        # ============ 合并 ============
        merged_dir = os.path.join(save_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)

        if self.is_qlora:
            # QLoRA：加载 bf16 base + 加载 LoRA 参数 + 合并
            merged = merge_qlora_to_bf16(
                model_name=self._model_name,
                qlora_model=self._model,
                lora_config=self._lora_config,
            )
        else:
            # 普通 LoRA：直接深拷贝当前模型，搬到 CPU 后合并
            merged = deepcopy(self._model).cpu()
            merge_lora(merged)

        merged.save_pretrained(merged_dir)
        self._tokenizer.save_pretrained(merged_dir)
        logger.info(f"[HFModel] 合并模型已保存到 {merged_dir}")

        del merged
        torch.cuda.empty_cache()
