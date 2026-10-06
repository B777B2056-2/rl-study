from dataclasses import dataclass
from src.data import InstructDataset
from src.plotter import Plotter
import torch
import math
from tqdm import tqdm
import os
from .lora import *


@dataclass
class SFTConfig(object):
    n_epoch: int
    max_lr: float
    n_lr_warmup_steps: int
    dataset: InstructDataset
    save_dir: str
    grad_accum: int = 0
    max_grad_norm: float = 1.0
    lora_config: LoraConfig = None  # lora配置，为None时代表不使用lora
    device: str = "cuda"

class SFTTrainer(object):
    """SFT训练器"""
    def __init__(self, tokenizer, model, config: SFTConfig):
        # 若有需要，则注入lora层
        if config.lora_config is not None:
            inject_lora(model, config.lora_config)

        self._tokenizer = tokenizer
        self._model = model.to(config.device)
        self._config = config
        self._train_dataloader = self._config.dataset.build_train_data_loader()
        self._test_dataloader = self._config.dataset.build_test_data_loader()

        # 开启梯度检查
        self._model.gradient_checkpointing_enable()

        # 优化器
        self._trainable_params = [p for p in self._model.parameters() if p.requires_grad] # 只优化 requires_grad=True 的参数
        self._optimizer = torch.optim.AdamW(self._trainable_params, lr=config.max_lr)
        # 余弦退火调度器
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=self._warmup_cosine_lr_lambda)

        # 绘图
        self._plotter = Plotter()

    def _warmup_cosine_lr_lambda(self, current_step: int) -> float:
        """学习率调度：预热 + 余弦退火"""
        warmup = self._config.n_lr_warmup_steps
        total = len(self._train_dataloader) * self._config.n_epoch

        # 预热阶段：线性上升
        if current_step < warmup:
            return float(current_step) / float(max(1, warmup))

        # 余弦退火：从 1 降到 0
        progress = float(current_step - warmup) / float(max(1, total - warmup))
        progress = min(1.0, progress)                 # ← 关键：防止反弹
        return 0.5 * (1.0 + math.cos(progress * math.pi))

    def train(self):
        """SFT后训练"""
        self._model.train()

        for epoch in range(self._config.n_epoch):
            total_loss = 0.0
            step_cnt = 0

            # 0. 遍历训练集
            for batch in tqdm(self._train_dataloader, desc="Training"):
                input_ids = batch["input_ids"].to(self._config.device)
                labels = batch["labels"].to(self._config.device)
                attention_mask = batch["attention_mask"].to(self._config.device)

                # 1. 前向传播一次，获取预测值
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits

                # 2. shift移位
                shift_logits = logits[:, :-1, :]  # 维度=(batch_size, seq_len, vocab_size), 对于预测结果，第i个结果预测的是第i+1个token，最后一个结果无意义
                shift_labels = labels[:, 1:]   # 维度=(batch_size, seq_len), 对于监督标签，需要右移一位与shift_logits对齐

                # 3. 交叉熵损失函数
                vocab_size = shift_logits.shape[2]
                loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(shift_logits.reshape(-1, vocab_size), shift_labels.reshape(-1))

                # 4. 反向传播
                loss.backward()
                total_loss += loss.item()
                step_cnt += 1

                # batch_size很小时，需要配置梯度累积更新，防止loss震荡；grad_accum配0则关闭梯度累积
                if (step_cnt + 1) % self._config.grad_accum == 0:
                    # 5. 梯度裁剪，LLM需防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(
                        self._trainable_params,
                        self._config.max_grad_norm,
                    )

                    # 6. 参数更新
                    self._optimizer.step()
                    self._scheduler.step()

                    # 7. 清空梯度
                    self._optimizer.zero_grad()

            self._plotter.log(name="train_loss", value=total_loss / step_cnt, step=epoch)
        self._save()
        self._plotter.plot(names=['train_loss'], save=os.path.join(self._config.save_dir, 'train.png'))
    
    @torch.no_grad()
    def eval(self):
        self._model.eval()

        for batch in tqdm(self._test_dataloader, desc="Testing"):
            input_ids = batch["input_ids"].to(self._config.device)
            labels = batch["labels"].to(self._config.device)
            attention_mask = batch["attention_mask"].to(self._config.device)

            # 1. 前向传播一次，获取预测值
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

            # 2. shift
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            # 2. 逐token比较，统计Token 级准确率
            pred_ids = shift_logits.argmax(dim=-1) 
            valid_mask = shift_labels != -100
            correct = (pred_ids == shift_labels) & valid_mask
            acc = correct.sum() / valid_mask.sum()

            self._plotter.log(name="test_token_acc", value=acc)

        self._plotter.plot(names=['test_token_acc'], save=os.path.join(self._config.save_dir, 'eval.png'))
        self._model.train()

    def _save(self):
        """保存：LoRA adapter + 合并后的完整模型"""
        os.makedirs(self._config.save_dir, exist_ok=True)

        # lora
        if self._config.lora_config is not None:
            # 1. 保存 LoRA adapter（几 MB）
            save_lora(self._model, os.path.join(self._config.save_dir, "lora.pt"))

            # 2. 保存合并后的完整模型
            import copy
            merged = copy.deepcopy(self._model)
            merge_lora(merged)
            merged.save_pretrained(os.path.join(self._config.save_dir, "merged"))
        else:
            self._model.save_pretrained(self._config.save_dir)

        self._tokenizer.save_pretrained(self._config.save_dir)
        print(f"[SFTTrainer] 模型已保存到 {self._config.save_dir}")


if __name__ == "__main__":
    from src.models import Qwen2_5
    from src.data import GSM8kDatasetAdapter

    tokenizer = Qwen2_5().tokenizer()
    model = Qwen2_5().model()

    gsm8k_adapter = GSM8kDatasetAdapter()
    gsm8k_sft_dataset = InstructDataset(tokenizer=tokenizer, adapter=gsm8k_adapter, batch_size=1)

    sft_trainer = SFTTrainer(tokenizer=tokenizer,model=model, config=SFTConfig(
        n_epoch=2,
        max_lr=2e-4,
        n_lr_warmup_steps=150,
        dataset=gsm8k_sft_dataset,
        save_dir='./outputs/gsm8k-sft-qwen2_5-0_5B',
        grad_accum=8,
        lora_config=LoraConfig(
            alpha=16,
            rank=8,
            dropout=0.05,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        ),
    ))

    sft_trainer.train()
    sft_trainer.eval()
