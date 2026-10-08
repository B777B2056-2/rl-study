from dataclasses import dataclass
import torch
from tqdm import tqdm
from copy import deepcopy
from src.models import LoraConfig, inject_lora
from src.plotter import Plotter
from src.data import DPODataset
from src.ckpt import *


@dataclass
class DPOConfig:
    dataset: DPODataset
    ckpt_config: CkptConfig
    n_epoch: int
    learning_rate: float
    reward_beta: float
    grad_clip_eps: float
    policy_model_lora_config: LoraConfig = None
    device: str = "cuda"


class Policy(torch.nn.Module):
    """
        策略模型
    """
    def __init__(self, base_model, lora_config: LoraConfig):
        super(Policy, self).__init__()
        # 冻结SFT模型
        for p in base_model.parameters():
            p.requires_grad = False

        if lora_config is not None:
            # 注入 LoRA
            inject_lora(base_model, lora_config)

        self._policy = base_model
        self._lora_config = lora_config

    def forward(self, input_ids, attention_mask):
        # 获取policy模型前向传播的输出
        logits = self._policy(input_ids, attention_mask=attention_mask).logits
        return logits

    def model(self):
        return self._policy

    def lora_config(self):
        return self._lora_config


class Reference(torch.nn.Module):
    """
        参考模型：冻结权重，不更新，主要目的为计算KL惩罚，防止PPO微调结果与SFT的结果有较大差距
    """
    def __init__(self, base_model):
        super().__init__()
        self._ref_model = base_model

        # 冻结
        for p in self._ref_model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self._ref_model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )
        return outputs.logits


class DPOTrainer(CheckpointableTrainer):
    def __init__(self, base_model, tokenizer, config: DPOConfig):
        self._plotter = Plotter()
        ckpt_manager = CheckpointManager(config=config.ckpt_config)
        super().__init__(ckpt_manager, self._plotter)

        self._config = config
        self._tokenizer = tokenizer

        self._policy_model = Policy(base_model=deepcopy(base_model), lora_config=config.policy_model_lora_config)
        self._ref_model = Reference(base_model=base_model)

        self._trainable_params = [p for p in self._policy_model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(
            self._trainable_params,
            lr=config.learning_rate,
        )

    @classmethod
    def _calc_log_probs(cls, model, input_ids, input_attention_mask, response_ids, response_attention_mask):
        """对已有回答计算一次前向的对数概率，相当于给已有回答打分，与PPO不同"""
        # 1. 拼接为完整id列表，完整id列表 = input_ids + response_ids
        full_ids = torch.concat(tensors=(input_ids, response_ids), dim=1)
        full_attention_mask = torch.concat(tensors=(input_attention_mask, response_attention_mask), dim=1)

        # 2. 进行一次前向，获取对数概率，即各token打分值
        prompt_len = input_ids.shape[1]
        full_logits = model(full_ids, full_attention_mask)
        response_logits = full_logits[:, prompt_len-1:-1, :]                      # 切出response在词表里的原始logits
        log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)      # 取对数概率
        log_probs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)  # 取response各实际token的实际对数概率

        # 3. 忽略 padding
        log_probs = log_probs * response_attention_mask.float()  

        # 4. 整段求和
        log_probs = log_probs.sum(dim=-1)
        return log_probs

    def train(self) -> None:
        """dpo训练"""
        # 从 checkpoint 恢复
        self.maybe_resume()
        
        train_data_loader = self._config.dataset.build_train_data_loader()
        for ep in range(self._epoch, self._config.n_epoch):
            self.set_epoch(ep)

            batch_loss, step_cnt = 0.0, 0
            for batch in tqdm(train_data_loader, desc="Training"):
                # 1. 取偏好对（prompt-chosen， prompt-rejected）
                input_ids = batch["input_ids"].to(self._config.device)
                input_attention_mask = batch["input_attention_mask"].to(self._config.device)

                chosen_ids = batch["chosen_ids"].to(self._config.device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(self._config.device)

                rejected_ids = batch["rejected_ids"].to(self._config.device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(self._config.device)

                # 2. 通过模型一次前向，计算对数概率，给已有回答打分（策略模型）
                policy_chosen_logp = DPOTrainer._calc_log_probs(
                    model=self._policy_model,
                    input_ids=input_ids,
                    input_attention_mask=input_attention_mask,
                    response_ids=chosen_ids,
                    response_attention_mask=chosen_attention_mask,
                )
                policy_rejected_logp = DPOTrainer._calc_log_probs(
                    model=self._policy_model,
                    input_ids=input_ids,
                    input_attention_mask=input_attention_mask,
                    response_ids=rejected_ids,
                    response_attention_mask=rejected_attention_mask,
                )

                # 3. 通过模型一次前向，计算对数概率，给已有回答打分（参考模型）
                ref_chosen_logp = DPOTrainer._calc_log_probs(
                    model=self._ref_model,
                    input_ids=input_ids,
                    input_attention_mask=input_attention_mask,
                    response_ids=chosen_ids,
                    response_attention_mask=chosen_attention_mask,
                )
                ref_rejected_logp = DPOTrainer._calc_log_probs(
                    model=self._ref_model,
                    input_ids=input_ids,
                    input_attention_mask=input_attention_mask,
                    response_ids=rejected_ids,
                    response_attention_mask=rejected_attention_mask,
                )

                # 4. 计算隐式奖励差
                rewards_diff = self._config.reward_beta * ((policy_chosen_logp - ref_chosen_logp) - (policy_rejected_logp - ref_rejected_logp))

                # 5. 算损失
                loss = -torch.nn.functional.logsigmoid(rewards_diff).mean()
                batch_loss += loss.item()
                step_cnt += 1

                # 6. 梯度更新
                self._optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪：防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self._trainable_params, self._config.grad_clip_eps)
                self._optimizer.step()

                # 7. 定期保存 checkpoint
                self.maybe_save()

            self._plotter.log(name="train_loss", value=batch_loss/step_cnt, step=ep)
        self._plotter.plot(names=['train_loss'])

    @torch.no_grad()
    def eval(self):
        """在测试集上评估 DPO loss 和准确率"""
        self._policy_model.eval()

        n = 0

        test_data_loader = self._config.dataset.build_test_data_loader()
        for batch in tqdm(test_data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self._config.device)
            input_attention_mask = batch["input_attention_mask"].to(self._config.device)

            chosen_ids = batch["chosen_ids"].to(self._config.device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(self._config.device)

            rejected_ids = batch["rejected_ids"].to(self._config.device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(self._config.device)

            policy_chosen_logp = DPOTrainer._calc_log_probs(
                model=self._policy_model,
                input_ids=input_ids,
                input_attention_mask=input_attention_mask,
                response_ids=chosen_ids,
                response_attention_mask=chosen_attention_mask,
            )
            policy_rejected_logp = DPOTrainer._calc_log_probs(
                model=self._policy_model,
                input_ids=input_ids,
                input_attention_mask=input_attention_mask,
                response_ids=rejected_ids,
                response_attention_mask=rejected_attention_mask,
            )

            ref_chosen_logp = DPOTrainer._calc_log_probs(
                model=self._ref_model,
                input_ids=input_ids,
                input_attention_mask=input_attention_mask,
                response_ids=chosen_ids,
                response_attention_mask=chosen_attention_mask,
            )
            ref_rejected_logp = DPOTrainer._calc_log_probs(
                model=self._ref_model,
                input_ids=input_ids,
                input_attention_mask=input_attention_mask,
                response_ids=rejected_ids,
                response_attention_mask=rejected_attention_mask,
            )

            acc = ((policy_chosen_logp - ref_chosen_logp) > (policy_rejected_logp - ref_rejected_logp)).float().mean().item()
            n += 1

            self._plotter.log(name="test_acc", value=acc, step=n)
        self._plotter.plot(names=['test_acc'])
        self._policy_model.train()

    def _state_dict(self) -> dict:
        return {
            "policy_lora": {
                n: p.data.cpu()
                for n, p in self._policy_model.named_parameters()
                if p.requires_grad
            },
            "optimizer": self._optimizer.state_dict(),
            "rng": get_rng_state(),
        }

    def _load_state_dict(self, state: dict):
        current = dict(self._policy_model.named_parameters())
        for n, p in state["policy_lora"].items():
            if n in current:
                current[n].data.copy_(p.to(current[n].device))

        self._optimizer.load_state_dict(state["optimizer"])
        if "rng" in state:
            set_rng_state(state["rng"])

    def policy(self):
        return self._policy_model


if __name__ == "__main__":
    from src.models import Qwen2_5, LoraConfig
    from src.data import GSM8kDatasetAdapter

    # ============================================================
    # 1. 加载 SFT 模型
    # ============================================================
    qwen = Qwen2_5.from_pretrained(
        save_dir="./outputs/gsm8k-sft-qwen2_5-0_5B",
    )
    sft_model = qwen.model()
    tokenizer = qwen.tokenizer()
    print(f"SFT 模型已加载")
    print(f"设备: {qwen.device}")

    # ============================================================
    # 2. 构建 DPO 数据集
    # ============================================================
    adapter = GSM8kDatasetAdapter()
    dataset = DPODataset(
        tokenizer=tokenizer,
        adapter=adapter,
        batch_size=1,                # 6GB 显存建议为 1
        max_length=512,              # prompt + response 最大长度
        max_prompt_length=256,       # prompt 最大长度
    )

    train_loader = dataset.build_train_data_loader()
    test_loader = dataset.build_test_data_loader()
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    print(f"训练集 batch 数: {len(train_loader)}")

    # ============================================================
    # 3. 构建 DPOTrainer
    # ============================================================
    dpo_trainer = DPOTrainer(
        base_model=sft_model,
        tokenizer=tokenizer,
        config=DPOConfig(
            # 数据
            dataset=dataset,

            # checkpoint
            ckpt_config=CkptConfig(
                ckpt_dir="./outputs/gsm8k-dpo/ckpt",
                save_steps=500,
                keep_last=3,
            ),

            # 训练
            n_epoch=3,
            learning_rate=5e-6,
            reward_beta=0.1,
            grad_clip_eps=1.0,

            # Policy 的 LoRA
            policy_model_lora_config=LoraConfig(
                alpha=16,
                rank=8,
                dropout=0.05,
                use_qlora=False,     # SFT base 已经是 bf16
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),

            device=qwen.device,
        ),
    )

    # ============================================================
    # 4. 训练
    # ============================================================
    dpo_trainer.train()

    # ============================================================
    # 5. 评估
    # ============================================================
    dpo_trainer.eval()

    # ============================================================
    # 6. 保存
    # ============================================================
    policy = dpo_trainer.policy()
    qwen_ppo = Qwen2_5.from_model(
        model=policy.model(),
        lora_config=policy.lora_config(),
        tokenizer=tokenizer,
    )
    qwen_ppo.save(save_dir='./outputs/gsm8k-ppo-qwen2_5-0_5B')