"""
RL 数据集：只需要 prompt，不需要 response

和 InstructDataset 的区别：
    - SFT：返回 input_ids + labels（response 已给定）
    - RL：返回 input_ids + ground_truth（response 由 Actor 生成）
"""
from src.data.adapters import BaseDatasetAdapter
import torch


# ============================================================
# RL 数据集
# ============================================================
class RLDataset(object):
    """
    RL 数据集：把原始数据转成 prompt 的 token 序列

    返回：
        - input_ids:      prompt 的 token ID
        - attention_mask: 1 表示有效
        - ground_truth:   标准答案（reward 函数用）
    """

    def __init__(self, tokenizer, adapter: BaseDatasetAdapter,
                 batch_size: int = 4, max_length: int = 256):
        self._tokenizer = tokenizer
        self._adapter = adapter
        self._batch_size = batch_size
        self._max_length = max_length

        raw_train_dataset = adapter.get_raw_train_dataset()
        raw_test_dataset = adapter.get_raw_test_dataset()

        # 过滤超长样本
        self._train_dataset = (
            raw_train_dataset
            .map(self._process_one, desc="Processing train")
            .filter(lambda x: 0 < len(x["input_ids"]) <= self._max_length)
        )
        self._test_dataset = (
            raw_test_dataset
            .map(self._process_one, desc="Processing test")
            .filter(lambda x: 0 < len(x["input_ids"]) <= self._max_length)
        )

        # 去掉原始字段（节省内存）
        self._train_dataset = self._train_dataset.remove_columns(
            [c for c in self._train_dataset.column_names
             if c not in ("input_ids", "attention_mask", "ground_truth", "prompt_text")]
        )
        self._test_dataset = self._test_dataset.remove_columns(
            [c for c in self._test_dataset.column_names
             if c not in ("input_ids", "attention_mask", "ground_truth", "prompt_text")]
        )

        self._train_dataloader = None
        self._test_dataloader = None

    # ============================================================
    # 单条处理
    # ============================================================
    def _process_one(self, data: dict) -> dict:
        """把一条原始数据转成 prompt 的 token 序列"""
        user_msg = self._adapter.build_user_message(data)

        # 应用 chat 模板，末尾加 assistant 开头（推理时必须）
        prompt_text = self._tokenizer.apply_chat_template(
            user_msg,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 分词
        input_ids = self._tokenizer(
            prompt_text, add_special_tokens=False,
        ).input_ids

        # 截断
        input_ids = input_ids[:self._max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "ground_truth": self._adapter.build_ground_truth(data),
            "prompt_text": prompt_text,     # 可选，方便调试
        }

    # ============================================================
    # collate_fn：padding + 转 tensor
    # ============================================================
    def _collate_fn(self, batch):
        """把一批样本 padding 到同一长度，转 tensor"""
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids_list = []
        attention_mask_list = []

        for item in batch:
            ids = item["input_ids"]
            mask = item["attention_mask"]
            pad_len = max_len - len(ids)

            pad_token_id = self._tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self._tokenizer.eos_token_id

            input_ids_list.append(ids + [pad_token_id] * pad_len)
            attention_mask_list.append(mask + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            # 非 tensor 字段单独返回
            "ground_truth": [x["ground_truth"] for x in batch],
            "prompt_text": [x["prompt_text"] for x in batch],
        }

    # ============================================================
    # DataLoader
    # ============================================================
    def build_train_data_loader(self) -> torch.utils.data.DataLoader:
        if self._train_dataloader is None:
            self._train_dataloader = torch.utils.data.DataLoader(
                self._train_dataset,
                batch_size=self._batch_size,
                shuffle=True,
                collate_fn=self._collate_fn,
            )
        return self._train_dataloader

    def build_test_data_loader(self) -> torch.utils.data.DataLoader:
        if self._test_dataloader is None:
            self._test_dataloader = torch.utils.data.DataLoader(
                self._test_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                collate_fn=self._collate_fn,
            )
        return self._test_dataloader