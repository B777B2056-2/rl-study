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

class DPODataset:
    """DPO 数据集：返回偏好对的 token 序列"""

    def __init__(self, tokenizer, adapter: BaseDatasetAdapter,
                 batch_size: int = 1,
                 max_length: int = 512,
                 max_prompt_length: int = 256):
        """
        参数:
            tokenizer:          HuggingFace tokenizer
            adapter:            DPO 数据适配器
            batch_size:         DataLoader 的 batch size
            max_length:         prompt + response 的最大长度
            max_prompt_length:  prompt 单独的最大长度
        """
        self._tokenizer = tokenizer
        self._adapter = adapter
        self._batch_size = batch_size
        self._max_length = max_length
        self._max_prompt_length = max_prompt_length

        # 处理数据
        raw_train = adapter.get_raw_train_dataset()
        raw_test = adapter.get_raw_test_dataset()

        self._train_dataset = (
            raw_train
            .map(self._process_one, desc="Processing train")
            .filter(self._filter_fn)
        )
        self._test_dataset = (
            raw_test
            .map(self._process_one, desc="Processing test")
            .filter(self._filter_fn)
        )

        # 只保留需要的字段
        keep = {
            "input_ids", "input_attention_mask",
            "chosen_ids", "chosen_attention_mask",
            "rejected_ids", "rejected_attention_mask",
        }
        self._train_dataset = self._train_dataset.remove_columns(
            [c for c in self._train_dataset.column_names if c not in keep]
        )
        self._test_dataset = self._test_dataset.remove_columns(
            [c for c in self._test_dataset.column_names if c not in keep]
        )

        self._train_dataloader = None
        self._test_dataloader = None

    # ============================================================
    # 分词
    # ============================================================
    def _tokenize_prompt(self, text: str) -> list:
        """对 prompt 分词，末尾加 assistant 开头"""
        prompt_text = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = self._tokenizer(
            prompt_text, add_special_tokens=False,
        ).input_ids
        return ids[:self._max_prompt_length]

    def _tokenize_response(self, text: str) -> list:
        """对 response 分词，末尾加 EOS"""
        ids = self._tokenizer(
            text, add_special_tokens=False,
        ).input_ids
        ids = ids + [self._tokenizer.eos_token_id]
        return ids

    # ============================================================
    # 单条处理
    # ============================================================
    def _process_one(self, data: dict) -> dict:
        # 1. prompt
        input_ids = self._tokenize_prompt(self._adapter.build_prompt(data))

        # 2. response 最大长度
        max_resp = self._max_length - len(input_ids)
        if max_resp <= 0:
            # prompt 太长，丢弃（filter 会过滤掉）
            return {
                "input_ids": input_ids,
                "input_attention_mask": [1] * len(input_ids),
                "chosen_ids": [],
                "chosen_attention_mask": [],
                "rejected_ids": [],
                "rejected_attention_mask": [],
            }

        # 3. chosen 和 rejected
        chosen_ids = self._tokenize_response(
            self._adapter.build_chosen(data)
        )[:max_resp]
        rejected_ids = self._tokenize_response(
            self._adapter.build_rejected(data)
        )[:max_resp]

        return {
            "input_ids": input_ids,
            "input_attention_mask": [1] * len(input_ids),
            "chosen_ids": chosen_ids,
            "chosen_attention_mask": [1] * len(chosen_ids),
            "rejected_ids": rejected_ids,
            "rejected_attention_mask": [1] * len(rejected_ids),
        }

    def _filter_fn(self, data: dict) -> bool:
        """过滤无效样本"""
        return (
            len(data["input_ids"]) > 0
            and len(data["chosen_ids"]) > 0
            and len(data["rejected_ids"]) > 0
        )

    # ============================================================
    # Padding 工具
    # ============================================================
    def _pad(self, batch_ids, pad_id):
        """把一批变长序列 padding 到同一长度"""
        max_len = max(len(ids) for ids in batch_ids)
        padded = []
        masks = []
        for ids in batch_ids:
            pad_len = max_len - len(ids)
            padded.append(ids + [pad_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long),
        )

    # ============================================================
    # Collate
    # ============================================================
    def _collate_fn(self, batch):
        """
        把一批偏好对 padding 成张量

        返回:
            input_ids:              (B, P)    prompt
            input_attention_mask:   (B, P)
            chosen_ids:             (B, R1)   chosen response
            chosen_attention_mask:  (B, R1)
            rejected_ids:           (B, R2)
            rejected_attention_mask:(B, R2)
        """
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id

        input_ids, input_mask = self._pad(
            [x["input_ids"] for x in batch], pad_id
        )
        chosen_ids, chosen_mask = self._pad(
            [x["chosen_ids"] for x in batch], pad_id
        )
        rejected_ids, rejected_mask = self._pad(
            [x["rejected_ids"] for x in batch], pad_id
        )

        return {
            "input_ids": input_ids,
            "input_attention_mask": input_mask,
            "chosen_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
        }

    # ============================================================
    # DataLoader
    # ============================================================
    def build_train_data_loader(self):
        if self._train_dataloader is None:
            self._train_dataloader = torch.utils.data.DataLoader(
                self._train_dataset,
                batch_size=self._batch_size,
                shuffle=True,
                collate_fn=self._collate_fn,
            )
        return self._train_dataloader

    def build_test_data_loader(self):
        if self._test_dataloader is None:
            self._test_dataloader = torch.utils.data.DataLoader(
                self._test_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                collate_fn=self._collate_fn,
            )
        return self._test_dataloader