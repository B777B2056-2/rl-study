import torch
from src.data.adapters import BaseDatasetAdapter


class InstructDataset(object):
    """指令数据集：将其调整为SFT格式"""
    def __init__(self, tokenizer, adapter: BaseDatasetAdapter, batch_size: int = 64):
        self._tokenizer = tokenizer
        self._adapter = adapter
        self._batch_size = batch_size

        raw_train_dataset = self._adapter.get_raw_train_dataset()
        raw_test_dataset = self._adapter.get_raw_test_dataset()

        self._train_dataset = raw_train_dataset.map(
            self._process_one, 
            remove_columns=raw_train_dataset.column_names,
            desc="Tokenizing and generate labels for train",
        )
        self._test_dataset = raw_test_dataset.map(
            self._process_one, 
            remove_columns=raw_test_dataset.column_names,
            desc="Tokenizing and generate labels for test",
        )

        self._train_dataloader = None
        self._test_dataloader = None

    def _tokenize(self, data: dict):
        """对单个raw数据进行分词向量化"""
        user_msg = self._adapter.build_user_message(data)
        assistant_msg = self._adapter.build_assistant_message(data)

        # prompt：只有 user，末尾加 assistant 开头
        prompt_text = self._tokenizer.apply_chat_template(
            user_msg,
            tokenize=False,
            add_generation_prompt=True,
        )

        # full：user + assistant
        full_text = self._tokenizer.apply_chat_template(
            user_msg + assistant_msg,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self._tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = self._tokenizer(full_text, add_special_tokens=False).input_ids
        
        return prompt_ids, full_ids

    def _generate_label(self, prompt_ids, full_ids):
        """对单个raw数据的向量，构造对应的标签"""
        # prompt的内容不参与预测，因此将其token全部调整为-100，相当于掩码
        # assistant的回答才是需要预测的，因此full_ids从超过prompt的部分开始截取，与之前的掩码拼在一起
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return labels

    def _process_one(self, data: dict):
        """分词 + 构造 labels + attention_mask"""
        prompt_ids, full_ids = self._tokenize(data)
        labels = self._generate_label(prompt_ids, full_ids)

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }

    def _collate_fn(self, batch):
        """找最大长度、逐条 padding、转 tensor"""
        # 找最大长度
        max_len = max([len(x["input_ids"]) for x in batch])
        # 逐个padding
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for item in batch:
            input_ids = item["input_ids"]
            labels = item["labels"]
            attention_mask = item["attention_mask"]

            pad_len = max_len - len(input_ids)
            input_ids_pad_token_id = self._tokenizer.pad_token_id
            if input_ids_pad_token_id is None:
                input_ids_pad_token_id = self._tokenizer.eos_token_id

            input_ids_list.append(input_ids + [input_ids_pad_token_id] * pad_len)
            labels_list.append(labels + [-100] * pad_len)
            attention_mask_list.append(attention_mask + [0] * pad_len)

        # 转 tensor
        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        }

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

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from data.adapters import GSM8kDatasetAdapter

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    from models import Qwen2_5
    tokenizer = Qwen2_5().tokenizer()

    gsm8k_adapter = GSM8kDatasetAdapter()
    sft_dataset = InstructDataset(tokenizer=tokenizer, adapter=gsm8k_adapter)

    train_loader = sft_dataset.build_train_data_loader()
    test_loader = sft_dataset.build_test_data_loader()

    print(f"\n训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    print(f"训练集 batch 数: {len(train_loader)}")
    print(f"测试集 batch 数: {len(test_loader)}")

    def print_batch(batch: dict, tag: str, batch_idx: int, max_samples: int = 2):
        """打印一个 batch 的关键信息"""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        B, T = input_ids.shape
        print(f"\n{'=' * 70}")
        print(f"[{tag}] batch {batch_idx} | shape: {tuple(input_ids.shape)}")
        print(f"{'=' * 70}")

        for i in range(min(B, max_samples)):
            ids = input_ids[i].tolist()
            lbs = labels[i].tolist()
            mask = attention_mask[i].tolist()

            valid_len = sum(mask)
            loss_tokens = sum(1 for l in lbs if l != -100)

            print(f"\n  --- 样本 {i} ---")
            print(f"  有效长度: {valid_len} / {T}")
            print(f"  参与 loss 的 token 数: {loss_tokens}")

            text = tokenizer.decode(ids, skip_special_tokens=False)
            print(f"  input  : {repr(text[:150])}")

            response_ids = [l for l in lbs if l != -100]
            if response_ids:
                response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
                print(f"  response: {repr(response_text[:150])}")
            else:
                print(f"  response: <空>")

            print(f"  input_ids[:10]: {ids[:10]}")
            print(f"  labels[:10]   : {lbs[:10]}")

    def iterate(loader, tag: str, max_batches: int = 2):
        """遍历一个 DataLoader"""
        total_batches = 0
        total_samples = 0
        for batch_idx, batch in enumerate(loader):
            total_batches += 1
            total_samples += batch["input_ids"].size(0)
            print_batch(batch, tag, batch_idx)
            if batch_idx + 1 >= max_batches:
                break
        print(f"\n[{tag}] 共遍历 {total_batches} 个 batch, {total_samples} 条样本")

    # ============================================================
    # 统计函数
    # ============================================================
    def collect_stats(dataset):
        """
        遍历数据集，收集长度统计

        返回:
            total_lens:    总长度列表
            prompt_lens:   prompt 长度列表
            response_lens: response 长度列表
        """
        total_lens = []
        prompt_lens = []
        response_lens = []

        for i in tqdm(range(len(dataset)), desc="统计中"):
            item = dataset[i]
            ids = item["input_ids"]
            lbs = item["labels"]

            total_lens.append(len(ids))

            # response 长度 = labels 里非 -100 的个数
            response_len = sum(1 for l in lbs if l != -100)
            response_lens.append(response_len)

            # prompt 长度 = 总长 - response
            prompt_lens.append(len(ids) - response_len)

        return np.array(total_lens), np.array(prompt_lens), np.array(response_lens)

    def print_summary(name, total, prompt, response):
        """打印统计摘要"""
        print(f"\n{'=' * 70}")
        print(f"{name} 统计")
        print(f"{'=' * 70}")
        print(f"样本数: {len(total)}")
        print(f"总长度   - 均值: {total.mean():.1f}, "
              f"中位数: {np.median(total):.1f}, "
              f"最大: {total.max()}, "
              f"95%分位: {np.percentile(total, 95):.1f}, "
              f"99%分位: {np.percentile(total, 99):.1f}")
        print(f"Prompt   - 均值: {prompt.mean():.1f}, 最大: {prompt.max()}")
        print(f"Response - 均值: {response.mean():.1f}, 最大: {response.max()}")
        print(f"Prompt 占比:   {prompt.mean() / total.mean() * 100:.1f}%")
        print(f"Response 占比: {response.mean() / total.mean() * 100:.1f}%")

    def plot_stats(train_stats, test_stats, save_path=None):
        """画统计图"""
        train_total, train_prompt, train_response = train_stats
        test_total, test_prompt, test_response = test_stats

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        bins = 50

        # 1. 总长度分布
        ax = axes[0, 0]
        ax.hist(train_total, bins=bins, alpha=0.6, label='train', color='C0')
        ax.hist(test_total, bins=bins, alpha=0.6, label='test', color='C1')
        ax.axvline(train_total.mean(), color='C0', linestyle='--',
                   label=f'train mean={train_total.mean():.0f}')
        ax.axvline(test_total.mean(), color='C1', linestyle='--',
                   label=f'test mean={test_total.mean():.0f}')
        ax.set_xlabel("总长度 (token)")
        ax.set_ylabel("样本数")
        ax.set_title("总长度分布")
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. prompt 长度分布
        ax = axes[0, 1]
        ax.hist(train_prompt, bins=bins, alpha=0.6, label='train', color='C0')
        ax.hist(test_prompt, bins=bins, alpha=0.6, label='test', color='C1')
        ax.set_xlabel("Prompt 长度 (token)")
        ax.set_ylabel("样本数")
        ax.set_title("Prompt 长度分布")
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. response 长度分布
        ax = axes[0, 2]
        ax.hist(train_response, bins=bins, alpha=0.6, label='train', color='C0')
        ax.hist(test_response, bins=bins, alpha=0.6, label='test', color='C1')
        ax.set_xlabel("Response 长度 (token)")
        ax.set_ylabel("样本数")
        ax.set_title("Response 长度分布")
        ax.legend()
        ax.grid(alpha=0.3)

        # 4. 累积分布 CDF
        ax = axes[1, 0]
        for data, name in [(train_total, 'train'), (test_total, 'test')]:
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, cdf, label=name)
        ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='95%')
        ax.axhline(0.99, color='orange', linestyle='--', alpha=0.5, label='99%')
        ax.set_xlabel("总长度 (token)")
        ax.set_ylabel("累积比例")
        ax.set_title("累积分布 (CDF)")
        ax.legend()
        ax.grid(alpha=0.3)

        # 5. prompt vs response 平均长度对比
        ax = axes[1, 1]
        categories = ['Train', 'Test']
        prompt_means = [train_prompt.mean(), test_prompt.mean()]
        response_means = [train_response.mean(), test_response.mean()]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width/2, prompt_means, width, label='prompt', color='C0')
        ax.bar(x + width/2, response_means, width, label='response', color='C1')

        for i, (p, r) in enumerate(zip(prompt_means, response_means)):
            ax.text(i - width/2, p, f'{p:.0f}', ha='center', va='bottom')
            ax.text(i + width/2, r, f'{r:.0f}', ha='center', va='bottom')

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("平均长度 (token)")
        ax.set_title("Prompt vs Response 平均长度")
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # 6. prompt / response 占比
        ax = axes[1, 2]
        train_prompt_pct = train_prompt.mean() / train_total.mean() * 100
        train_response_pct = train_response.mean() / train_total.mean() * 100

        ax.bar(['Train'], [train_prompt_pct], label='prompt', color='C0')
        ax.bar(['Train'], [train_response_pct], bottom=[train_prompt_pct],
               label='response', color='C1')

        ax.text(0, train_prompt_pct / 2, f'{train_prompt_pct:.1f}%',
                ha='center', va='center', color='white', fontweight='bold')
        ax.text(0, train_prompt_pct + train_response_pct / 2,
                f'{train_response_pct:.1f}%',
                ha='center', va='center', color='white', fontweight='bold')

        ax.set_ylabel("占比 (%)")
        ax.set_title("Prompt / Response 占比")
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f"\n图已保存到 {save_path}")

        plt.show()

    # ============================================================
    # 执行
    # ============================================================
    # 1. 遍历训练集（打印前 2 个 batch）
    iterate(train_loader, "TRAIN", max_batches=2)

    # 2. 遍历测试集（打印前 1 个 batch）
    iterate(test_loader, "TEST", max_batches=1)

    # 3. 统计长度分布
    print("\n收集训练集统计...")
    train_stats = collect_stats(sft_dataset._train_dataset)

    print("收集测试集统计...")
    test_stats = collect_stats(sft_dataset._test_dataset)

    # 4. 打印摘要
    print_summary("训练集", *train_stats)
    print_summary("测试集", *test_stats)

    # 5. 画图
    plot_stats(train_stats, test_stats, save_path="dataset_stats.png")
        