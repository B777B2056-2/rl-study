from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset


class BaseDatasetAdapter(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_raw_train_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def get_raw_test_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def build_user_message(self, data) -> list[dict]:
        pass

    @abstractmethod
    def build_assistant_message(self, data) -> list[dict]:
        pass

    @abstractmethod
    def build_ground_truth(self, data) -> str:
        """构造标准答案（用于奖励函数）"""
        pass


class GSM8kDatasetAdapter(BaseDatasetAdapter):
    def __init__(self):
        super().__init__()

        raw_dataset = load_dataset("openai/gsm8k", "main")
        self._raw_train_dataset = raw_dataset['train']
        self._raw_test_dataset = raw_dataset['test']

    def name(self) -> str:
        return "openai/gsm8k"

    def get_raw_train_dataset(self) -> Dataset:
        return self._raw_train_dataset

    def get_raw_test_dataset(self) -> Dataset:
        return self._raw_test_dataset

    def build_user_message(self, data) -> list[dict]:
        return [{"role": "user", "content": data["question"]}]

    def build_assistant_message(self, data) -> list[dict]:
        return [{"role": "assistant", "content": data["answer"]}]

    def build_ground_truth(self, data) -> str:
        """从 answer 里提取 #### 后面的数字"""
        answer = data["answer"]
        if "####" in answer:
            return answer.split("####")[-1].strip()
        return answer.strip()