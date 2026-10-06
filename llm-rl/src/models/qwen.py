import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Qwen2_5(object):
    def __init__(self):
        self._model_name = "Qwen/Qwen2.5-0.5B"
        self._init_tokenizer()
        self._init_model()

    def _init_tokenizer(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _init_model(self):
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            dtype=torch.bfloat16,
        )

    def model(self):
        return self._model

    def tokenizer(self):
        return self._tokenizer


if __name__ == '__main__':
    print(type(Qwen2_5().tokenizer()))
    print(type(Qwen2_5().model()))
