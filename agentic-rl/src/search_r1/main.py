from src.infra.models.hf_model import HFModel
from src.infra.models.lora import LoraConfig
from src.infra.environment.env import Environment
from src.infra.rollout.backends import InferenceBackendConfig, SamplingParams, HFBackendConfig
from src.infra.trainer.ckpt import CkptConfig
from src.infra.trainer.grpo import GRPOConfig, GRPOTrainer
from src.infra.log import setup_logging

from src.search_r1.dataset import NqAndHotpotqaDataset
from src.search_r1.tools import SearchTool, TagRouter
from src.search_r1.reward import SearchR1Reward


def main():
    BATCH_SIZE = 1
    MAX_TURNS = 5
    DEVICE = "cuda"

    # 日志：infra 内部的 print 统一输出到文件
    setup_logging("outputs/search-r1/infra.log")

    # 1. 加载模型
    hf_model = HFModel(model_name='Qwen/Qwen2.5-0.5B', device=DEVICE, use_qlora=True)
    model = hf_model.model()
    tokenizer = hf_model.tokenizer()

    # 2. 加载数据集
    dataset = NqAndHotpotqaDataset(batch_size=BATCH_SIZE)

    # 3. 建立环境
    tool_router = TagRouter(tools=[SearchTool()])
    env = Environment(
        max_turns=MAX_TURNS, 
        tool_router=tool_router, 
    )

    # 4. 构建奖励计算器
    reward_fn = SearchR1Reward()

    # 5. 构建训练器
    trainer = GRPOTrainer(
        base_model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        env=env,
        reward_fn=reward_fn,
        config=GRPOConfig(
            ckpt_config=CkptConfig(
                ckpt_dir="outputs/search-r1/ckpt",
                save_steps=5,
                keep_last=3,
            ),

            # 采样
            n_sample_group=4,
            inference_backend_config=InferenceBackendConfig(
                params=SamplingParams(max_new_tokens=256, max_turns=MAX_TURNS),
                backend=HFBackendConfig()
            ),

            # 训练
            n_epoch=2,
            learning_rate=1e-6,
            kl_beta=0,
            ratio_clip_eps=0.2,
            grad_clip_eps=1.0,

            # LoRA
            actor_lora_config=LoraConfig(
                alpha=16,
                rank=8,
                dropout=0.05,
                use_qlora=True,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),

            device=DEVICE,
        ),
    )

    # 6. 训练
    trainer.train(save="outputs/search-r1/train_loss.png", show=False)

    # 7. 评估
    trainer.eval(save="outputs/search-r1/eval_acc.png", show=False)


if __name__ == '__main__':
    main()
