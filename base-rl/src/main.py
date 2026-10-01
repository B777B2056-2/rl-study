# main.py
from algorithms.manager import AgentType, create_agent
from trainer import Trainer
import gymnasium as gym


if __name__ == "__main__":
    # 1. 创建环境
    SEED = 42
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    eval_env = gym.make("FrozenLake-v1", is_slippery=False)

    # 2. 创建算法
    agent = create_agent(agent_type=AgentType.Q_LEARNING, env=env)

    # 3. 创建 Trainer
    trainer = Trainer(
        agent=agent,
        eval_env=eval_env,
        n_episodes=10000,
        eval_interval=500,
        success_fn=lambda r, l, info: r > 0,
    )

    # 4. 训练 + 出图
    trainer.train(
        save_path="frozenlake_result.png",
        action_names=["左", "下", "右", "上"],
    )