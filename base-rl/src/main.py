# main.py
from algorithms.manager import AgentType, create_agent
from trainer import Trainer
from evaluator import Evaluator
import gymnasium as gym


if __name__ == "__main__":
    # 1. 创建环境
    SEED = 42
    env = gym.make("FrozenLake-v1", is_slippery=False)

    # 2. 创建算法
    agent = create_agent(agent_type=AgentType.Q_LEARNING, env=env)

    # 3. 创建 Trainer
    trainer = Trainer(
        agent=agent,
        train_env=env,
        seed=SEED,
        n_episodes=10000,
    )

    # 4. 训练
    trainer.train(action_names=["左", "下", "右", "上"])

    # 5. 评估
    eval_env = gym.make("FrozenLake-v1", is_slippery=False)
    evaluator = Evaluator(env=eval_env, n_episodes=100, seed=SEED)
    evaluator.evaluate(agent)
