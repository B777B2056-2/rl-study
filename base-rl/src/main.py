# main.py
from algorithms.manager import AgentType, create_agent
from env_factory import EnvType, create_env
from trainer import Trainer
from evaluator import Evaluator


def q_learning():
    # 1. 创建环境
    SEED = 42
    env = create_env(env_type=EnvType.FrozenLake, is_slippery=False)

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
    trainer.train(action_names=["左", "下", "右", "上"], save_path='outputs/qlearning/train.png')

    # 5. 评估
    eval_env = create_env(env_type=EnvType.FrozenLake, is_slippery=False)
    evaluator = Evaluator(env=eval_env, n_episodes=100, seed=SEED)
    evaluator.evaluate(agent, save_path='outputs/qlearning/eval.png')


def sarsa():
    # 1. 创建环境
    SEED = 42
    env = create_env(env_type=EnvType.FrozenLake, is_slippery=False)

    # 2. 创建算法
    agent = create_agent(agent_type=AgentType.SARSA, env=env)

    # 3. 创建 Trainer
    trainer = Trainer(
        agent=agent,
        train_env=env,
        seed=SEED,
        n_episodes=10000,
    )

    # 4. 训练
    trainer.train(action_names=["左", "下", "右", "上"], save_path='outputs/sarsa/train.png')

    # 5. 评估
    eval_env = create_env(env_type=EnvType.FrozenLake, is_slippery=False)
    evaluator = Evaluator(env=eval_env, n_episodes=100, seed=SEED)
    evaluator.evaluate(agent, save_path='outputs/sarsa/eval.png')

def dqn():
    # 1. 创建环境
    SEED = 42
    env = create_env(env_type=EnvType.CartPoleV1, is_slippery=False)

    # 2. 创建算法
    agent = create_agent(agent_type=AgentType.DQN, env=env)

    # 3. 创建 Trainer
    trainer = Trainer(
        agent=agent,
        train_env=env,
        seed=SEED,
        n_episodes=500,
    )

    # 4. 训练
    trainer.train(action_names=["左", "右"], save_path='outputs/dqn/train.png')

    # 5. 评估
    eval_env = create_env(env_type=EnvType.CartPoleV1, is_slippery=False)
    evaluator = Evaluator(env=eval_env, n_episodes=100, seed=SEED)
    evaluator.evaluate(agent, save_path='outputs/dqn/eval.png')

def ppo():
    # 1. 创建环境
    SEED = 42
    env = create_env(env_type=EnvType.CartPoleV1, is_slippery=False)

    # 2. 创建算法
    agent = create_agent(agent_type=AgentType.PPO, env=env)

    # 3. 创建 Trainer
    trainer = Trainer(
        agent=agent,
        train_env=env,
        seed=SEED,
        n_episodes=500,
    )

    # 4. 训练
    trainer.train(action_names=["左", "右"], save_path='outputs/ppo/train.png')

    # 5. 评估
    eval_env = create_env(env_type=EnvType.CartPoleV1, is_slippery=False)
    evaluator = Evaluator(env=eval_env, n_episodes=100, seed=SEED)
    evaluator.evaluate(agent, save_path='outputs/ppo/eval.png')

if __name__ == "__main__":
    # q_learning()
    # sarsa()
    # dqn()
    ppo()
