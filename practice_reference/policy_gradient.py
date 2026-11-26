"""表格 softmax REINFORCE：使用固定回合起点折扣目标。"""
import numpy as np
from .grid_adapter import GridAdapter, ACTIONS
from .week01 import state_to_index
from .week04 import discounted_returns


def softmax(logits):
    """数值稳定的 softmax，输入和输出均为动作维向量。"""
    z = logits - np.max(logits)
    probabilities = np.exp(z)
    return probabilities / probabilities.sum()


def collect(logits, rng, max_steps=200):
    """冻结当前 logits 采样完整回合或截断片段，并记录状态、动作、奖励。"""
    env = GridAdapter(max_steps=max_steps)
    state, _ = env.reset()
    trajectory = []
    while True:
        s = state_to_index(state, 5)
        probabilities = softmax(logits[s])
        action = int(rng.choice(len(ACTIONS), p=probabilities))
        nxt, reward, terminated, truncated, _ = env.step(action)
        trajectory.append((s, action, reward))
        state = nxt
        if terminated or truncated:
            return trajectory, truncated


def batch_gradient(logits, trajectories, gamma=.9, baseline=None):
    """计算 REINFORCE 梯度：回合内求和、对回合平均。

    baseline 若提供则按状态索引读取，不对其反向传播。截断片段的尾部
    采用 0 是有限片段目标，会有相对无限任务的截断偏差，需记录。
    """
    gradient = np.zeros_like(logits)
    for trajectory in trajectories:
        returns = discounted_returns([item[2] for item in trajectory], gamma)
        for t, ((s, action, _), value) in enumerate(zip(trajectory, returns)):
            score = -softmax(logits[s])
            score[action] += 1
            advantage = value - (0. if baseline is None else baseline[s])
            gradient[s] += gamma**t * advantage * score
    return gradient / len(trajectories)


def train(steps=10000, seed=0, gamma=.9, alpha=.05, batch_size=10):
    """按批冻结策略采样并更新，直到环境交互步数达到或略超预算。"""
    rng = np.random.default_rng(seed)
    logits = np.zeros((25, len(ACTIONS)))
    steps_used, episodes, truncations = 0, 0, 0
    while steps_used < steps:
        batch = []
        for _ in range(batch_size):
            trajectory, truncated = collect(logits, rng)
            batch.append(trajectory)
            steps_used += len(trajectory)
            episodes += 1
            truncations += int(truncated)
            if steps_used >= steps:
                break
        logits += alpha * batch_gradient(logits, batch, gamma)
    return logits, {'env_steps': steps_used, 'episodes': episodes, 'truncated': truncations}
