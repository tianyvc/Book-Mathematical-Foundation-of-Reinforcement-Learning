"""蒙特卡洛评估与控制：只通过环境交互收集训练数据。"""
import numpy as np
from .grid_adapter import GridAdapter, ACTIONS
from .week01 import state_to_index
from .week04 import discounted_returns


def epsilon_policy(q_row, epsilon):
    """构造 epsilon-greedy 概率，并把贪心质量均分给并列动作。"""
    probabilities = np.full(len(q_row), epsilon / len(q_row))
    best = np.flatnonzero(np.isclose(q_row, np.max(q_row)))
    probabilities[best] += (1 - epsilon) / len(best)
    return probabilities


def sample_episode(env, policy, rng, start=None, first_action=None):
    """采集一条轨迹；支持指定起点动作以研究 exploring starts。"""
    state, _ = env.reset(start)
    episode = []
    while True:
        index = state_to_index(state, 5)
        probabilities = policy[index]
        action = int(first_action) if not episode and first_action is not None else int(rng.choice(len(ACTIONS), p=probabilities))
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((index, action, reward, terminated, truncated))
        state = next_state
        if terminated or truncated:
            return episode


def mc_evaluate(policy, episodes=1000, seed=0, gamma=.9, first_visit=True):
    """固定策略 MC 状态评估；截断时忽略尾项，并单独返回截断数。"""
    rng, env = np.random.default_rng(seed), GridAdapter()
    values, counts = np.zeros(25), np.zeros(25, dtype=int)
    truncations = 0
    for _ in range(episodes):
        trajectory = sample_episode(env, policy, rng)
        truncations += int(trajectory[-1][4])
        returns = discounted_returns([item[2] for item in trajectory], gamma)
        seen = set()
        for (state, _, _, _, _), reward_to_go in zip(trajectory, returns):
            if first_visit and state in seen:
                continue
            seen.add(state)
            counts[state] += 1
            values[state] += (reward_to_go - values[state]) / counts[state]
    return values, counts, truncations


def mc_control(episodes=1000, seed=0, gamma=.9, epsilon=.1):
    """on-policy epsilon-greedy MC 控制；每回合结束后更新策略。"""
    rng, env = np.random.default_rng(seed), GridAdapter()
    q, counts = np.zeros((25, len(ACTIONS))), np.zeros((25, len(ACTIONS)), dtype=int)
    policy = np.full_like(q, 1 / len(ACTIONS))
    steps = 0
    for _ in range(episodes):
        trajectory = sample_episode(env, policy, rng)
        steps += len(trajectory)
        returns = discounted_returns([item[2] for item in trajectory], gamma)
        seen = set()
        for (state, action, _, _, _), value in zip(trajectory, returns):
            if (state, action) in seen:
                continue
            seen.add((state, action))
            counts[state, action] += 1
            q[state, action] += (value - q[state, action]) / counts[state, action]
        for state, _ in seen:
            policy[state] = epsilon_policy(q[state], epsilon)
    return q, counts, steps


def mc_exploring_starts(episodes=1000, seed=0, gamma=.9):
    """从随机合法状态和随机首动作启动，逐回合更新 first-visit Q。

    此函数需要环境支持人工设置起始位置，真实问题不一定具备该条件。
    策略在整条回合中保持固定；截断片段以零尾项近似，需注意其偏差。
    """
    from .week01 import FORBIDDEN_STATES, TARGET_STATE, index_to_state
    rng, env = np.random.default_rng(seed), GridAdapter()
    q = np.zeros((25, len(ACTIONS)))
    counts = np.zeros_like(q, dtype=int)
    policy = np.full_like(q, 1 / len(ACTIONS))
    legal = [index_to_state(i, 5) for i in range(25)
             if index_to_state(i, 5) not in FORBIDDEN_STATES and index_to_state(i, 5) != TARGET_STATE]
    for _ in range(episodes):
        start = legal[int(rng.integers(len(legal)))]
        first_action = int(rng.integers(len(ACTIONS)))
        trajectory = sample_episode(env, policy, rng, start, first_action)
        returns = discounted_returns([item[2] for item in trajectory], gamma)
        seen = set()
        for (state, action, _, _, _), value in zip(trajectory, returns):
            if (state, action) in seen:
                continue
            seen.add((state, action))
            counts[state, action] += 1
            q[state, action] += (value - q[state, action]) / counts[state, action]
        for state, _ in seen:
            policy[state] = epsilon_policy(q[state], 0.)
    return q, counts


def mc_basic(policy, rollouts_per_pair=10, seed=0, gamma=.9):
    """MC Basic 的固定策略评估步骤：对每个起始状态动作独立采样。

    外层控制可在评估结束后从 Q 贪心生成新策略，再重复本函数；
    故本函数仅完成单轮策略评估，不把它称为完整 MC Basic 控制。
    """
    from .week01 import FORBIDDEN_STATES, TARGET_STATE, index_to_state
    rng, env = np.random.default_rng(seed), GridAdapter()
    q = np.zeros((25, len(ACTIONS)))
    for s in range(25):
        state = index_to_state(s, 5)
        if state in FORBIDDEN_STATES or state == TARGET_STATE:
            continue
        for action in range(len(ACTIONS)):
            returns = []
            for _ in range(rollouts_per_pair):
                episode = sample_episode(env, policy, rng, state, action)
                returns.append(discounted_returns([item[2] for item in episode], gamma)[0])
            q[s, action] = np.mean(returns)
    return q
