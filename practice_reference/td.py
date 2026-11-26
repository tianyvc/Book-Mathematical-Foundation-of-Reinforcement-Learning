"""表格 TD：固定策略评估、Sarsa、Expected Sarsa、Q-learning 与 n-step Sarsa。"""
import numpy as np
from .grid_adapter import GridAdapter, ACTIONS
from .week01 import state_to_index
from .monte_carlo import epsilon_policy


def td_evaluate(steps=10000, seed=0, gamma=.9, alpha=.1):
    """用均匀策略采样并执行 TD(0)，返回价值、访问次数和截断次数。"""
    rng, env = np.random.default_rng(seed), GridAdapter()
    state, _ = env.reset()
    values, counts = np.zeros(25), np.zeros(25, dtype=int)
    truncations = 0
    for _ in range(steps):
        index = state_to_index(state, 5)
        action = int(rng.integers(len(ACTIONS)))
        nxt, reward, terminated, truncated, _ = env.step(action)
        next_index = state_to_index(nxt, 5)
        target = reward + gamma * (not terminated) * values[next_index]
        values[index] += alpha * (target - values[index])
        counts[index] += 1
        truncations += int(truncated)
        state = env.reset()[0] if terminated or truncated else nxt
    return values, counts, truncations


def control(method='q_learning', steps=10000, seed=0, gamma=.9, alpha=.1, epsilon=.1, n=3):
    """在固定交互预算下训练表格控制方法，返回 Q、访问次数和截断次数。

    n-step 缓冲区以实际交互逐步收集；回合末尾逐项冲刷。时间截断
    时保留 Sarsa 尾部 bootstrap，而真正终止时尾部为零。
    """
    if method not in ('sarsa', 'expected_sarsa', 'q_learning', 'n_step_sarsa'):
        raise ValueError('未知控制算法')
    if n < 1:
        raise ValueError('n 必须为正整数')
    rng, env = np.random.default_rng(seed), GridAdapter()
    q = np.zeros((25, len(ACTIONS)))
    counts = np.zeros_like(q, dtype=int)
    state, _ = env.reset()
    s = state_to_index(state, 5)
    action = int(rng.choice(len(ACTIONS), p=epsilon_policy(q[s], epsilon)))
    pending = []
    truncations = 0

    def update_prefix(length, bootstrap):
        """更新队首状态动作；bootstrap 为 n 步后或截断边界的 Q 估计。"""
        target = bootstrap
        for entry in reversed(pending[:length]):
            target = entry[2] + gamma * target
        old_s, old_a = pending[0][:2]
        q[old_s, old_a] += alpha * (target - q[old_s, old_a])
        pending.pop(0)

    for _ in range(steps):
        nxt, reward, terminated, truncated, _ = env.step(action)
        ns = state_to_index(nxt, 5)
        counts[s, action] += 1
        probs = epsilon_policy(q[ns], epsilon)
        next_action = int(rng.choice(len(ACTIONS), p=probs)) if not terminated else 0
        if method == 'n_step_sarsa':
            pending.append((s, action, reward))
            if terminated or truncated:
                tail = 0.0 if terminated else q[ns, next_action]
                while pending:
                    update_prefix(len(pending), tail)
            elif len(pending) >= n:
                update_prefix(n, q[ns, next_action])
        else:
            if method == 'sarsa':
                future = q[ns, next_action]
            elif method == 'expected_sarsa':
                future = np.dot(probs, q[ns])
            else:
                future = np.max(q[ns])
            target = reward + gamma * (not terminated) * future
            q[s, action] += alpha * (target - q[s, action])
        if terminated or truncated:
            truncations += int(truncated)
            state, _ = env.reset()
            s = state_to_index(state, 5)
            action = int(rng.choice(len(ACTIONS), p=epsilon_policy(q[s], epsilon)))
        else:
            s, action = ns, next_action
    # 预算结束也属于采样片段边界；尾部以最后观测状态 bootstrap。
    if method == 'n_step_sarsa':
        tail = q[s, action]
        while pending:
            update_prefix(len(pending), tail)
    return q, counts, truncations


def evaluate_greedy(q, episodes=20, seed=1000, gamma=.9):
    """冻结 Q，独立评估贪心策略；返回平均折扣回报和成功率。"""
    env = GridAdapter()
    returns, successes = [], []
    for _ in range(episodes):
        state, _ = env.reset()
        total, factor = 0., 1.
        while True:
            s = state_to_index(state, 5)
            action = int(np.argmax(q[s]))
            state, reward, done, truncated, _ = env.step(action)
            total += factor * reward
            factor *= gamma
            if done or truncated:
                returns.append(total)
                successes.append(done)
                break
    return float(np.mean(returns)), float(np.mean(successes))
