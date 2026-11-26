"""模型法参考：构建 MDP、策略评估、VI、PI 和截断 PI。"""
import numpy as np
from .grid_adapter import ACTIONS
from .week01 import ENV_SIZE, TARGET_STATE, FORBIDDEN_STATES, transition, state_to_index
from .week02 import evaluate_policy
from .week03 import value_iteration


def build_model():
    """构造模式 A 的 P[S,A,S]、R[S,A] 和合法决策状态 mask。

    目标及不可达禁区保留为零奖励吸收行，方便与 25 维图形索引统一。
    model-free 训练函数只使用环境 step，不读取此模型。
    """
    width, height = ENV_SIZE
    n = width * height
    p = np.zeros((n, len(ACTIONS), n))
    r = np.zeros((n, len(ACTIONS)))
    mask = np.zeros(n, dtype=bool)
    for y in range(height):
        for x in range(width):
            state = (x, y)
            i = state_to_index(state, width)
            mask[i] = state not in FORBIDDEN_STATES and state != TARGET_STATE
            for action_id, action in enumerate(ACTIONS):
                if not mask[i]:
                    p[i, action_id, i] = 1.0
                    continue
                next_state, reward, _ = transition(state, action)
                j = state_to_index(next_state, width)
                p[i, action_id, j], r[i, action_id] = 1.0, reward
    return p, r, mask


def policy_model(p, rewards, policy):
    """按策略加权聚合，返回 P_pi[S,S] 和 r_pi[S]。"""
    return np.einsum('sa,sak->sk', policy, p), np.sum(policy * rewards, axis=1)


def iterative_evaluation(p_pi, r_pi, gamma=.9, tolerance=1e-10, max_iterations=100000):
    """同步迭代固定策略 Bellman 方程，返回价值和每轮更新幅度。"""
    values = np.zeros(len(r_pi))
    history = []
    for _ in range(max_iterations):
        updated = r_pi + gamma * p_pi @ values
        history.append(float(np.max(np.abs(updated - values))))
        values = updated
        if history[-1] < tolerance:
            return values, history
    raise RuntimeError('策略评估超出最大轮数')


def action_values(p, rewards, values, gamma=.9):
    """根据已知模型做一次前瞻，得到每个状态动作的 Q 值。"""
    return rewards + gamma * np.einsum('sak,k->sa', p, values)


def greedy_policy(q):
    """提取确定性贪心策略；并列最大时选择第一个，避免随机切换。"""
    policy = np.zeros_like(q, dtype=float)
    policy[np.arange(q.shape[0]), np.argmax(q, axis=1)] = 1.0
    return policy


def policy_iteration(p, rewards, gamma=.9, evaluation_steps=None):
    """完整 PI 或截断 PI；返回价值、策略和改进轮数。

    evaluation_steps=None 使用线性求解；正整数时每轮只做指定次数
    同步评估并继承上轮价值。截断时额外检查 Bellman 残差，不能只看
    策略是否暂时不变；最终返回前再精确评估已获得策略。
    """
    n, a = rewards.shape
    policy = np.ones((n, a)) / a
    values = np.zeros(n)
    for iteration in range(1, 10001):
        p_pi, r_pi = policy_model(p, rewards, policy)
        if evaluation_steps is None:
            values = evaluate_policy(p_pi, r_pi, gamma)
        else:
            for _ in range(evaluation_steps):
                values = r_pi + gamma * p_pi @ values
        updated = greedy_policy(action_values(p, rewards, values, gamma))
        stable = np.array_equal(updated, policy)
        if stable:
            p_new, r_new = policy_model(p, rewards, updated)
            exact = evaluate_policy(p_new, r_new, gamma)
            residual = np.max(np.abs(action_values(p, rewards, exact, gamma).max(axis=1) - exact))
            if residual < 1e-9:
                return exact, updated, iteration
        policy = updated
    raise RuntimeError('策略迭代未达到停止条件')
