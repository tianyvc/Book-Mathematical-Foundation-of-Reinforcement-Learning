"""第 6 周参考实现：三种一步控制 target。"""


def control_targets(reward, gamma, next_q, next_action, next_policy, terminated):
    """返回 Sarsa、Expected Sarsa 和 Q-learning 的 target。

    next_q 是下一状态各动作的 Q 值；next_action 是行为策略实际采样的
    下一动作；next_policy 是 Expected Sarsa 要使用的目标策略概率。
    """
    if len(next_q) != len(next_policy):
        raise ValueError("next_q 和 next_policy 长度必须相同")
    mask = 0.0 if terminated else 1.0
    expected = sum(q * probability for q, probability in zip(next_q, next_policy))
    return {
        "sarsa": reward + gamma * mask * next_q[next_action],
        "expected_sarsa": reward + gamma * mask * expected,
        "q_learning": reward + gamma * mask * max(next_q),
    }
