"""第 10 周参考实现：TD advantage 和 GAE。"""


def td_advantage(reward, value, next_value, gamma, terminated):
    """用 TD error 估计 advantage；terminated 时不 bootstrap。"""
    return reward + gamma * (0.0 if terminated else next_value) - value


def gae(deltas, continuation, gamma, lam):
    """从后往前计算 GAE。

    continuation[t]=0 表示当前步之后不能把后续 GAE 接回来，例如
    到达真正终止状态或跨越一个不连续的 rollout 边界。
    """
    if len(deltas) != len(continuation):
        raise ValueError("deltas 和 continuation 长度必须相同")
    result = [0.0] * len(deltas)
    running = 0.0
    for t in range(len(deltas) - 1, -1, -1):
        running = deltas[t] + gamma * lam * continuation[t] * running
        result[t] = running
    return result
