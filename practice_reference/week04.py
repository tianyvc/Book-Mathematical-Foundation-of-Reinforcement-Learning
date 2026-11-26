"""第 4 周参考实现：折扣回报和 first-visit 记录。"""


def discounted_returns(rewards, gamma):
    """从后往前递推 reward-to-go，最后返回时间顺序。

    先把 running_return 设为 0，因为回合末尾之后没有未来奖励；
    每次把当前奖励加到 gamma 倍的后缀回报上。
    """
    if gamma < 0:
        raise ValueError("gamma 不能为负")
    result = [0.0] * len(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        result[t] = running
    return result


def first_visit_returns(state_actions, returns):
    """记录每个状态动作对在一条轨迹中第一次出现时的回报。

    这里必须从前往后遍历；如果从后往前遍历，第一次遇到的其实是
    时间上最后一次访问，容易误写成 last-visit。
    """
    if len(state_actions) != len(returns):
        raise ValueError("state_actions 和 returns 长度必须相同")
    result = {}
    for pair, value in zip(state_actions, returns):
        if pair not in result:
            result[pair] = value
    return result
