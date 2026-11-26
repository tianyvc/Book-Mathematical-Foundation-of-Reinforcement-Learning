"""第 11 周参考实现：普通重要性采样和连续 bandit 梯度。"""


def importance_estimate(actions, behavior, target, values):
    """用行为策略样本估计目标策略下的期望。

    每个样本只取实际 action 的 pi(a)/beta(a)；若 beta(a)=0 而目标
    概率非零，目标分布没有被行为数据覆盖，直接报告错误更安全。
    """
    if len(actions) == 0:
        raise ValueError("至少需要一个样本")
    if len(behavior) != len(target) or len(target) != len(values):
        raise ValueError("behavior、target 和 values 长度必须相同")
    # 支持集是对全部可能动作的要求，不仅仅是对本次抽到的动作检查。
    if any(p > 0 and b <= 0 for b, p in zip(behavior, target)):
        raise ValueError("行为策略没有覆盖目标策略的动作")
    weighted_values = []
    for action in actions:
        behavior_probability = behavior[action]
        target_probability = target[action]
        if behavior_probability <= 0:
            if target_probability > 0:
                raise ValueError("行为策略没有覆盖目标策略的动作")
            weight = 0.0
        else:
            weight = target_probability / behavior_probability
        weighted_values.append(weight * values[action])
    return sum(weighted_values) / len(weighted_values)


def bandit_gradient(theta):
    """J(theta)=-(theta-0.7)^2 的解析梯度。"""
    return -2.0 * (theta - 0.7)
