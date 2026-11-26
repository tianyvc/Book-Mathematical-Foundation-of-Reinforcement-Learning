"""第 5 周参考实现：增量均值和 TD target。"""


def incremental_mean(old_mean, sample, count):
    """加入第 count 个样本后的新均值。

    count 从 1 开始；公式等价于完整样本平均，但不需要保存所有样本。
    """
    if count < 1:
        raise ValueError("count 必须从 1 开始")
    return old_mean + (sample - old_mean) / count


def td_target(reward, gamma, next_value, terminated):
    """计算一步 TD 目标。

    真正终止时未来价值为 0；注意 Python 中 bool 可参与数值运算，
    但显式使用 mask 更容易阅读和避免把 truncated 当成 terminated。
    """
    mask = 0.0 if terminated else 1.0
    return reward + gamma * mask * next_value
