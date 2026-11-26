"""第 9 周参考实现：softmax log-gradient 和 REINFORCE 权重。"""
import numpy as np


def log_softmax_gradient(logits, action):
    """返回 d log pi(action)/d logits = one_hot(action)-softmax(logits)。

    先减去最大 logit 防止 exp 溢出；梯度各分量之和应为 0，
    反映所有 logits 同时平移不会改变 softmax 概率。
    """
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 1 or not (0 <= action < logits.size):
        raise ValueError("logits 应为一维，action 必须是合法下标")
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    gradient = -probabilities
    gradient[action] += 1.0
    return gradient


def reinforce_weights(rewards, gamma):
    """返回有限回合中每个时间步的 gamma^t * G_t。"""
    result = [0.0] * len(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        result[t] = (gamma ** t) * running
    return result
