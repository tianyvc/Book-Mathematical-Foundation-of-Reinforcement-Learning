"""第 7 周参考实现：线性函数逼近的半梯度 TD。"""
import numpy as np


def linear_td_update(weights, features, reward, next_features, gamma, alpha, terminated):
    """执行一次 w <- w + alpha * delta * phi(s)。

    预测值是 phi(s)@w；target 中的下一状态预测在本次更新中被视为
    常量，因此只沿当前特征 phi(s) 更新，这就是半梯度形式。
    """
    weights = np.asarray(weights, dtype=float)
    features = np.asarray(features, dtype=float)
    next_features = np.asarray(next_features, dtype=float)
    if weights.shape != features.shape or features.shape != next_features.shape:
        raise ValueError("weights、features、next_features 形状必须一致")
    current = float(features @ weights)
    mask = 0.0 if terminated else 1.0
    target = reward + gamma * mask * float(next_features @ weights)
    delta = target - current
    return weights + alpha * delta * features
