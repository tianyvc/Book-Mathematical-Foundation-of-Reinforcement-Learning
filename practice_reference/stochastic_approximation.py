"""均值估计和一维随机梯度下降。"""
import numpy as np


def track_mean(samples, alpha=None):
    """alpha=None 时使用 1/n，否则使用常数步长，返回每一步估计。"""
    mean = 0.
    history = []
    for count, value in enumerate(samples, 1):
        rate = 1 / count if alpha is None else alpha
        mean += rate * (value - mean)
        history.append(mean)
    return np.asarray(history)


def linear_regression_sgd(xs, ys, alpha=.01, epochs=10):
    """拟合 y=w*x+b；按样本依次更新，并返回参数和损失历史。"""
    weights = np.zeros(2)
    history = []
    for _ in range(epochs):
        for x, y in zip(xs, ys):
            feature = np.array([x, 1.])
            error = weights @ feature - y
            weights -= alpha * error * feature
        history.append(float(np.mean((weights[0] * np.asarray(xs) + weights[1] - ys) ** 2)))
    return weights, history
