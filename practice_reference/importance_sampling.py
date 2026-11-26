"""重要性采样的普通、自归一化和裁剪估计。"""
import numpy as np


def estimates(actions, behavior, target, values, clip=None):
    """依据动作样本计算三种估计；严格检查目标支持集是否被行为覆盖。"""
    actions = np.asarray(actions, dtype=int)
    behavior, target, values = map(lambda x: np.asarray(x, dtype=float), (behavior, target, values))
    if np.any((target > 0) & (behavior <= 0)):
        raise ValueError('目标策略支持集未被行为策略覆盖')
    weights = target[actions] / behavior[actions]
    ordinary = np.mean(weights * values[actions])
    normalized = np.sum(weights * values[actions]) / np.sum(weights)
    clipped = ordinary if clip is None else np.mean(np.minimum(weights, clip) * values[actions])
    return {'ordinary': float(ordinary), 'normalized': float(normalized), 'clipped': float(clipped)}
