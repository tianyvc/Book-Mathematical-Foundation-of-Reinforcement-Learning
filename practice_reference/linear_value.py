"""线性价值函数的特征、最小二乘和半梯度 TD。"""
import numpy as np
from .week01 import index_to_state
from .week07 import linear_td_update


def features(index, kind='one_hot'):
    """生成 one-hot、归一化坐标或二阶多项式特征。"""
    if kind == 'one_hot':
        result = np.zeros(25)
        result[index] = 1.
        return result
    x, y = index_to_state(index, 5)
    x, y = x / 4., y / 4.
    if kind == 'coordinates':
        return np.array([1., x, y])
    if kind == 'quadratic':
        return np.array([1., x, y, x*x, x*y, y*y])
    raise ValueError('未知特征类型')


def fit_true_values(values, indices, kind='quadratic'):
    """用已知 DP 真值做监督最小二乘，仅用于诊断表示能力。"""
    design = np.array([features(i, kind) for i in indices])
    weights, *_ = np.linalg.lstsq(design, np.asarray(values)[indices], rcond=None)
    return weights, design @ weights


def update(weights, state, reward, next_state, gamma=.9, alpha=.01, terminated=False, kind='quadratic'):
    """将整数状态索引转换为特征，并执行一次半梯度 TD。"""
    return linear_td_update(weights, features(state, kind), reward, features(next_state, kind), gamma, alpha, terminated)
