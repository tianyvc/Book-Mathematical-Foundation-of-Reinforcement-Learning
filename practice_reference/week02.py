"""第 2 周参考实现：固定策略的矩阵策略评估。"""
import numpy as np


def evaluate_policy(p_pi, r_pi, gamma):
    """求解 v = r + gamma P v。

    p_pi 的形状是 [状态数, 状态数]，r_pi 是 [状态数]。使用
    solve 而非显式求逆，既更稳定也更符合数值线性代数的习惯。
    返回一维数组，便于与后续的价值向量和检查器对接。
    """
    p_pi = np.asarray(p_pi, dtype=float)
    r_pi = np.asarray(r_pi, dtype=float).reshape(-1)
    if p_pi.ndim != 2 or p_pi.shape[0] != p_pi.shape[1] or p_pi.shape[0] != r_pi.size:
        raise ValueError("P_pi 必须为 [n,n]，r_pi 必须为 [n]")
    if not (0 <= gamma < 1):
        raise ValueError("本参考实现要求 0 <= gamma < 1")
    system = np.eye(r_pi.size) - gamma * p_pi
    return np.linalg.solve(system, r_pi)
