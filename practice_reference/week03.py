"""第 3 周参考实现：同步值迭代。"""
import numpy as np


def value_iteration(p, rewards, gamma, tolerance=1e-10):
    """求解 Bellman 最优方程的值迭代。

    p[s,a,s_next] 是模型转移概率，rewards[s,a] 是期望即时奖励。
    每轮先用旧 V 计算全部 Q，再沿动作维取 max；这样是同步更新。
    target 状态的零奖励吸收约定应在 p/rewards 建模阶段体现。
    """
    p = np.asarray(p, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    if p.ndim != 3 or rewards.shape != p.shape[:2] or p.shape[0] != p.shape[2]:
        raise ValueError("p 应为 [S,A,S]，rewards 应为 [S,A]")
    if not (0 <= gamma < 1):
        raise ValueError("本参考实现要求 0 <= gamma < 1")
    values = np.zeros(p.shape[0], dtype=float)
    for _ in range(100000):
        action_values = rewards + gamma * np.einsum("sak,k->sa", p, values)
        updated = action_values.max(axis=1)
        if np.max(np.abs(updated - values)) < tolerance:
            return updated
        values = updated
    raise RuntimeError("值迭代在 100000 轮内没有收敛")
