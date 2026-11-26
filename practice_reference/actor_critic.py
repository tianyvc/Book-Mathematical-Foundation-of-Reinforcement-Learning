"""表格 Actor-Critic：完整回合批次与一步 TD 优势。"""
import numpy as np
from .grid_adapter import GridAdapter, ACTIONS
from .week01 import state_to_index
from .policy_gradient import softmax


def train(steps=10000, seed=0, gamma=.9, actor_lr=.03, critic_lr=.05):
    """on-policy 一步 Actor-Critic；每回合收集后使用冻结策略批量更新。

    Actor 使用 gamma^t 的回合起点折扣权重；Critic 用半梯度目标。
    回合时间上限只截断采样，TD 目标保留下一状态价值。

    logits[s,a] 保存策略参数，values[s] 保存 Critic 估计。先冻结
    logits 采集片段，逐步计算 delta = r + gamma V(s') - V(s)，
    用 delta 更新 V；Actor 将 gamma^t * delta * (one_hot(a)-pi)
    累加后一次更新。返回策略参数、价值和交互统计。预算耗尽时
    最后片段也可能未结束，因此 episodes 统计的是采集片段数。
    """
    rng, env = np.random.default_rng(seed), GridAdapter()
    logits, values = np.zeros((25, len(ACTIONS))), np.zeros(25)
    used, episodes, truncated_count = 0, 0, 0
    while used < steps:
        state, _ = env.reset()
        trajectory = []
        while used < steps:
            s = state_to_index(state, 5)
            probs = softmax(logits[s])
            action = int(rng.choice(len(ACTIONS), p=probs))
            nxt, reward, terminated, truncated, _ = env.step(action)
            ns = state_to_index(nxt, 5)
            trajectory.append((s, action, reward, ns, terminated, probs))
            used += 1
            state = nxt
            if terminated or truncated:
                truncated_count += int(truncated)
                break
        # Actor 梯度在参数更新前集中计算，避免使用旧数据逐步改变采样策略。
        gradient = np.zeros_like(logits)
        for t, (s, action, reward, ns, terminated, probs) in enumerate(trajectory):
            delta = reward + gamma * (not terminated) * values[ns] - values[s]
            score = -probs.copy()
            score[action] += 1
            gradient[s] += gamma**t * delta * score
            values[s] += critic_lr * delta
        logits += actor_lr * gradient
        episodes += 1
    return logits, values, {'env_steps': used, 'episodes': episodes, 'truncated': truncated_count}


def train_qac(steps=10000, seed=0, gamma=.9, actor_lr=.03, critic_lr=.05):
    """表格 QAC：用 Sarsa 学习 Q Critic，并用它改进随机 Actor。

    为便于对照教材，这里每一步先计算下一动作和 TD target，再更新
    Q Critic，最后用更新后的 Q 值修改 Actor；截断保留 bootstrap。
    q 与 logits 均为 [状态数, 动作数]；Sarsa 目标用实际采样的
    下一动作，Actor 沿 gamma^t Q(s,a) grad log pi(a|s) 上升。
    返回训练后的 logits 和 q；常数学习率下不保证有限预算收敛。
    """
    rng, env = np.random.default_rng(seed), GridAdapter()
    logits = np.zeros((25, len(ACTIONS)))
    q = np.zeros_like(logits)
    state, _ = env.reset()
    s = state_to_index(state, 5)
    probs = softmax(logits[s])
    action = int(rng.choice(len(ACTIONS), p=probs))
    t = 0
    for _ in range(steps):
        # 采样动作时的概率用于 score；下一动作在本次 Actor 更新前采样。
        nxt, reward, terminated, truncated, _ = env.step(action)
        ns = state_to_index(nxt, 5)
        next_probs = softmax(logits[ns])
        next_action = int(rng.choice(len(ACTIONS), p=next_probs)) if not terminated else 0
        delta = reward + gamma * (not terminated) * q[ns, next_action] - q[s, action]
        q[s, action] += critic_lr * delta
        score = -probs.copy()
        score[action] += 1
        logits[s] += actor_lr * gamma**t * q[s, action] * score
        t += 1
        if terminated or truncated:
            state, _ = env.reset()
            s = state_to_index(state, 5)
            probs = softmax(logits[s])
            action = int(rng.choice(len(ACTIONS), p=probs))
            t = 0
        else:
            s, action = ns, next_action
            probs = next_probs
    return logits, q
