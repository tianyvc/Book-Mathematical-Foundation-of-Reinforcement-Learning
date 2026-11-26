"""第 8 周参考实现：DQN 的批量 target。"""


def dqn_targets(rewards, next_q, terminated, gamma):
    """由目标网络的下一状态 Q 值构造 detached target。

    next_q 形状 [B,A]，动作维是 dim=1；max 得到 [B]。
    `torch.no_grad` 确保目标不会向 target network 反传梯度。
    """
    import torch
    rewards = rewards.reshape(-1)
    terminated = terminated.reshape(-1).bool()
    if next_q.ndim != 2 or next_q.shape[0] != rewards.shape[0]:
        raise ValueError("next_q 应为 [B,A]，rewards 应为 [B]")
    with torch.no_grad():
        next_best = next_q.max(dim=1).values
        return (rewards + gamma * (~terminated).to(next_q.dtype) * next_best).detach()
