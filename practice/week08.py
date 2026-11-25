"""Week 8: batched DQN targets with PyTorch."""


def dqn_targets(rewards, next_q, terminated, gamma):
    """Return detached Tensor[B] targets from next_q Tensor[B,A]."""
    raise NotImplementedError("Complete dqn_targets for week 8")
