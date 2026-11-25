"""Week 5: stochastic approximation and TD target."""


def incremental_mean(old_mean, sample, count):
    """Update a mean when sample is observation number count (count >= 1)."""
    raise NotImplementedError("Complete incremental_mean for week 5")


def td_target(reward, gamma, next_value, terminated):
    """Return one-step target; terminated is a boolean."""
    raise NotImplementedError("Complete td_target for week 5")
