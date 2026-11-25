"""Week 10: advantage and generalized advantage estimation."""


def td_advantage(reward, value, next_value, gamma, terminated):
    """Return one-step TD advantage."""
    raise NotImplementedError("Complete td_advantage for week 10")


def gae(deltas, continuation, gamma, lam):
    """Return GAE from deltas with a continuation mask per transition."""
    raise NotImplementedError("Complete gae for week 10")
