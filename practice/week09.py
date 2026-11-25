"""Week 9: policy gradients for categorical softmax actions."""


def log_softmax_gradient(logits, action):
    """Return d log pi(action) / d logits as a 1D array."""
    raise NotImplementedError("Complete log_softmax_gradient for week 9")


def reinforce_weights(rewards, gamma):
    """Return gamma**t * reward-to-go[t] for a complete trajectory."""
    raise NotImplementedError("Complete reinforce_weights for week 9")
