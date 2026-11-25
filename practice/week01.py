"""Week 1: state indexing and the original GridWorld transition."""


def state_to_index(state, width):
    """Return the row-major index of (x, y)."""
    raise NotImplementedError("Complete state_to_index for week 1")


def index_to_state(index, width):
    """Return the (x, y) position for a row-major index."""
    raise NotImplementedError("Complete index_to_state for week 1")


def transition(state, action):
    """Return (next_state, reward, terminated) for original default GridWorld.

    state and action are coordinate tuples. Only nonterminal, legal states
    are used by the check; stop interactions when terminated becomes true.
    """
    raise NotImplementedError("Complete transition for week 1")
