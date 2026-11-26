"""GridWorld adapter: reusable wrapper mapping integer actions to delta tuples."""
from practice.week01 import transition

# 动作编码：0上，1右，2下，3左
ACTION_MAP = {
    0: (0, -1),  # up
    1: (1, 0),   # right
    2: (0, 1),   # down
    3: (-1, 0)   # left
}
REVERSE_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}


class GridAdapter:
    def __init__(self, start_state=(2,2), width=5):
        self.start_state = start_state
        self.width = width
        self.state = start_state
        self.terminated = False
        self.truncated = False

    def reset(self):
        self.state = self.start_state
        self.terminated = False
        self.truncated = False
        return self.state

    def step(self, action_int):
        """
        action_int: 0/1/2/3
        returns (next_state, reward, terminated, truncated, info)
        """
        delta = ACTION_MAP[action_int]
        next_state, reward, terminated = transition(self.state, delta)
        self.state = next_state
        self.terminated = terminated
        # truncated 这里默认False，留给外部步数上限控制
        info = {}
        return next_state, reward, terminated, self.truncated, info
