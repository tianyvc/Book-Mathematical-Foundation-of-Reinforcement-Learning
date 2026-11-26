"""统一网格环境接口：区分真正终止与采样截断。"""
from .week01 import ENV_SIZE, TARGET_STATE, FORBIDDEN_STATES, transition

# 动作顺序与上游 GridWorld、策略矩阵列顺序完全一致。
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]


class GridAdapter:
    """以整数动作交互的回合式网格环境。

    默认任务与原环境相同：首次到达目标终止；max_steps 表示采样上限，
    达到上限但未到目标只标记 truncated，不将其当成真正终止。
    """

    def __init__(self, start_state=(2, 2), max_steps=200):
        self.start_state = start_state
        self.max_steps = max_steps
        self.state = start_state
        self.steps = 0
        self.finished = False

    def reset(self, state=None):
        """重置回合；state 可用于 exploring starts，仅允许合法非终止格。"""
        state = self.start_state if state is None else tuple(state)
        x, y = state
        if not (0 <= x < ENV_SIZE[0] and 0 <= y < ENV_SIZE[1]) or state in FORBIDDEN_STATES or state == TARGET_STATE:
            raise ValueError(f"非法起始状态: {state}")
        self.state, self.steps, self.finished = state, 0, False
        return self.state, {}

    def step(self, action):
        """执行整数动作，返回 (state, reward, terminated, truncated, info)。"""
        if self.finished:
            raise RuntimeError("回合已结束，请先 reset")
        if not isinstance(action, int) or not 0 <= action < len(ACTIONS):
            raise ValueError("action 应为 0 到 4 的整数")
        self.state, reward, terminated = transition(self.state, ACTIONS[action])
        self.steps += 1
        truncated = self.steps >= self.max_steps and not terminated
        self.finished = terminated or truncated
        return self.state, reward, terminated, truncated, {}
