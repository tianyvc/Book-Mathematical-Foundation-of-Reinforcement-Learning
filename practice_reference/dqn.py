"""最小 DQN：回放缓冲区、目标网络和冻结评估。"""
from collections import deque
import random
import numpy as np
import torch
from torch import nn
from .grid_adapter import GridAdapter, ACTIONS
from .week01 import state_to_index
from .week08 import dqn_targets


def encode(state):
    """将网格坐标缩放到 [0,1]，作为神经网络的两个输入。"""
    return np.asarray(state, dtype=np.float32) / 4.


class QNetwork(nn.Module):
    """输入二维坐标，输出五个动作的 Q 值。"""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, states):
        """前向传播；states 形状为 [B,2]，输出为 [B,5]。"""
        return self.network(states)


class ReplayBuffer:
    """保存最近的转移；使用独立 RNG 进行均匀抽样。"""
    def __init__(self, capacity=10000, seed=0):
        self.data = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, state, action, reward, next_state, terminated, truncated):
        """保存完整 transition，截断信息单独保存。"""
        self.data.append((state, action, reward, next_state, terminated, truncated))

    def sample(self, batch_size):
        """抽取一批 transition，逐列组织为张量。"""
        batch = self.rng.sample(list(self.data), batch_size)
        return self._tensor_batch(batch)

    def latest(self):
        """返回最新一条转移，用于不使用 replay 的在线对照。"""
        return self._tensor_batch([self.data[-1]])

    @staticmethod
    def _tensor_batch(batch):
        """把 transition 列表整理为状态、动作、奖励等列张量。"""
        state, action, reward, nxt, done, truncated = zip(*batch)
        return (torch.tensor(np.array([encode(s) for s in state])), torch.tensor(action, dtype=torch.long),
                torch.tensor(reward, dtype=torch.float32), torch.tensor(np.array([encode(s) for s in nxt])),
                torch.tensor(done, dtype=torch.bool), torch.tensor(truncated, dtype=torch.bool))


def update(online, target, optimizer, batch, gamma=.9):
    """对一个 replay batch 执行一次 Q-learning 半梯度更新。

    batch 包含状态、动作、奖励、下一状态、终止和截断列。先用
    gather 选出实际动作的 Q(s,a)，目标网络计算 r+gamma max Q(s',a')；
    真正终止时去掉后项，截断时保留。目标不反传梯度，再最小化均方
    误差并返回损失标量。截断后的状态仍可用于估计后续价值。
    """
    states, actions, rewards, next_states, terminated, _ = batch
    selected = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        targets = dqn_targets(rewards, target(next_states), terminated, gamma)
    loss = nn.functional.mse_loss(selected, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def train(steps=10000, seed=0, gamma=.9, use_target=True, use_replay=True):
    """训练最小 DQN；开关用于对照目标网络和回放缓冲区。

    不使用 replay 时每步只用最新转移更新，batch 大小因此不同；比较
    结果时必须报告这项限制，不能解释成只改变样本相关性。
    先以 epsilon-greedy 交互并存入缓冲区；积累 1000 条后开始更新。
    使用目标网络时每 200 次更新同步权重。返回训练后的在线网络和
    周期性记录的 (交互步数, 更新次数, 损失, epsilon)。
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env, memory = GridAdapter(), ReplayBuffer(seed=seed)
    online, target = QNetwork(), QNetwork()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    state, _ = env.reset()
    updates, logs = 0, []
    for t in range(steps):
        epsilon = max(.05, 1. - .95 * t / 20000)
        if rng.random() < epsilon:
            action = int(rng.integers(len(ACTIONS)))
        else:
            with torch.no_grad():
                action = int(online(torch.tensor(encode(state))[None]).argmax(dim=1).item())
        nxt, reward, terminated, truncated, _ = env.step(action)
        memory.add(state, action, reward, nxt, terminated, truncated)
        state = env.reset()[0] if terminated or truncated else nxt
        if len(memory.data) >= 1000:
            batch = memory.sample(64) if use_replay else memory.latest()
            # 不使用目标网络时，直接取在线网络当前值，但仍停止 target 梯度。
            target_for_update = target if use_target else online
            loss = update(online, target_for_update, optimizer, batch, gamma)
            updates += 1
            if use_target and updates % 200 == 0:
                target.load_state_dict(online.state_dict())
            if updates % 500 == 0:
                logs.append((t + 1, updates, loss, epsilon))
    return online, logs


def evaluate(network, gamma=.9):
    """冻结网络、从固定起点运行贪心策略，返回折扣回报和是否到达目标。"""
    env = GridAdapter()
    state, _ = env.reset()
    total, factor = 0., 1.
    while True:
        with torch.no_grad():
            action = int(network(torch.tensor(encode(state))[None]).argmax(dim=1).item())
        state, reward, terminated, truncated, _ = env.step(action)
        total += factor * reward
        factor *= gamma
        if terminated or truncated:
            return total, terminated
