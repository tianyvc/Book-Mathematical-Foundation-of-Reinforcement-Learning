"""第 1 周参考实现：状态索引和网格环境的一步转移。"""

# 参考实现中的常量与仓库 Python 默认参数保持一致。
ENV_SIZE = (5, 5)
TARGET_STATE = (4, 4)
FORBIDDEN_STATES = frozenset({(2, 1), (3, 3), (1, 3)})
REWARD_TARGET = 10.0
REWARD_FORBIDDEN = -5.0
REWARD_STEP = -1.0


def state_to_index(state, width):
    """把二维坐标映射成一维索引。

    仓库约定 x 是列坐标、y 是行坐标，索引按行优先排列：
    第 y 行前面有 y*width 个格子，当前行再偏移 x 个格子。
    例如 width=5 时，(2,2) -> 2*5+2=12。
    """
    x, y = state
    if width <= 0:
        raise ValueError("width 必须为正数")
    if not (0 <= x < width and 0 <= y):
        raise ValueError(f"坐标超出索引范围: state={state}, width={width}")
    return y * width + x


def index_to_state(index, width):
    """把一维索引还原为 (x,y)。

    整除得到行号 y，取模得到该行内的列号 x；它是
    state_to_index 的逆运算。
    """
    if width <= 0:
        raise ValueError("width 必须为正数")
    if index < 0:
        raise ValueError("index 不能为负数")
    return (index % width, index // width)


def transition(state, action):
    """执行默认 5x5 GridWorld 的一步转移。

    注意原仓库 Python 环境的规则：
    1. 越界时留在原状态，奖励 -5；
    2. 尝试进入禁区时也留在原状态，奖励 -5；
    3. 进入目标得到 +10，并报告 terminated=True；
    4. 其他有效移动（包括 stay）得到 -1；
    5. 本函数不加入 grid_world.py 为了画轨迹而加入的高斯扰动。

    练习的常见错误是只检查越界而忘记检查禁区，导致禁区被当成普通格。
    """
    if len(state) != 2 or len(action) != 2:
        raise ValueError("state 和 action 都必须是长度为 2 的序列")
    x, y = state
    dx, dy = action
    candidate = (x + dx, y + dy)

    # 边界判断要先做：越界时下一状态是原 state，而不是裁剪后的 candidate。
    outside = not (0 <= candidate[0] < ENV_SIZE[0] and 0 <= candidate[1] < ENV_SIZE[1])
    if outside or candidate in FORBIDDEN_STATES:
        return state, REWARD_FORBIDDEN, False

    if candidate == TARGET_STATE:
        return candidate, REWARD_TARGET, True

    return candidate, REWARD_STEP, False
