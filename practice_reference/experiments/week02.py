"""第 2 周参考实验：调用统一实现。"""
from ..run import run_week


def run(seed, steps, output):
    """运行第 2 周示范并保存结果；更完整的消融和报告可自行扩展。"""
    return run_week(2, seed, steps, output)
