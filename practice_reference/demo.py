"""运行参考实现的几个最小例子。"""
from .week01 import transition
from .week04 import discounted_returns
from .week06 import control_targets


def main():
    print("进入禁区:", transition((2, 2), (0, -1)))
    print("到达目标:", transition((3, 4), (1, 0)))
    print("returns:", discounted_returns([-1, -1, 10], 0.5))
    print("targets:", control_targets(1, 0.9, [1, 3, 2], 0, [0.2, 0.3, 0.5], False))


if __name__ == "__main__":
    main()
