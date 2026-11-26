# practice_reference：带中文注释的参考实现

这个目录与 `practice/` 分开，保留你的练习代码不被覆盖。建议顺序是：先在 `practice/` 独立实现；检查失败后只打开对应的参考文件，定位思路并回到自己的文件修改。

从仓库根目录运行下列命令。依赖与主练习相同：NumPy、Matplotlib；第 8 周检查和实验还需 PyTorch。先激活已安装这些依赖的环境，或按[指南环境准备](../巩固强化指南.md#53-环境准备)创建环境。可用 `python -c "import numpy, matplotlib, torch; print(numpy.__version__, torch.__version__)"` 检查；若提示缺少模块，需在当前环境安装对应依赖。

```bash
python -m practice_reference.check 1
python -m practice_reference.check all
python -m practice_reference.demo
```

这里提供 1—11 周检查器所需核心函数的完整实现，并提供最小可运行演示。它用于解释公式、数组形状、终止处理和边界条件；完整的多种子训练、画图、调参和实验报告仍应在 `practice/experiments/` 中自己完成。

不要直接复制整个文件作为提交答案：先逐函数理解输入、输出、更新量和为什么这样处理终止。

## 逐周运行参考实验

```bash
python -m practice_reference.run 1 --steps 100
python -m practice_reference.run 3 --steps 10000
python -m practice_reference.run 6 --steps 10000
python -m practice_reference.run 8 --steps 50000
```

把周数换为 1—12，可运行该周一个代表性示范；支持 `--seed`、`--steps` 和 `--output`。结果默认写在 `practice_reference/results/`，不会写入你的 `practice/results/`。第 8 周需要 PyTorch，第一次运行可能较慢。

| 文件 | 可参考的内容 |
| --- | --- |
| `week01.py`—`week11.py` | 对应练习文件中全部必填函数的参考答案 |
| `grid_adapter.py` | 五个动作、真正终止和时间截断 |
| `dp.py` | 建模、策略评估、值迭代、策略迭代、截断策略迭代 |
| `monte_carlo.py`、`td.py` | MC 评估与控制、TD(0)、Sarsa、Expected Sarsa、n-step 和 Q-learning |
| `linear_value.py`、`stochastic_approximation.py` | 特征与线性 TD、增量均值和 SGD |
| `dqn.py` | Q 网络、经验回放、目标网络和训练循环 |
| `policy_gradient.py`、`actor_critic.py` | 表格 REINFORCE 和一步 Actor-Critic |
| `importance_sampling.py` | 普通、自归一化和裁剪的重要性采样 |
| `run.py`、`experiments/weekNN.py` | 各周可执行的代表性实验入口 |

**范围说明：** 1—11 周练习骨架的函数均已实现且可用原检查器核对；12 个实验入口均可运行。每周实验是教学起步示例，没有自动完成指南中所有额外扩展，例如完整的多种子比较、所有消融、梯度有限差分报告、完整 QAC/DDPG 和第 12 周综合项目报告；这些仍需按指南自行实现。实验的随机表现也不保证每个预算和种子都收敛。
