# 强化学习逐周练习

从仓库根目录打开[巩固强化指南](../巩固强化指南.md)，它包含阅读顺序、逐周代码要求、实验步骤和预期输出。这里的 `week01.py` 至 `week11.py` 以及 `experiments/week01.py` 至 `week12.py` 是需要你填写的练习骨架；初次运行会报 `NotImplementedError`。

```bash
# 在已安装 numpy、matplotlib 的 Python 环境中，从仓库根目录执行
python -m practice.reference
python -m practice.check 1
```

完成第 1 周的函数和实验后：

```bash
python -m practice.check 1
python -m practice.run 1 --seed 0 --steps 100 --output practice/results/week01/seed0
```

第 8 周检查需要 PyTorch。全部练习完成后，`python -m practice.check all` 检查第 1—11 周的关键确定性行为；实验质量与第 12 周综合项目仍需按照指南独立验收。
