"""第 1 周参考实验：打印固定路径并保存 CSV。"""
import csv
import json
from pathlib import Path
from ..week01 import state_to_index, transition


def run(seed=0, steps=100, output=Path("practice_reference/results/week01")):
    """运行最小路径实验；seed/steps 保留统一接口，路径本身是确定性的。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({"seed": seed, "steps": steps}, indent=2), encoding="utf-8")
    state = (2, 2)
    actions = [(1, 0), (1, 0), (0, 1), (0, 1)]
    rows = [["step", "state", "index", "action", "next_state", "reward", "terminated"]]
    for step, action in enumerate(actions):
        next_state, reward, terminated = transition(state, action)
        rows.append([step, state, state_to_index(state, 5), action, next_state, reward, terminated])
        print(f"step {step}: state={state}, index={state_to_index(state,5)}, action={action}, next={next_state}, reward={reward}, terminated={terminated}")
        state = next_state
        if terminated:
            break
    # 除路径外，单独输出全部 25 个状态索引，方便检查映射是否覆盖全网格。
    with (output / "states.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x", "y", "index"])
        for y in range(5):
            for x in range(5):
                writer.writerow([x, y, state_to_index((x, y), 5)])
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)
    (output / "summary.md").write_text(
        "# 第 1 周参考实验\n\n固定路径为右、右、下、下；目标终止。\n",
        encoding="utf-8",
    )
    print(f"Saved metrics to {output / 'metrics.csv'}")
