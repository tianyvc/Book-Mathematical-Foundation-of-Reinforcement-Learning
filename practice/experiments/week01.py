"""Week 1 experiment: implement run according to the weekly workbook."""
import csv
import pathlib as Path
from practice.week01 import state_to_index, index_to_state, transition

def run(seed, steps, output):
    """Run the experiment and write results to the pathlib.Path output directory."""
    output.mkdir(exist_ok=True, parents=True)
    csv_path = output / "metrics.csv"

    width = 5
    state = (2, 2)
    actions = [(1, 0), (1, 0), (0, 1), (0, 1)]

    rows = []
    rows.append(["step", "x", "y", "index", "reward", "terminated"])

    for step_idx, act in enumerate(actions):
        next_state, reward, terminated = transition(state, act)
        x, y = state
        idx = state_to_index(state, width)
        rows.append([step_idx, x, y, idx, reward, terminated])

        print(f"step {step_idx}: state={state}, index={idx}, reward={reward}, terminated={terminated}")

        state = next_state
        if terminated:
            break

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"\nSaved metrics to {csv_path}")
