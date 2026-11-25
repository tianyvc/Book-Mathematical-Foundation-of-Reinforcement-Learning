"""Run deterministic checkpoints: python -m practice.check 1 (or all).

Checks exercise interfaces and important edge cases; a passing check does not
replace the weekly training experiment and its report.
"""

import argparse
import importlib
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Code for grid world" / "python_version"
sys.path.insert(0, str(SOURCE))


def original_environment():
    # The upstream environment parses sys.argv at import time.
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        from src.grid_world import GridWorld
    finally:
        sys.argv = original_argv
    return GridWorld()


def model():
    """Mode A: terminal and inaccessible cells are zero-reward absorbing."""
    env = original_environment()
    width, height = env.env_size
    p = np.zeros((width * height, len(env.action_space), width * height))
    r = np.zeros((width * height, len(env.action_space)))
    for i in range(width * height):
        state = (i % width, i // width)
        for a, action in enumerate(env.action_space):
            if state == env.target_state or state in env.forbidden_states:
                next_state, reward = state, 0.0
            else:
                next_state, reward = env._get_next_state_and_reward(state, action)
            j = next_state[1] * width + next_state[0]
            p[i, a, j], r[i, a] = 1.0, reward
    return env, p, r


def close(actual, expected, label, atol=1e-8):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=atol, rtol=0):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def check_01(m):
    env = original_environment()
    for y in range(5):
        for x in range(5):
            i = y * 5 + x
            assert m.state_to_index((x, y), 5) == i
            assert tuple(m.index_to_state(i, 5)) == (x, y)
            if (x, y) in env.forbidden_states or (x, y) == env.target_state:
                continue
            for action in env.action_space:
                env.reset()
                env.agent_state = (x, y)
                state, reward, done, _ = env.step(action)
                got = m.transition((x, y), action)
                assert tuple(got[0]) == state and got[1] == reward and bool(got[2]) == done, (
                    (x, y), action, got, (state, reward, done))
    print("  25 index round trips; 105 transitions agree with upstream")


def check_02(m):
    _, p, r = model()
    p_pi, r_pi = p.mean(axis=1), r.mean(axis=1)
    values = np.asarray(m.evaluate_policy(p_pi, r_pi, 0.9))
    assert values.shape == (25,), f"expected shape (25,), got {values.shape}"
    close(values[12], -18.38412377296025, "uniform start value")
    close(values - r_pi - 0.9 * p_pi @ values, np.zeros(25), "Bellman residual", 1e-9)
    print(f"  uniform policy V(2,2) = {values[12]:.8f}; Bellman residual < 1e-9")


def check_03(m):
    _, p, r = model()
    values = np.asarray(m.value_iteration(p, r, 0.9, tolerance=1e-12))
    assert values.shape == (25,), f"expected shape (25,), got {values.shape}"
    close(values[12], 4.58, "optimal start value")
    close(values[24], 0, "terminal value")
    target = np.max(r + 0.9 * np.einsum("sak,k->sa", p, values), axis=1)
    close(values, target, "optimality residual", 1e-9)
    print(f"  V*(2,2) = {values[12]:.2f}; V*(target) = {values[24]:.2f}")


def check_04(m):
    returns = m.discounted_returns([-1, -1, 10], 0.5)
    close(returns, [1, 4, 10], "reward-to-go")
    pairs = [("s", "stay"), ("s", "stay"), ("s", "right")]
    first = m.first_visit_returns(pairs, returns)
    assert set(first) == {pairs[0], pairs[2]}
    close([first[pairs[0]], first[pairs[2]]], [1, 10], "first visit")
    print("  returns = [1, 4, 10]; first repeated visit = 1")


def check_05(m):
    mean = 0.0
    for count, sample in enumerate([2, 4, 9], 1):
        mean = m.incremental_mean(mean, sample, count)
    close(mean, 5, "incremental mean")
    close([m.td_target(2, .9, 10, False), m.td_target(2, .9, 10, True)], [11, 2], "TD target")
    print("  incremental mean = 5; TD targets = [11, 2]")


def check_06(m):
    args = (1, .9, [1, 3, 2], 0, [.2, .3, .5])
    output = m.control_targets(*args, terminated=False)
    close([output["sarsa"], output["expected_sarsa"], output["q_learning"]],
          [1.9, 2.89, 3.7], "control targets")
    output = m.control_targets(*args, terminated=True)
    close([output["sarsa"], output["expected_sarsa"], output["q_learning"]], [1, 1, 1], "terminal targets")
    print("  Sarsa / Expected Sarsa / Q-learning = [1.9, 2.89, 3.7]")


def check_07(m):
    w, phi, next_phi = np.array([1., 2.]), np.array([1., 0.]), np.array([0., 1.])
    close(m.linear_td_update(w.copy(), phi, 1, next_phi, .9, .1, False), [1.18, 2], "linear TD")
    close(m.linear_td_update(w.copy(), phi, 1, next_phi, .9, .1, True), [1, 2], "terminal linear TD")
    print("  semi-gradient update = [1.18, 2.0]; terminal update = [1.0, 2.0]")


def check_08(m):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Week 8 needs PyTorch; install it in the selected Python environment") from exc
    rewards = torch.tensor([1., 2.])
    next_q = torch.tensor([[1., 3.], [10., 20.]], requires_grad=True)
    result = m.dqn_targets(rewards, next_q, torch.tensor([False, True]), .9)
    assert isinstance(result, torch.Tensor), "return a PyTorch Tensor"
    assert tuple(result.shape) == (2,), f"expected shape (2,), got {tuple(result.shape)}"
    assert not result.requires_grad, "targets must be detached from next_q"
    close(result.detach().cpu().numpy(), [3.7, 2], "DQN targets", atol=1e-6)
    print("  target shape = (2,); values = [3.7, 2.0]; target detached")


def check_09(m):
    close(m.log_softmax_gradient([0., 0., 0.], 1), [-1/3, 2/3, -1/3], "log-softmax gradient")
    close(m.reinforce_weights([-1, -1, 10], .5), [1, 2, 2.5], "discounted REINFORCE weights")
    print("  grad log pi = [-1/3, 2/3, -1/3]; weighted returns = [1, 2, 2.5]")


def check_10(m):
    close(m.td_advantage(1, 2, 4, .9, False), 2.6, "TD advantage")
    close(m.td_advantage(1, 2, 4, .9, True), -1, "terminal advantage")
    close(m.gae([1, 2, 3], [1, 0, 0], .9, .5), [1.9, 2, 3], "GAE boundary")
    print("  TD advantages = [2.6, -1]; GAE = [1.9, 2, 3]")


def check_11(m):
    close(m.importance_estimate([0, 1], [.5, .5], [.8, .2], [1, -1]), .6, "importance estimate")
    close(m.bandit_gradient(0.), 1.4, "bandit gradient")
    close(m.bandit_gradient(.7), 0., "bandit optimum")
    print("  IS estimate = 0.6; bandit gradient at theta=0 is 1.4")


CHECKS = {i: globals()[f"check_{i:02}"] for i in range(1, 12)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("week", help="week number 1–11, or all")
    args = parser.parse_args()
    if args.week == "all":
        weeks = list(CHECKS)
    else:
        try:
            weeks = [int(args.week)]
            if weeks[0] not in CHECKS:
                raise ValueError
        except ValueError:
            parser.error("week must be 1–11 or all")
    failed = []
    for week in weeks:
        print(f"Week {week:02}", flush=True)
        try:
            module = importlib.import_module(f"practice.week{week:02}")
            CHECKS[week](module)
        except Exception as exc:
            failed.append(week)
            print(f"  FAIL: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(file=sys.stdout)
        else:
            print("  PASS", flush=True)
    if failed:
        print("Incomplete or failed weeks:", ", ".join(map(str, failed)))
        return 1
    print("All requested checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
