"""Print independent environment and analytic baselines for the exercises."""

import numpy as np

from practice.check import model


def main():
    env, p, rewards = model()
    values = np.zeros(env.num_states)
    for _ in range(10000):
        new_values = (rewards + .9 * np.einsum("sak,k->sa", p, values)).max(axis=1)
        if np.max(np.abs(new_values - values)) < 1e-12:
            values = new_values
            break
        values = new_values
    else:
        raise RuntimeError("Value iteration did not converge")
    p_pi = p.mean(axis=1)
    r_pi = rewards.mean(axis=1)
    uniform = np.linalg.solve(np.eye(env.num_states) - .9 * p_pi, r_pi)
    print(f"Grid: {env.env_size[0]}x{env.env_size[1]}, legal decision states: 21")
    print(f"Start: {env.start_state}; target: {env.target_state}; gamma: 0.9")
    print(f"Uniform V(start): {uniform[12]:.8f}")
    print(f"Optimal V(start): {values[12]:.2f}")
    print(f"Optimal V(target): {values[24]:.2f}")
    print("MC returns for [-1, -1, 10], gamma=0.5: [1, 4, 10]")
    print("One-step control targets: [1.9, 2.89, 3.7]")


if __name__ == "__main__":
    main()
