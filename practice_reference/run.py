"""运行逐周参考实验：python -m practice_reference.run 2 --steps 1000。"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .dp import build_model, policy_model, iterative_evaluation, policy_iteration, value_iteration, greedy_policy, action_values
from .week02 import evaluate_policy


def save_results(output, config, rows, summary):
    """保存配置、统一格式的指标 CSV 和中文说明。"""
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    keys = sorted({key for row in rows for key in row}) if rows else ['message']
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.md').write_text(summary + '\n', encoding='utf-8')


def run_week(week, seed, steps, output):
    """按周运行代表性实验。各周完整拓展实验仍需要自行增加。"""
    rows = []
    summary = f'# 第 {week} 周参考实验\n\n这是一个起步示范，结果可能因种子与计算预算变化。'
    p, rewards, mask = build_model()
    uniform = np.full(rewards.shape, .2)
    p_pi, r_pi = policy_model(p, rewards, uniform)
    truth = evaluate_policy(p_pi, r_pi, .9)
    if week == 1:
        from .experiments.week01 import run
        run(seed, steps, output)
        return
    if week == 2:
        from .monte_carlo import mc_evaluate
        iterative, history = iterative_evaluation(p_pi, r_pi)
        mc, counts, truncated = mc_evaluate(uniform, episodes=max(1, steps // 50), seed=seed)
        rows = [{'method': 'exact', 'start_value': truth[12], 'residual': float(np.max(np.abs(truth-r_pi-.9*p_pi@truth)))},
                {'method': 'iterative', 'start_value': iterative[12], 'residual': history[-1]},
                {'method': 'mc', 'start_value': mc[12], 'visits': int(counts[12]), 'truncated': truncated}]
    elif week == 3:
        vi = value_iteration(p, rewards, .9)
        rows.append({'method':'VI','start_value':vi[12]})
        for n in (None, 1, 3, 10):
            values, _, rounds = policy_iteration(p, rewards, evaluation_steps=n)
            rows.append({'method': 'PI' if n is None else f'truncated_PI_{n}', 'start_value':values[12], 'rounds':rounds, 'max_error':float(np.max(np.abs(values-vi)))})
    elif week == 4:
        from .monte_carlo import mc_evaluate, mc_control
        for first in (True, False):
            values, counts, truncated = mc_evaluate(uniform, episodes=max(1, steps//100), seed=seed, first_visit=first)
            rows.append({'method':'first_visit' if first else 'every_visit','start_value':values[12], 'visits':int(counts[12]), 'truncated':truncated})
        q, counts, used = mc_control(episodes=max(1, steps//100), seed=seed)
        rows.append({'method':'mc_control','env_steps':used,'coverage':float(np.mean(counts[mask]>0))})
        summary += '\n\nMC 的运行参数这里是回合数近似，实际交互步数见 env_steps；达到时间上限的片段忽略尾部。'
    elif week == 5:
        from .stochastic_approximation import track_mean
        from .td import td_evaluate
        rng = np.random.default_rng(seed)
        samples = np.r_[rng.normal(0,1,500), rng.normal(3,1,500)]
        mean = track_mean(samples)
        constant = track_mean(samples, .05)
        values, counts, truncated = td_evaluate(steps, seed)
        rows = [{'method':'mean_1_over_n','estimate':float(mean[-1])}, {'method':'mean_constant','estimate':float(constant[-1])},
                {'method':'TD0','estimate':values[12], 'rmse':float(np.sqrt(np.mean((values[mask]-truth[mask])**2))), 'truncated':truncated}]
    elif week == 6:
        from .td import control, evaluate_greedy
        for method in ['sarsa','expected_sarsa','q_learning','n_step_sarsa']:
            q, counts, truncated = control(method, steps, seed)
            value, success = evaluate_greedy(q)
            rows.append({'method':method,'env_steps':steps,'greedy_return':value,'success':success,'coverage':float(np.mean(counts[mask]>0)),'truncated':truncated})
    elif week == 7:
        from .linear_value import fit_true_values
        indices = np.flatnonzero(mask)
        for kind in ('one_hot','coordinates','quadratic'):
            weights, predictions = fit_true_values(truth, indices, kind)
            rows.append({'features':kind,'rmse':float(np.sqrt(np.mean((predictions-truth[indices])**2))), 'dimensions':len(weights)})
        summary += '\n\n该示范是用已知真值检查表示能力；请另实现并比较固定策略下的线性 TD。'
    elif week == 8:
        from .dqn import train, evaluate
        network, history = train(steps, seed)
        value, success = evaluate(network)
        rows = [{'env_steps':t,'updates':u,'loss':loss,'epsilon':eps} for t,u,loss,eps in history]
        rows.append({'env_steps':steps,'eval_return':value,'success':int(success)})
        summary += '\n\n只提供最小 DQN；目标网络和 replay 的消融需在独立实验中完成。'
    elif week == 9:
        from .policy_gradient import train
        logits, info = train(steps, seed)
        rows = [dict(info, method='REINFORCE', start_greedy_action=int(np.argmax(logits[12])))]
        summary += '\n\n初始批次采用纯回报；baseline 方差对照和有限差分推导可按指南继续扩展。'
    elif week == 10:
        from .actor_critic import train
        logits, values, info = train(steps, seed)
        rows = [dict(info, method='episode_batch_actor_critic', start_value=values[12], start_greedy_action=int(np.argmax(logits[12])))]
        summary += '\n\n此示范按回合批量更新，是一步 TD Actor-Critic，不是并行采样的标准 A2C。'
    elif week == 11:
        from .importance_sampling import estimates
        rng = np.random.default_rng(seed)
        for beta in ([.5,.5],[.05,.95]):
            actions = rng.choice(2,size=steps,p=beta)
            rows.append(dict(behavior=str(beta), **estimates(actions,beta,[.8,.2],[1.,-1.],clip=5)))
    elif week == 12:
        from .td import control, evaluate_greedy
        q, counts, truncated = control('q_learning', steps, seed)
        value, success = evaluate_greedy(q)
        rows = [{'method':'DP','start_value':value_iteration(p,rewards,.9)[12]},
                {'method':'Q_learning','env_steps':steps,'greedy_return':value,'success':success,'truncated':truncated}]
        summary += '\n\n这是综合项目起点：还应加入 DQN/A2C、多种子、消融与完整报告。'
    else:
        raise ValueError('week 必须在 1 到 12 之间')
    save_results(output, {'week':week,'seed':seed,'steps':steps,'gamma':.9}, rows, summary)
    for row in rows:
        print(row)
    print(f'结果已保存到 {output}')


def main():
    """解析命令行，并将输出限制在用户指定的参考结果目录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('week', type=int, choices=range(1,13))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps 必须为正')
    output = args.output or Path('practice_reference/results') / f'week{args.week:02}' / f'seed{args.seed}'
    run_week(args.week,args.seed,args.steps,output)


if __name__ == '__main__':
    main()
