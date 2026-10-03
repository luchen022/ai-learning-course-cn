"""第十七节：CPU PyTorch REINFORCE，以及按状态估计的基线。

运行：uv run python lessons/lesson-17-policy-gradient.py
沿用第十六节的随机动作小网格；不训练语言模型，不需要外部环境。
"""
from __future__ import annotations

import json
import random
import runpy
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


HERE = Path(__file__).resolve().parent
OUT = HERE / "lesson-17-output"
PREVIOUS = runpy.run_path(str(HERE / "lesson-16-dqn.py"))
NoisyGrid = PREVIOUS["NoisyGrid"]
encode = PREVIOUS["encode"]
STATES = PREVIOUS["STATES"]
ACTIONS = PREVIOUS["ACTIONS"]
GAMMA = PREVIOUS["GAMMA"]
MAX_STEPS = PREVIOUS["MAX_STEPS"]
SEEDS = [17, 117, 217]
EPISODES = 1200
UPDATE_EVERY = 8
CHECKPOINTS = {0, 200, 400, 800, 1200}


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(6, 32), nn.Tanh(), nn.Linear(32, 4))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)  # logits, shape (..., 4)


def returns_to_go(rewards: list[float]) -> list[float]:
    following = 0.0
    values = []
    for reward in reversed(rewards):
        following = reward + GAMMA * following
        values.append(following)
    return list(reversed(values))


def collect(policy: Policy, seed: int):
    """一整回合的 on-policy 样本，训练动作来自当前策略分布。"""
    env = NoisyGrid(seed)
    states, log_probs, rewards, actions = [], [], [], []
    for _ in range(MAX_STEPS):
        state = env.env.state
        distribution = Categorical(logits=policy(encode(state)))
        action = distribution.sample()
        after, reward, terminal, _actual, _slipped = env.step(int(action))
        states.append(state)
        actions.append(ACTIONS[int(action)])
        log_probs.append(distribution.log_prob(action))
        rewards.append(reward)
        if terminal:
            break
    return {"states": states, "actions": actions, "log_probs": log_probs,
            "rewards": rewards, "returns_to_go": returns_to_go(rewards),
            "end_state": env.env.state,
            "discounted_return": sum(GAMMA**t * r for t, r in enumerate(rewards))}


def evaluate(policy: Policy | None, episodes: int, mode: str = "sample") -> dict:
    """固定评估环境种子，策略采样用独立 Python RNG，不消耗训练 RNG。"""
    outcomes = []
    sample_trace = []
    for i in range(episodes):
        env = NoisyGrid(80_000 + i)
        chooser = random.Random(180_000 + i)
        rewards = []
        trace = []
        for _ in range(MAX_STEPS):
            state = env.env.state
            if policy is None:
                intended = chooser.randrange(4)
            else:
                with torch.no_grad():
                    logits = policy(encode(state))
                    if mode == "greedy":
                        intended = int(logits.argmax().item())
                    else:
                        probabilities = torch.softmax(logits, dim=0).tolist()
                        intended = chooser.choices(range(4), weights=probabilities, k=1)[0]
            after, reward, terminal, actual, slipped = env.step(intended)
            rewards.append(reward)
            if i == 0:
                trace.append({"state": state, "intended": ACTIONS[intended],
                              "actual": ACTIONS[actual], "slipped": slipped,
                              "next_state": after, "reward": reward})
            if terminal:
                break
        outcomes.append({"return": sum(GAMMA**t * r for t, r in enumerate(rewards)),
                         "end_state": env.env.state})
        if i == 0:
            sample_trace = trace
    values = [x["return"] for x in outcomes]
    return {"episodes": episodes, "mean_return": float(np.mean(values)),
            "std_return": float(np.std(values, ddof=1)) if episodes > 1 else 0.0,
            "destination_counts": {s: sum(x["end_state"] == s for x in outcomes)
                                   for s in ("C", "G")},
            "timeout_count": sum(x["end_state"] not in ("C", "G") for x in outcomes),
            "returns": values, "first_trace": sample_trace}


def train(seed: int, baseline_enabled: bool) -> dict:
    torch.manual_seed(seed)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.005)
    state_baseline = {state: 0.0 for state in STATES}
    baseline_seen = {state: False for state in STATES}
    train_returns = []
    losses = []
    checkpoints = []
    first_episode = None
    example_goal_trajectory = None

    def record(episode: int):
        sampled = evaluate(policy, 60, "sample")
        with torch.no_grad():
            probs = torch.softmax(policy(encode("S")), dim=0).tolist()
        checkpoints.append({"episode": episode,
                            "sample_eval_mean": sampled["mean_return"],
                            "sample_eval_std": sampled["std_return"],
                            "sample_eval_goal_count": sampled["destination_counts"]["G"],
                            "recent_train_mean": float(np.mean(train_returns[-80:])) if train_returns else None,
                            "recent_loss_mean": float(np.mean(losses[-10:])) if losses else None,
                            "start_probabilities": dict(zip(ACTIONS, probs)),
                            "baseline_S": state_baseline["S"] if baseline_enabled else None})

    record(0)
    for batch_start in range(1, EPISODES + 1, UPDATE_EVERY):
        terms = []
        seen_returns = defaultdict(list)
        for episode in range(batch_start, batch_start + UPDATE_EVERY):
            trajectory = collect(policy, seed * 100_000 + episode)
            train_returns.append(trajectory["discounted_return"])
            if first_episode is None:
                first_episode = {"states": trajectory["states"],
                                 "actions": trajectory["actions"],
                                 "rewards": trajectory["rewards"],
                                 "returns_to_go": trajectory["returns_to_go"],
                                 "end_state": trajectory["end_state"]}
            if example_goal_trajectory is None and trajectory["end_state"] == "G":
                example_goal_trajectory = {"episode": episode,
                                           "states": trajectory["states"],
                                           "actions": trajectory["actions"],
                                           "rewards": trajectory["rewards"],
                                           "returns_to_go": trajectory["returns_to_go"],
                                           "end_state": trajectory["end_state"]}
            for t, (state, log_prob, outcome) in enumerate(zip(
                    trajectory["states"], trajectory["log_probs"], trajectory["returns_to_go"])):
                baseline = state_baseline[state] if baseline_enabled else 0.0
                advantage = outcome - baseline
                terms.append(-GAMMA**t * log_prob * advantage)
                seen_returns[state].append(outcome)
        # 一次更新汇总 8 条完整新轨迹；旧轨迹不会进入下一批。
        loss = torch.stack(terms).sum() / UPDATE_EVERY
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        losses.append(float(loss.item()))
        # 本批 loss 使用旧基线；更新后的基线只给下一批使用，避免依赖当前动作。
        if baseline_enabled:
            for state, values in seen_returns.items():
                batch_mean = float(np.mean(values))
                if baseline_seen[state]:
                    state_baseline[state] = 0.9 * state_baseline[state] + 0.1 * batch_mean
                else:
                    state_baseline[state] = batch_mean
                    baseline_seen[state] = True
        completed = batch_start + UPDATE_EVERY - 1
        if completed in CHECKPOINTS:
            record(completed)
    sampled = evaluate(policy, 150, "sample")
    greedy = evaluate(policy, 150, "greedy")
    with torch.no_grad():
        probabilities = {state: dict(zip(ACTIONS, torch.softmax(policy(encode(state)), dim=0).tolist()))
                         for state in STATES if state not in ("C", "G")}
    return {"seed": seed, "baseline_enabled": baseline_enabled,
            "checkpoints": checkpoints, "sampled_evaluation": sampled,
            "greedy_evaluation": greedy, "state_action_probabilities": probabilities,
            "first_training_episode": first_episode,
            "example_goal_trajectory": example_goal_trajectory,
            "final_baselines": state_baseline if baseline_enabled else None,
            "optimizer_updates": EPISODES // UPDATE_EVERY}


def two_route_math(probability: float = 0.4, baseline: float = 4.0) -> dict:
    """二动作独立示例，用来检查 score-function 恒等式。"""
    p, b = probability, baseline
    raw_goal = 10 * (1 - p)
    raw_coin = 2 * (-p)
    goal = (10 - b) * (1 - p)
    coin = (2 - b) * (-p)
    expectation = p * goal + (1 - p) * coin
    variance = p * (goal - expectation)**2 + (1 - p) * (coin - expectation)**2
    raw_variance = p * (raw_goal - expectation)**2 + (1 - p) * (raw_coin - expectation)**2
    assert abs(expectation - 8 * p * (1 - p)) < 1e-12
    return {"p_goal": p, "baseline": b, "expected_reward": 2 + 8*p,
            "true_derivative": 8*p*(1-p), "sample_gradient_goal": goal,
            "sample_gradient_coin": coin, "gradient_expectation": expectation,
            "gradient_variance": variance, "variance_without_baseline": raw_variance}


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    baseline = evaluate(None, 300)
    runs = [train(seed, enabled) for enabled in (False, True) for seed in SEEDS]
    math = two_route_math()
    assert all(len(x["checkpoints"]) == len(CHECKPOINTS) for x in runs)
    assert all(x["optimizer_updates"] == 150 for x in runs)
    result = {"environment": {"grid": PREVIOUS["BASE"]["GRID"],
                               "states": STATES, "actions": ACTIONS,
                               "gamma": GAMMA, "max_steps": MAX_STEPS,
                               "slip_probability": PREVIOUS["SLIP"]},
              "configuration": {"episodes": EPISODES, "update_every_episodes": UPDATE_EVERY,
                                "seeds": SEEDS, "learning_rate": 0.005,
                                "network": "one-hot 6 -> tanh 32 -> logits 4",
                                "baseline": "previous-batch state return EMA (0.9 old + 0.1 batch mean)"},
              "random_baseline": baseline, "two_route_math": math, "runs": runs}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = HERE / "lesson-17-policy-gradient.html"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        opener = '<script id="lesson-data" type="application/json">'
        start = html.index(opener) + len(opener)
        end = html.index("</script>", start)
        html_path.write_text(html[:start] + json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                             + html[end:], encoding="utf-8")
    print("随机策略评估均值", round(baseline["mean_return"], 3))
    for run in runs:
        print("基线", run["baseline_enabled"], "种子", run["seed"],
              "采样策略", round(run["sampled_evaluation"]["mean_return"], 3),
              "贪心策略", round(run["greedy_evaluation"]["mean_return"], 3),
              "起点 P(下)", round(run["state_action_probabilities"]["S"]["down"], 3))
    print("结果保存到", OUT / "results.json")


if __name__ == "__main__":
    main()
