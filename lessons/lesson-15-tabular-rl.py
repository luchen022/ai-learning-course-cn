"""第十五节：价值迭代、MC/TD 预测与 NumPy 表格 Q-learning。

运行：uv run python lessons/lesson-15-tabular-rl.py
沿用第十四节的本地网格，不下载环境或模型。
"""
from __future__ import annotations

import json
import random
import runpy
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "lesson-15-output"
PREVIOUS = runpy.run_path(str(HERE / "lesson-14-rl-basics.py"))
GridWorld = PREVIOUS["GridWorld"]
TRANSITIONS = PREVIOUS["transition_table"]()
GOAL_ROUTE = PREVIOUS["GOAL_ROUTE"]
STATES = ["S", "A", "B", "D", "C", "G"]
ACTIONS = ["right", "down", "up", "left"]  # 平手时先向右，便于展示 ε=0 的局限。
STATE_ID = {state: i for i, state in enumerate(STATES)}
TERMINAL = {"C", "G"}
GAMMAS = (0.5, 0.8)
EPSILONS = (0.0, 0.1, 0.3)
ALPHA = 0.5
EPISODES = 800
MAX_STEPS = 20
CHECKPOINTS = {0, 10, 25, 50, 100, 200, 400, 800}


def value_iteration(gamma: float, sweeps: int = 8) -> list[dict]:
    """已知完整 P、r，采用同步 Bellman 最优备份。"""
    values = {state: 0.0 for state in STATES}
    history = [{"sweep": 0, "values": values.copy()}]
    for sweep in range(1, sweeps + 1):
        next_values = {}
        for state in STATES:
            if state in TERMINAL:
                next_values[state] = 0.0
                continue
            next_values[state] = max(
                item["reward"] + (0.0 if item["done"] else gamma * values[item["next_state"]])
                for item in TRANSITIONS[state].values()
            )
        values = next_values
        history.append({"sweep": sweep, "values": values.copy()})
    return history


def goal_trace() -> list[dict]:
    env = GridWorld()
    trace = []
    for action in GOAL_ROUTE:
        state = env.state
        next_state, reward, done = env.step(action)
        trace.append({"state": state, "action": action,
                      "next_state": next_state, "reward": reward, "done": done})
    assert trace[-1]["done"] and trace[-1]["next_state"] == "G"
    return trace


def mc_td_prediction(gamma: float = 0.8, alpha: float = 0.5, episodes: int = 6) -> dict:
    """固定沿长路线，估计 V^π；不在这里改进策略。"""
    trace = goal_trace()
    mc = {state: 0.0 for state in STATES}
    td = {state: 0.0 for state in STATES}
    true_returns = {}
    future = 0.0
    for item in reversed(trace):
        future = item["reward"] + gamma * future
        true_returns[item["state"]] = future
    records = [{"episode": 0, "mc": mc.copy(), "td": td.copy()}]
    first_td_updates = []
    for episode in range(1, episodes + 1):
        env = GridWorld()
        observed = []
        # TD(0)：环境每返回一步，就用下一状态的当前估计更新。
        for action in GOAL_ROUTE:
            state = env.state
            next_state, reward, done = env.step(action)
            old = td[state]
            target = reward + (0.0 if done else gamma * td[next_state])
            td[state] += alpha * (target - old)
            observed.append({"state": state, "reward": reward})
            if episode == 1:
                first_td_updates.append({"state": state, "action": action,
                                         "reward": reward, "old": old,
                                         "target": target, "new": td[state]})
        # MC：回合结束后倒推每个位置的完整实际回报，再更新。
        future = 0.0
        for item in reversed(observed):
            future = item["reward"] + gamma * future
            state = item["state"]
            mc[state] += alpha * (future - mc[state])
        records.append({"episode": episode, "mc": mc.copy(), "td": td.copy()})
    assert [true_returns[s] for s in ("S", "A", "B", "D")] == [5.12, 6.4, 8.0, 10.0] or all(
        abs(true_returns[s] - target) < 1e-12
        for s, target in (("S", 5.12), ("A", 6.4), ("B", 8), ("D", 10))
    )
    assert abs(records[1]["mc"]["S"] - 2.56) < 1e-12
    assert records[1]["td"]["S"] == 0
    assert records[1]["td"]["D"] == 5
    return {"gamma": gamma, "alpha": alpha, "trace": trace,
            "true_returns": true_returns, "episodes": records,
            "first_td_updates": first_td_updates}


def greedy_action(q: np.ndarray, state: str) -> str:
    return ACTIONS[int(np.argmax(q[STATE_ID[state]]))]


def rollout(q: np.ndarray | None, gamma: float, rng: random.Random | None = None) -> dict:
    """q=None 是均匀随机策略；否则按 Q 表贪心，不做探索。"""
    env = GridWorld()
    trace = []
    discounted_return = 0.0
    for t in range(MAX_STEPS):
        state = env.state
        action = rng.choice(ACTIONS) if q is None else greedy_action(q, state)
        next_state, reward, done = env.step(action)
        discounted_return += gamma**t * reward
        trace.append({"t": t, "state": state, "action": action,
                      "next_state": next_state, "reward": reward,
                      "done": done})
        if done:
            break
    return {"return": discounted_return, "end_state": env.state,
            "steps": len(trace), "terminated": env.state in TERMINAL,
            "trace": trace}


def q_learning(gamma: float, epsilon: float, seed: int) -> dict:
    rng = random.Random(seed)
    q = np.zeros((len(STATES), len(ACTIONS)), dtype=np.float64)
    returns = []
    destinations = []
    checkpoints = []

    def save_checkpoint(episode: int):
        greedy = rollout(q, gamma)
        checkpoints.append({"episode": episode,
                            "q_start": {action: float(q[STATE_ID["S"], i])
                                        for i, action in enumerate(ACTIONS)},
                            "greedy_end": greedy["end_state"],
                            "greedy_return": greedy["return"],
                            "recent_train_mean": float(np.mean(returns[-50:])) if returns else None})

    save_checkpoint(0)
    for episode in range(1, EPISODES + 1):
        env = GridWorld()
        total = 0.0
        for t in range(MAX_STEPS):
            state = env.state
            if rng.random() < epsilon:
                action = rng.choice(ACTIONS)
            else:
                action = greedy_action(q, state)
            next_state, reward, done = env.step(action)
            s, a = STATE_ID[state], ACTIONS.index(action)
            target = reward if done else reward + gamma * float(np.max(q[STATE_ID[next_state]]))
            q[s, a] += ALPHA * (target - q[s, a])
            total += gamma**t * reward
            if done:
                break
        returns.append(total)
        destinations.append(env.state)
        if episode in CHECKPOINTS:
            save_checkpoint(episode)
    greedy = rollout(q, gamma)
    return {"gamma": gamma, "epsilon": epsilon, "seed": seed,
            "alpha": ALPHA, "episodes": EPISODES, "max_steps": MAX_STEPS,
            "q_shape": list(q.shape), "q_table": q.tolist(),
            "checkpoints": checkpoints,
            "training_destination_counts": {state: destinations.count(state)
                                            for state in ("C", "G")},
            "training_timeout_count": sum(s not in TERMINAL for s in destinations),
            "final_greedy": greedy,
            "final_policy": {s: greedy_action(q, s) for s in STATES if s not in TERMINAL}}


def random_baseline(gamma: float, episodes: int = 300) -> dict:
    rng = random.Random(151500 + int(gamma * 100))
    outcomes = [rollout(None, gamma, rng) for _ in range(episodes)]
    return {"episodes": episodes, "mean_return": sum(x["return"] for x in outcomes) / episodes,
            "destination_counts": {s: sum(x["end_state"] == s for x in outcomes)
                                   for s in ("C", "G")},
            "timeout_count": sum(not x["terminated"] for x in outcomes)}


def sarsa_q_numeric_example(gamma: float = 0.8, alpha: float = 0.5) -> dict:
    old, reward, next_actual, next_best = 1.0, 0.0, 1.0, 5.0
    sarsa_target = reward + gamma * next_actual
    q_target = reward + gamma * next_best
    return {"old": old, "reward": reward, "gamma": gamma, "alpha": alpha,
            "next_chosen_q": next_actual, "next_max_q": next_best,
            "sarsa_target": sarsa_target,
            "sarsa_new": old + alpha * (sarsa_target - old),
            "q_learning_target": q_target,
            "q_learning_new": old + alpha * (q_target - old)}


def main():
    OUT.mkdir(exist_ok=True)
    vi = {str(gamma): value_iteration(gamma) for gamma in GAMMAS}
    assert vi["0.8"][1]["values"]["D"] == 10
    assert vi["0.8"][1]["values"]["S"] == 2
    assert abs(vi["0.8"][4]["values"]["S"] - 5.12) < 1e-12
    assert vi["0.5"][-1]["values"]["S"] == 2
    prediction = mc_td_prediction()
    runs = []
    for gamma in GAMMAS:
        for epsilon in EPSILONS:
            seed = 15000 + int(gamma * 1000) + int(epsilon * 100)
            runs.append(q_learning(gamma, epsilon, seed))
    for run in runs:
        final = run["final_greedy"]
        if run["epsilon"] == 0:
            assert final["end_state"] == "C"
        elif run["gamma"] == 0.8:
            assert final["end_state"] == "G"
            assert abs(final["return"] - 5.12) < 1e-9
        else:
            assert final["end_state"] == "C"
        assert run["q_shape"] == [6, 4]
    baselines = {str(gamma): random_baseline(gamma) for gamma in GAMMAS}
    example = sarsa_q_numeric_example()
    assert abs(example["sarsa_new"] - 0.9) < 1e-12
    assert abs(example["q_learning_new"] - 2.5) < 1e-12
    result = {
        "environment": {"grid": PREVIOUS["GRID"], "states": STATES,
                        "actions": ACTIONS, "terminal": sorted(TERMINAL),
                        "max_steps": MAX_STEPS,
                        "transition_table": TRANSITIONS},
        "value_iteration": vi, "mc_td_prediction": prediction,
        "q_learning": {"episodes": EPISODES, "alpha": ALPHA,
                       "runs": runs, "random_baselines": baselines},
        "sarsa_vs_q_numeric": example,
    }
    (OUT / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("状态×动作 Q 表：", (len(STATES), len(ACTIONS)), "；训练", EPISODES, "回合/设置")
    print("价值迭代 γ=0.8：V₀(S)=0 → V₄(S)=", vi["0.8"][4]["values"]["S"])
    print("固定长路线第一回合：MC V(S)=", prediction["episodes"][1]["mc"]["S"],
          "，TD V(S)=", prediction["episodes"][1]["td"]["S"])
    for run in runs:
        print("γ=", run["gamma"], "ε=", run["epsilon"],
              "贪心评估终点", run["final_greedy"]["end_state"],
              "回报", f"{run['final_greedy']['return']:.3f}")
    print("结果保存到", OUT)


if __name__ == "__main__":
    main()
