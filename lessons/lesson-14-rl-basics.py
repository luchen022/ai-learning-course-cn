"""第十四节：确定性网格世界、折扣回报与探索/利用。

运行：uv run python lessons/lesson-14-rl-basics.py
仅用 Python 标准库，不下载环境或模型。下一节再实现 Q-learning。
"""
from __future__ import annotations

import json
import random
from pathlib import Path


OUT = Path(__file__).with_name("lesson-14-output")
GRID = [["S", "C", "#", "#"], ["A", "B", "D", "G"]]
POSITION = {name: (r, c) for r, row in enumerate(GRID)
            for c, name in enumerate(row) if name != "#"}
DELTA = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
TERMINAL = {"C", "G"}
GAMMAS = (0.5, 0.8, 0.95)
COIN_ROUTE = ["right"]
GOAL_ROUTE = ["down", "right", "right", "right"]


class GridWorld:
    def __init__(self):
        self.state = "S"

    def reset(self) -> str:
        self.state = "S"
        return self.state

    def step(self, action: str) -> tuple[str, int, bool]:
        if action not in DELTA:
            raise ValueError(f"未知动作：{action}")
        if self.state in TERMINAL:
            return self.state, 0, True
        r, c = POSITION[self.state]
        dr, dc = DELTA[action]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < len(GRID) and 0 <= nc < len(GRID[0])) or GRID[nr][nc] == "#":
            return self.state, 0, False
        self.state = GRID[nr][nc]
        reward = 2 if self.state == "C" else 10 if self.state == "G" else 0
        return self.state, reward, self.state in TERMINAL


def simulate(actions: list[str], gamma: float) -> dict:
    env = GridWorld()
    trace = []
    total = 0.0
    for t, action in enumerate(actions):
        before = env.state
        after, reward, done = env.step(action)
        discounted = gamma**t * reward
        total += discounted
        trace.append({"t": t, "state": before, "action": action,
                      "next_state": after, "reward": reward, "discounted": discounted,
                      "done": done})
        if done:
            break
    return {"actions": actions, "trace": trace, "discounted_return": total,
            "end_state": env.state}


def transition_table() -> dict:
    transitions = {}
    for state in POSITION:
        transitions[state] = {}
        for action in DELTA:
            env = GridWorld()
            env.state = state
            next_state, reward, done = env.step(action)
            transitions[state][action] = {"next_state": next_state,
                                          "reward": reward, "done": done,
                                          "probability": 1.0}
    return transitions


def optimal_values(gamma: float, transitions: dict) -> dict[str, float]:
    """内部核验：有限状态确定性环境的 Bellman 最优备份。第十五节细讲。"""
    values = {state: 0.0 for state in POSITION}
    for _ in range(100):
        following = {}
        for state in POSITION:
            if state in TERMINAL:
                following[state] = 0.0
            else:
                following[state] = max(
                    item["reward"] + gamma * values[item["next_state"]]
                    for item in transitions[state].values()
                )
        values = following
    return values


def route_experiment(epsilon: float, episodes: int = 30, gamma: float = 0.8) -> dict:
    """只在起点选整条路线，末尾观察回报；不是逐状态 Q-learning。"""
    rng = random.Random(1414 + round(epsilon * 100))
    estimates = {"coin": 0.0, "goal": 0.0}
    counts = {"coin": 0, "goal": 0}
    records = []
    for episode in range(1, episodes + 1):
        explore = rng.random() < epsilon
        if explore:
            route = rng.choice(["coin", "goal"])
        else:
            route = max(("coin", "goal"), key=lambda name: estimates[name])
        actions = COIN_ROUTE if route == "coin" else GOAL_ROUTE
        outcome = simulate(actions, gamma)
        observed = outcome["discounted_return"]
        counts[route] += 1
        estimates[route] += (observed - estimates[route]) / counts[route]
        records.append({"episode": episode, "route": route, "explore": explore,
                        "observed_return": observed,
                        "coin_estimate": estimates["coin"],
                        "goal_estimate": estimates["goal"]})
    return {"epsilon": epsilon, "episodes": records,
            "counts": counts, "final_estimates": estimates,
            "first_goal_episode": next((x["episode"] for x in records if x["route"] == "goal"), None)}


def main():
    OUT.mkdir(exist_ok=True)
    transitions = transition_table()
    assert transitions["S"]["right"] == {
        "next_state": "C", "reward": 2, "done": True, "probability": 1.0
    }
    assert transitions["S"]["down"]["next_state"] == "A"
    assert transitions["S"]["up"]["next_state"] == "S"
    assert transitions["B"]["up"]["next_state"] == "C"
    assert transitions["G"]["left"]["done"]
    try:
        GridWorld().step("jump")
    except ValueError:
        pass
    else:
        raise AssertionError("非法动作应报错")

    route_results = []
    for gamma in GAMMAS:
        coin = simulate(COIN_ROUTE, gamma)
        goal = simulate(GOAL_ROUTE, gamma)
        assert coin["end_state"] == "C" and goal["end_state"] == "G"
        assert coin["discounted_return"] == 2
        assert abs(goal["discounted_return"] - 10 * gamma**3) < 1e-12
        assert [x["reward"] for x in goal["trace"]] == [0, 0, 0, 10]
        values = optimal_values(gamma, transitions)
        q_right = 2.0
        q_down = gamma * values["A"]
        assert abs(q_down - goal["discounted_return"]) < 1e-10
        assert abs(values["S"] - max(q_right, q_down)) < 1e-10
        route_results.append({"gamma": gamma, "coin": coin, "goal": goal,
                              "optimal_q_start": {"right": q_right, "down": q_down},
                              "optimal_value_start": values["S"],
                              "best_route": "coin" if q_right > q_down else "goal"})
    assert route_results[0]["best_route"] == "coin"
    assert route_results[1]["best_route"] == route_results[2]["best_route"] == "goal"

    exploration = [route_experiment(epsilon) for epsilon in (0.0, 0.2, 0.5)]
    assert exploration[0]["counts"]["goal"] == 0
    assert exploration[1]["counts"]["goal"] > 0
    assert exploration[2]["counts"]["goal"] > 0
    for run in exploration:
        assert sum(run["counts"].values()) == 30
        if run["counts"]["goal"]:
            assert abs(run["final_estimates"]["goal"] - 5.12) < 1e-10

    result = {
        "grid": GRID,
        "positions": {k: list(v) for k, v in POSITION.items()},
        "state_meanings": {"S": "起点", "C": "即时 +2，终止", "A": "长路线第一格",
                           "B": "长路线第二格", "D": "长路线第三格", "G": "延迟 +10，终止"},
        "actions": {k: list(v) for k, v in DELTA.items()},
        "transitions": transitions,
        "discount_cases": route_results,
        "exploration_at_gamma_0_8": exploration,
        "threshold_gamma": (2 / 10) ** (1 / 3),
    }
    (OUT / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("网格：2×4；状态", len(POSITION), "个；动作", len(DELTA), "个")
    for item in route_results:
        print("γ=", item["gamma"], "即时路线", f"{item['coin']['discounted_return']:.3f}",
              "延迟路线", f"{item['goal']['discounted_return']:.3f}",
              "起点较优", item["best_route"])
    for run in exploration:
        print("ε=", run["epsilon"], "30 次中尝试长路线", run["counts"]["goal"],
              "次；首次第", run["first_goal_episode"], "次")
    print("结果保存到", OUT)


if __name__ == "__main__":
    main()
