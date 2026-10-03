"""第十六节：在本地小网格中运行 CPU PyTorch DQN。

运行：uv run python lessons/lesson-16-dqn.py
沿用第十四节网格；不需要 Gym、不下载数据或模型。
"""
from __future__ import annotations

import json
import random
import runpy
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
OUT = HERE / "lesson-16-output"
BASE = runpy.run_path(str(HERE / "lesson-14-rl-basics.py"))
GridWorld = BASE["GridWorld"]
STATES = ["S", "A", "B", "D", "C", "G"]
ACTIONS = ["right", "down", "up", "left"]
INDEX = {state: i for i, state in enumerate(STATES)}
GAMMA = 0.8
SLIP = 0.1
MAX_STEPS = 20
EPISODES = 600
SEEDS = [16, 116, 216]
CHECKPOINTS = {0, 100, 200, 400, 600}
BATCH = 32
CAPACITY = 1000
TARGET_SYNC = 100  # 梯度更新次数


def encode(state: str) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(INDEX[state]), len(STATES)).float()


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(6, 32), nn.ReLU(),
                                    nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


class NoisyGrid:
    """每一步有 10% 概率将预定动作替换为四方向均匀随机动作。"""
    def __init__(self, seed: int):
        self.env = GridWorld()
        self.rng = random.Random(seed)

    def step(self, intended: int):
        slipped = self.rng.random() < SLIP
        actual = self.rng.randrange(len(ACTIONS)) if slipped else intended
        next_state, reward, terminal = self.env.step(ACTIONS[actual])
        return next_state, float(reward), terminal, actual, slipped


def epsilon_at(episode: int) -> float:
    return max(0.05, 1.0 - 0.95 * (episode - 1) / 399)


def greedy(net: QNetwork, state: str) -> int:
    with torch.no_grad():
        return int(net(encode(state).unsqueeze(0)).argmax(dim=1).item())


def rollout(net: QNetwork | None, seed: int, trace: bool = False) -> dict:
    env = NoisyGrid(seed)
    chooser = random.Random(seed + 900_000)
    total = 0.0
    steps = []
    for t in range(MAX_STEPS):
        state = env.env.state
        action = chooser.randrange(len(ACTIONS)) if net is None else greedy(net, state)
        after, reward, terminal, actual, slipped = env.step(action)
        total += GAMMA**t * reward
        if trace:
            steps.append({"state": state, "intended": ACTIONS[action],
                          "actual": ACTIONS[actual], "slipped": slipped,
                          "next_state": after, "reward": reward})
        if terminal:
            break
    return {"return": total, "end_state": env.env.state,
            "steps": t + 1, "trace": steps}


def evaluate(net: QNetwork | None, episodes: int = 100) -> dict:
    # 所有训练种子使用同一组环境随机种子，方便横向比较。
    outcomes = [rollout(net, 60_000 + i) for i in range(episodes)]
    returns = [x["return"] for x in outcomes]
    return {"episodes": episodes, "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns, ddof=1)) if episodes > 1 else 0.0,
            "destination_counts": {name: sum(x["end_state"] == name for x in outcomes)
                                   for name in ("C", "G")},
            "timeout_count": sum(x["end_state"] not in ("C", "G") for x in outcomes),
            "returns": returns}


def train(seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    chooser = random.Random(seed + 1_000)
    replay_rng = random.Random(seed + 2_000)
    online, target = QNetwork(), QNetwork()
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=0.003)
    replay = deque(maxlen=CAPACITY)
    train_returns = []
    losses = []
    records = []
    syncs = 0
    updates = 0
    first_batch = None

    def record(episode: int):
        score = evaluate(online, 40)
        with torch.no_grad():
            start_q = online(encode("S").unsqueeze(0)).squeeze(0).tolist()
        records.append({"episode": episode, "evaluation_mean": score["mean_return"],
                        "evaluation_std": score["std_return"],
                        "evaluation_goal_count": score["destination_counts"]["G"],
                        "recent_train_mean": float(np.mean(train_returns[-50:])) if train_returns else None,
                        "recent_loss_mean": float(np.mean(losses[-50:])) if losses else None,
                        "q_start": dict(zip(ACTIONS, start_q)), "replay_size": len(replay),
                        "updates": updates, "target_syncs": syncs})

    record(0)
    for episode in range(1, EPISODES + 1):
        env = NoisyGrid(seed * 100_000 + episode)
        episode_return = 0.0
        for t in range(MAX_STEPS):
            state = env.env.state
            action = chooser.randrange(4) if chooser.random() < epsilon_at(episode) else greedy(online, state)
            after, reward, terminal, _actual, _slipped = env.step(action)
            replay.append((state, action, reward, after, terminal, episode))
            episode_return += GAMMA**t * reward
            if len(replay) >= BATCH:
                examples = replay_rng.sample(replay, BATCH)
                states = torch.stack([encode(x[0]) for x in examples])
                actions = torch.tensor([x[1] for x in examples], dtype=torch.long).unsqueeze(1)
                rewards = torch.tensor([x[2] for x in examples], dtype=torch.float32)
                next_states = torch.stack([encode(x[3]) for x in examples])
                terminals = torch.tensor([x[4] for x in examples], dtype=torch.bool)
                chosen_q = online(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    next_best = target(next_states).max(dim=1).values
                    td_target = rewards + GAMMA * next_best * (~terminals).float()
                loss = torch.nn.functional.smooth_l1_loss(chosen_q, td_target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
                optimizer.step()
                losses.append(float(loss.item()))
                updates += 1
                if first_batch is None:
                    first_batch = [{"state": x[0], "action": ACTIONS[x[1]],
                                    "reward": x[2], "next_state": x[3],
                                    "terminal": x[4], "episode": x[5]}
                                   for x in examples[:8]]
                if updates % TARGET_SYNC == 0:
                    target.load_state_dict(online.state_dict())
                    syncs += 1
            if terminal:
                break
            # 20 步是采样截断，不是真正终止；回放记录不会把它标成 terminal。
        train_returns.append(episode_return)
        if episode in CHECKPOINTS:
            record(episode)
    final = evaluate(online, 100)
    with torch.no_grad():
        all_q = {state: dict(zip(ACTIONS, online(encode(state).unsqueeze(0)).squeeze(0).tolist()))
                 for state in STATES if state not in ("C", "G")}
    trace_index = next((i for i, value in enumerate(final["returns"])
                        if value < 10 * GAMMA**3 - 1e-6), 0)
    return {"seed": seed, "checkpoints": records, "final_evaluation": final,
            "sample_trace": rollout(online, 60_000 + trace_index, True),
            "q_values": all_q, "first_replay_batch": first_batch,
            "total_updates": updates, "target_syncs": syncs,
            "epsilon_final": epsilon_at(EPISODES)}


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    baseline = evaluate(None, 300)
    runs = [train(seed) for seed in SEEDS]
    result = {"environment": {"grid": BASE["GRID"], "states": STATES,
                               "actions": ACTIONS, "gamma": GAMMA,
                               "slip_probability": SLIP, "max_steps": MAX_STEPS,
                               "slip_rule": "with probability 0.1, replace intended action by uniform action among four directions"},
              "configuration": {"episodes": EPISODES, "seeds": SEEDS, "batch": BATCH,
                                "replay_capacity": CAPACITY, "target_sync_updates": TARGET_SYNC,
                                "optimizer": "Adam", "learning_rate": 0.003,
                                "loss": "SmoothL1Loss", "hidden_sizes": [32, 32],
                                "epsilon_start": 1.0, "epsilon_end": 0.05,
                                "epsilon_decay_episodes": 400},
              "random_baseline": baseline, "runs": runs}
    assert all(x["total_updates"] > 0 and x["target_syncs"] > 0 for x in runs)
    assert all(len(x["first_replay_batch"]) == 8 for x in runs)
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = HERE / "lesson-16-dqn.html"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        opener = '<script id="lesson-data" type="application/json">'
        start = html.index(opener) + len(opener)
        end = html.index("</script>", start)
        payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        html_path.write_text(html[:start] + payload + html[end:], encoding="utf-8")
    print("随机策略：", round(baseline["mean_return"], 3), "(300 回合)")
    for x in runs:
        e = x["final_evaluation"]
        print("种子", x["seed"], "评估均值", round(e["mean_return"], 3),
              "标准差", round(e["std_return"], 3), "到 G", e["destination_counts"]["G"], "/ 100")
    print("结果保存到", OUT / "results.json")


if __name__ == "__main__":
    main()
