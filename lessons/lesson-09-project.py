"""第九节：离线合成图像分类项目。

运行：uv run python lessons/lesson-09-project.py
依赖：项目已有的 CPU PyTorch 与 NumPy；不下载数据或模型权重。
源域训练出的 checkpoint 是真正经过训练的预训练权重，再迁移到目标域。
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


OUT = Path(__file__).with_name("lesson-09-output")
NAMES = ["横线", "竖线", "左上到右下", "左下到右上"]
SIZE = 24
EPOCHS_SOURCE = 10
EPOCHS_TARGET = 12


def make_images(per_class: int, seed: int, domain: str):
    """四种线条方向；按索引分层排列，独立种子保证数据不重叠。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    images, labels = [], []
    directions = [(1., 0.), (0., 1.), (2**-.5, 2**-.5), (2**-.5, -(2**-.5))]
    for label, (dx, dy) in enumerate(directions):
        for _ in range(per_class):
            cx, cy = rng.uniform(8, 16, size=2)
            length = rng.uniform(13, 21)
            width = rng.uniform(.75, 1.45)
            along = (xx-cx)*dx + (yy-cy)*dy
            perp = -(xx-cx)*dy + (yy-cy)*dx
            line = np.exp(-.5*(perp/width)**2) * (np.abs(along) < length/2)
            if domain == "source":
                image = .10 + .80*line + rng.normal(0, .07, (SIZE, SIZE))
            elif domain == "target":
                # 同一方向标签，较低对比度、更多噪声与局部干扰。
                image = .25 + .48*line + rng.normal(0, .21, (SIZE, SIZE))
                for _ in range(3):
                    px, py = rng.integers(1, SIZE-1, size=2)
                    image[py-1:py+1, px-1:px+1] += rng.uniform(.2, .55)
                if rng.random() < .45:
                    gap = rng.uniform(-4, 4)
                    image[(np.abs(along-gap)<1.2) & (np.abs(perp)<2.2)] *= .55
            else:
                raise ValueError(domain)
            images.append(np.clip(image, 0, 1).astype(np.float32)[None])
            labels.append(label)
    return torch.from_numpy(np.stack(images)), torch.tensor(labels, dtype=torch.long)


class LineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(24*6*6, 4))

    def forward(self, x):
        return self.classifier(self.features(x))


def score(model, x, y):
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        losses = nn.functional.cross_entropy(logits, y, reduction="none")
        probabilities = logits.softmax(1)
    return {
        "loss": float(losses.mean()),
        "accuracy": float((logits.argmax(1) == y).float().mean()),
        "prediction": logits.argmax(1).tolist(),
        "confidence": probabilities.max(1).values.tolist(),
        "probabilities": probabilities.tolist(),
    }


def fit(model, train, val, epochs, lr, seed):
    xt, yt = train
    xv, yv = val
    loader = DataLoader(TensorDataset(xt, yt), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    history, best_loss, best_epoch, best_state = [], float("inf"), None, None
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
        train_score, val_score = score(model, xt, yt), score(model, xv, yv)
        row = {"epoch": epoch, "train_loss": train_score["loss"],
               "train_accuracy": train_score["accuracy"],
               "val_loss": val_score["loss"], "val_accuracy": val_score["accuracy"]}
        history.append(row)
        if row["val_loss"] < best_loss:
            best_loss, best_epoch = row["val_loss"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return history, best_epoch, time.perf_counter()-started


def confusion(truth, prediction):
    matrix = [[0]*4 for _ in range(4)]
    for y, p in zip(truth, prediction):
        matrix[int(y)][int(p)] += 1
    return matrix


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    source_train = make_images(180, 901, "source")
    source_val = make_images(40, 902, "source")
    target_pool = make_images(64, 911, "target")
    target_val = make_images(40, 912, "target")
    target_test = make_images(80, 913, "target")

    torch.manual_seed(900)
    pretrained = LineCNN()
    source_history, source_best, source_seconds = fit(
        pretrained, source_train, source_val, EPOCHS_SOURCE, .003, 903)
    source_result = score(pretrained, *source_val)
    torch.save(pretrained.state_dict(), OUT / "source-pretrained.pt")
    # 真的从保存的权重恢复，而不是复用内存中的对象。
    restored = LineCNN()
    restored.load_state_dict(torch.load(OUT / "source-pretrained.pt", map_location="cpu", weights_only=True))
    torch.testing.assert_close(pretrained.features(target_val[0][:4]),
                               restored.features(target_val[0][:4]))
    print(f"源域预训练：验证准确率 {source_result['accuracy']:.1%}，最佳轮次 {source_best}。", flush=True)

    runs = []
    for per_class in (8, 32):
        # 两种数据预算使用同一个 target_pool 的前缀；验证/测试固定。
        indices = torch.cat([torch.arange(c*64, c*64+per_class) for c in range(4)])
        target_train = (target_pool[0][indices], target_pool[1][indices])
        for method in ("scratch", "frozen", "finetune"):
            # 相同种子给目标任务分类头相同初值；转移方法另加载特征层。
            torch.manual_seed(1000 + per_class)
            model = LineCNN()
            if method != "scratch":
                model.features.load_state_dict(restored.features.state_dict())
            if method == "frozen":
                for parameter in model.features.parameters():
                    parameter.requires_grad_(False)
            initial_feature = {k: v.detach().clone() for k, v in model.features.state_dict().items()}
            history, best_epoch, seconds = fit(model, target_train, target_val,
                                               EPOCHS_TARGET, .002, 1200+per_class)
            test = score(model, *target_test)
            feature_change = max(float((model.features.state_dict()[k]-v).abs().max())
                                 for k, v in initial_feature.items())
            if method == "frozen":
                assert feature_change == 0
            if method == "finetune":
                assert feature_change > 0
            result = {"per_class": per_class, "method": method,
                      "train_size": len(target_train[1]),
                      "best_epoch": best_epoch, "seconds": seconds,
                      "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "all_parameters": sum(p.numel() for p in model.parameters()),
                      "feature_change": feature_change,
                      "history": history,
                      "validation": score(model, *target_val)["accuracy"],
                      "test_accuracy": test["accuracy"],
                      "test_loss": test["loss"],
                      "confusion": confusion(target_test[1].tolist(), test["prediction"]),
                      "prediction": test["prediction"],
                      "confidence": test["confidence"],
                      "probabilities": test["probabilities"]}
            runs.append(result)
            print(f"目标域 {per_class}/类 {method}: val {result['validation']:.1%}, "
                  f"test {test['accuracy']:.1%}, {seconds:.2f}s", flush=True)

    # 页面示例取固定样本：包含每类代表、三个方法中的错误和低置信度。
    chosen = set([0, 80, 160, 240])
    for run in runs:
        wrong = [i for i, p in enumerate(run["prediction"]) if p != int(target_test[1][i])]
        chosen.update(wrong[:8])
        chosen.update(sorted(range(len(run["confidence"])), key=lambda i: run["confidence"][i])[:4])
    sample_ids = sorted(chosen)
    samples = [{"index": i, "label": int(target_test[1][i]),
                "pixels": np.round(target_test[0][i, 0].numpy(), 2).tolist()}
               for i in sample_ids]
    source_samples = [{"label": int(source_val[1][i]),
                       "pixels": np.round(source_val[0][i, 0].numpy(), 2).tolist()}
                      for i in (0, 40, 80, 120)]
    payload = {"torch_version": torch.__version__, "device": "cpu",
               "classes": NAMES, "source": {"train_size": len(source_train[1]),
               "val_size": len(source_val[1]), "best_epoch": source_best,
               "val_accuracy": source_result["accuracy"], "seconds": source_seconds,
               "history": source_history},
               "target": {"val_size": len(target_val[1]),
               "test_size": len(target_test[1]),
               "test_truth": target_test[1].tolist()},
               "runs": runs, "samples": samples, "source_samples": source_samples}
    (OUT / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["per_class", "method", "train_size",
                               "best_epoch", "validation", "test_accuracy", "test_loss",
                               "seconds", "trainable_parameters", "feature_change"])
        writer.writeheader()
        for run in runs:
            writer.writerow({key: run[key] for key in writer.fieldnames})
    print(f"结果保存到 {OUT}。", flush=True)


if __name__ == "__main__":
    main()
