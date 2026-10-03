"""第三节：泛化评估、过拟合、L2、真实早停记录和分类指标。只依赖 NumPy。"""
from pathlib import Path
import csv
import numpy as np


def truth(x):
    return 0.5 + 0.8 * x - 1.4 * x**2 + 0.5 * x**3


def data():
    x_train = np.linspace(-0.95, 0.94, 10)
    train_noise = np.array([0.2, -0.15, 0.3, -0.2, 0.1, 0.25, -0.3, 0.15, -0.15, 0.22])
    y_train = truth(x_train) + train_noise
    x_val = np.linspace(-0.99, 0.99, 21)
    y_val = truth(x_val) + 0.2 * np.sin(np.arange(21) * 2.3)
    x_test = np.linspace(-0.975, 0.975, 23)
    y_test = truth(x_test) + 0.2 * np.cos(np.arange(23) * 1.7 + 0.4)
    return x_train, y_train, x_val, y_val, x_test, y_test


def design(x, degree):
    # 每一列依次为 1、x、x²、…、x^degree。
    return np.vander(x, degree + 1, increasing=True)


def fit_polynomial(x, y, degree=3, regularization=0.0):
    A = design(x, degree)
    penalty = np.diag([0.0] + [1.0] * degree)  # 不惩罚常数项
    # 最小化 MSE + lambda * sum(w[1:] ** 2)。
    return np.linalg.solve(A.T @ A / len(x) + regularization * penalty, A.T @ y / len(x))


def mse(w, x, y):
    return float(np.mean((design(x, len(w) - 1) @ w - y) ** 2))


def early_stopping_records(x_train, y_train, x_val, y_val, steps=1000):
    A, V = design(x_train, 9), design(x_val, 9)
    w = np.zeros(10)
    records = []
    for step in range(steps + 1):
        if step % 100 == 0:
            records.append({
                "step": step,
                "train_mse": float(np.mean((A @ w - y_train) ** 2)),
                "val_mse": float(np.mean((V @ w - y_val) ** 2)),
                "weights": w.copy(),
            })
        if step < steps:
            w -= 0.3 * (2 / len(x_train) * A.T @ (A @ w - y_train))
    return records


def select_early_checkpoint(records, patience=3):
    best = records[0]
    failures = 0
    for record in records[1:]:
        if record["val_mse"] < best["val_mse"]:
            best, failures = record, 0
        else:
            failures += 1
        if failures >= patience:
            return best, record["step"]
    return best, None


def classification_metrics(y, scores, threshold=0.5):
    prediction = scores >= threshold
    positive = y == 1
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & ~positive))
    fn = int(np.sum(~prediction & positive))
    tn = int(np.sum(~prediction & ~positive))
    # 分母为 0 时本练习约定返回 0。
    ratio = lambda a, b: a / b if b else 0.0
    return {
        "threshold": threshold, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": ratio(tp + tn, len(y)),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def save_csv(path, records):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    xt, yt, xv, yv, x_test, y_test = data()
    output = Path(__file__).resolve().parent / "lesson-03-output"
    output.mkdir(exist_ok=True)
    candidates = [(0, 0), (1, 0), (3, 0), (9, 0), (9, 0.001), (9, 0.1), (9, 1)]
    comparison = []
    fitted = []
    for degree, strength in candidates:
        w = fit_polynomial(xt, yt, degree, strength)
        fitted.append(w)
        comparison.append({
            "degree": degree, "lambda": strength,
            "train_mse": mse(w, xt, yt), "val_mse": mse(w, xv, yv),
        })
        print(comparison[-1])
    save_csv(output / "polynomial-comparison.csv", comparison)

    # 只用验证集选方案，再对锁定方案评估一次测试集。
    winner = min(range(len(comparison)), key=lambda i: comparison[i]["val_mse"])
    print("Selected by validation:", comparison[winner])
    print("Final test MSE:", mse(fitted[winner], x_test, y_test))

    records = early_stopping_records(xt, yt, xv, yv)
    checkpoint, stop_step = select_early_checkpoint(records)
    print("Early stop at:", stop_step, "restore step:", checkpoint["step"])
    save_csv(output / "early-stopping.csv", [
        {k: r[k] for k in ("step", "train_mse", "val_mse")} for r in records
    ])
    # records 还保留未早停的后续走势，用于与恢复最佳参数的策略作教学对比。

    scores = np.array([.92, .71, .46, .22, .85, .66, .58, .43, .38, .31,
                       .27, .24, .20, .18, .16, .14, .12, .10, .08, .04])
    labels = np.array([1] * 4 + [0] * 16)
    metrics = [classification_metrics(labels, scores, t) for t in (0.5, 0.2, 1.0)]
    for result in metrics:
        print(result)
    save_csv(output / "classification-metrics.csv", metrics)
    print("CSV records saved to:", output)


if __name__ == "__main__":
    main()

