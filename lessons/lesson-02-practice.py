"""第二节：损失、批次训练、手写反向传播和数值梯度检查。只依赖 NumPy。"""
from pathlib import Path
import csv
import numpy as np


def mse(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match.")
    return float(np.mean((prediction - target) ** 2))


def binary_cross_entropy(y, p):
    # 教学例子限制到开区间；实际训练可使用稳定的 logits 版本损失。
    if y not in (0, 1) or not 0 < p < 1:
        raise ValueError("Expected y in {0, 1} and 0 < p < 1.")
    return float(-np.log(p) if y == 1 else -np.log1p(-p))


def forward(params, x=2.0, y=1.0):
    w1, b1, w2, b2 = params
    z = w1 * x + b1
    h = np.tanh(z)
    prediction = w2 * h + b2
    error = prediction - y
    return z, h, prediction, error, error**2


def backward(params, x=2.0, y=1.0):
    _, h, _, error, _ = forward(params, x, y)
    d_prediction = 2 * error
    d_h = d_prediction * params[2]
    d_z = d_h * (1 - h**2)
    return np.array([d_z * x, d_z, d_prediction * h, d_prediction])


def numerical_gradient(params, x=2.0, y=1.0, epsilon=1e-5):
    result = np.zeros_like(params, dtype=float)
    for index in range(len(params)):
        plus, minus = params.copy(), params.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        result[index] = (forward(plus, x, y)[-1] - forward(minus, x, y)[-1]) / (2 * epsilon)
    return result


def train(batch_size=2, learning_rate=0.05, epochs=30, seed=12345):
    if batch_size not in (1, 2, 6):
        raise ValueError("This exercise uses batch sizes 1, 2 or 6.")
    X = np.array([[-2.], [-1.], [0.], [1.], [2.], [3.]])
    y = np.array([[-2.5], [-1.3], [1.2], [2.6], [5.3], [6.8]])
    W, b = np.zeros((1, 1)), np.zeros((1,))
    rng = np.random.default_rng(seed)
    # Python 和网页使用各自的固定种子随机数生成器，批次顺序不逐项相同。
    records = [(0, 0, mse(X @ W + b, y), W.item(), b.item())]
    step = seen = 0
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            indices = order[start:start + batch_size]
            xb, yb = X[indices], y[indices]
            error = xb @ W + b - yb
            grad_W = 2 / len(xb) * xb.T @ error
            grad_b = 2 * error.mean(axis=0)
            # 梯度先基于旧 W、b 算好，再一起更新。
            W -= learning_rate * grad_W
            b -= learning_rate * grad_b
            step += 1
            seen += len(xb)
            loss = mse(X @ W + b, y)
            records.append((step, seen, loss, W.item(), b.item()))
            if not np.isfinite(loss) or loss > 1e12:
                print("Stopped: loss diverged. Lower the learning rate.")
                return records
    return records


def main():
    print("BCE(y=1, p=0.9):", binary_cross_entropy(1, 0.9))
    print("BCE(y=0, p=0.9):", binary_cross_entropy(0, 0.9))
    params = np.array([0.5, 0., 1., 0.])
    analytic = backward(params)
    numeric = numerical_gradient(params)
    for name, a, n in zip(("w1", "b1", "w2", "b2"), analytic, numeric):
        print(f"{name}: analytic={a:.8f}, numeric={n:.8f}, abs_diff={abs(a-n):.3e}")
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)
    updated = params - 0.1 * analytic
    print("Old loss:", forward(params)[-1], "new loss:", forward(updated)[-1])

    output = Path(__file__).resolve().parent / "lesson-02-output"
    output.mkdir(exist_ok=True)
    for batch_size in (1, 2, 6):
        records = train(batch_size=batch_size)
        with (output / f"batch-{batch_size}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("step", "processed_samples", "full_data_mse", "w", "b"))
            writer.writerows(records)
        step, seen, loss, w, b = records[-1]
        print(f"batch={batch_size}, epoch={seen/6:.0f}, steps={step}, MSE={loss:.6f}, w={w:.5f}, b={b:.5f}")
    print("CSV training records:", output)


if __name__ == "__main__":
    main()
