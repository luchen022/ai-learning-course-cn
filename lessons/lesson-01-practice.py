"""第一节：张量与必要基础。运行后将图保存到本文件旁的 lesson-01-output。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def array_examples():
    tensor = np.arange(24).reshape(2, 3, 4)
    print("Tensor:", tensor)
    print("shape:", tensor.shape, "ndim:", tensor.ndim, "size:", tensor.size)
    print("tensor[1, 2, 3]:", tensor[1, 2, 3])

    X = np.array([[1, 2], [3, 4], [5, 6]])
    for name, value in [("X[1]", X[1]), ("X[:, 0]", X[:, 0]), ("X[:, 0:1]", X[:, 0:1])]:
        print(name, "=", value, "shape:", value.shape)

    A = np.arange(1, 7).reshape(2, 3)
    print("A:", A)
    print("A.reshape(3, 2):", A.reshape(3, 2))
    print("A.T:", A.T)
    print("Broadcast row:", A + np.array([10, 20, 30]))
    print("Broadcast column:", A + np.array([[10], [20]]))
    try:
        A + np.array([10, 20])
    except ValueError as error:
        print("Expected incompatible broadcast:", error)

    prediction = np.zeros((5, 1))
    target = np.arange(5.0)
    print("Wrong error shape:", (prediction - target).shape)
    print("Correct error shape:", (prediction - target.reshape(-1, 1)).shape)


def train_linear_regression(learning_rate=0.05, steps=100):
    X = np.arange(5.0).reshape(-1, 1)
    y = 2 * X + 1
    W = np.zeros((1, 1))
    b = np.zeros((1,))
    history = [float(np.mean((X @ W + b - y) ** 2))]

    for _ in range(steps):
        prediction = X @ W + b
        if prediction.shape != y.shape:
            raise ValueError("Prediction and target shapes must match before loss computation.")
        error = prediction - y
        grad_W = 2 / len(X) * X.T @ error
        grad_b = 2 * error.mean(axis=0)
        W -= learning_rate * grad_W
        b -= learning_rate * grad_b
        loss = float(np.mean((X @ W + b - y) ** 2))
        history.append(loss)
        if not np.isfinite(loss) or loss > 1e12:
            print("Stopped: loss diverged. Try a smaller learning rate.")
            break

    print(f"lr={learning_rate}: W={W.item():.6f}, b={b.item():.6f}, MSE={history[-1]:.6f}")
    return X, y, W, b, history


def main():
    array_examples()
    X, y, W, b, history = train_linear_regression()
    output = Path(__file__).resolve().parent / "lesson-01-output"
    output.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(history)), history)
    ax.set(xlabel="Training step", ylabel="MSE", title="Linear regression training loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "loss.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(X[:, 0], y[:, 0], label="Target")
    ax.plot(X[:, 0], (X @ W + b)[:, 0], label="Prediction")
    ax.set(xlabel="x", ylabel="y", title="Target and learned regression line")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "regression.png", dpi=160)
    plt.close(fig)

    print("Plots saved to:", output)
    print("Compare learning rates:")
    for rate in (0.01, 0.05, 0.2):
        train_linear_regression(learning_rate=rate)


if __name__ == "__main__":
    main()
