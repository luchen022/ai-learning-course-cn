"""向量与矩阵求导：解析梯度、数值差分与 PyTorch 自动求导核对。"""

from __future__ import annotations

import numpy as np
import torch


EPS = 1e-6


def finite_difference_vector(function, x: np.ndarray) -> np.ndarray:
    """用中心差分计算标量函数对向量的梯度。"""
    gradient = np.zeros_like(x, dtype=np.float64)
    for index in np.ndindex(x.shape):
        plus = x.copy()
        minus = x.copy()
        plus[index] += EPS
        minus[index] -= EPS
        gradient[index] = (function(plus) - function(minus)) / (2 * EPS)
    return gradient


def finite_difference_jacobian(function, x: np.ndarray) -> np.ndarray:
    """用中心差分计算向量函数对向量的雅可比矩阵。"""
    output = function(x)
    jacobian = np.zeros((output.size, x.size), dtype=np.float64)
    for column in range(x.size):
        plus = x.copy()
        minus = x.copy()
        plus[column] += EPS
        minus[column] -= EPS
        jacobian[:, column] = (function(plus) - function(minus)) / (2 * EPS)
    return jacobian


def linear_loss(W: np.ndarray, x: np.ndarray, b: np.ndarray, target: np.ndarray) -> float:
    error = W @ x + b - target
    return float(0.5 * error @ error)


def check_vector_gradient() -> None:
    x = np.array([2.0, 1.0])
    analytic = np.array([2 * x[0], 4 * x[1]])
    numeric = finite_difference_vector(lambda value: value[0] ** 2 + 2 * value[1] ** 2, x)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-7)
    print("1. 标量对向量：∇L =", analytic, "（数值差分一致）")
    delta_x = np.array([0.1, 0.0])
    predicted_change = analytic @ delta_x
    actual_change = (x[0] + 0.1) ** 2 + 2 * x[1] ** 2 - (x[0] ** 2 + 2 * x[1] ** 2)
    print(f"   Δx={delta_x} 时，梯度预测 ΔL≈{predicted_change:.4f}，真实 ΔL={actual_change:.4f}")


def check_jacobian() -> None:
    W = np.array([[1.0, 2.0], [-1.0, 3.0]])
    b = np.array([0.0, 1.0])
    x = np.array([1.0, 2.0])
    numeric = finite_difference_jacobian(lambda value: W @ value + b, x)
    np.testing.assert_allclose(W, numeric, rtol=1e-6, atol=1e-7)
    print("2. 向量对向量：∂y/∂x =\n", W, "\n   （数值差分一致）", sep="")
    delta_x = np.array([0.0, 0.1])
    print("   只让 x₂ 增加 0.1 时，Δy = JΔx =", W @ delta_x)


def check_single_sample() -> None:
    W = np.array([[1.0, -1.0, 2.0], [0.5, 1.0, -1.0]])
    x = np.array([2.0, 1.0, -1.0])
    b = np.array([0.0, 1.0])
    target = np.array([0.0, 2.0])

    error = W @ x + b - target
    grad_W = np.outer(error, x)
    grad_b = error
    grad_x = W.T @ error

    numeric_W = finite_difference_vector(lambda value: linear_loss(value, x, b, target), W)
    numeric_b = finite_difference_vector(lambda value: linear_loss(W, x, value, target), b)
    numeric_x = finite_difference_vector(lambda value: linear_loss(W, value, b, target), x)

    np.testing.assert_allclose(grad_W, numeric_W, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(grad_b, numeric_b, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(grad_x, numeric_x, rtol=1e-6, atol=1e-7)

    print("3. 单样本线性层：")
    print("   ∂L/∂W =\n", grad_W)
    print("   ∂L/∂b =", grad_b)
    print("   ∂L/∂x =", grad_x, "（三项均通过数值差分）")
    changed_W = W.copy()
    changed_W[0, 0] += 0.1
    predicted_change = grad_W[0, 0] * 0.1
    actual_change = linear_loss(changed_W, x, b, target) - linear_loss(W, x, b, target)
    print(f"   W₁₁ 增加 0.1：梯度预测 ΔL≈{predicted_change:.4f}，真实 ΔL={actual_change:.4f}")


def check_batch_and_autograd() -> None:
    X = np.array([[2.0, 1.0, -1.0], [0.0, 2.0, 1.0]])
    W = np.array([[1.0, -1.0, 2.0], [0.5, 1.0, -1.0]])
    b = np.array([0.0, 1.0])
    target = np.array([[0.0, 2.0], [1.0, 0.0]])

    prediction = X @ W.T + b
    upstream = (prediction - target) / X.shape[0]  # 批次平均损失
    grad_W = upstream.T @ X
    grad_b = upstream.sum(axis=0)
    grad_X = upstream @ W

    X_t = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    W_t = torch.tensor(W, dtype=torch.float64, requires_grad=True)
    b_t = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    target_t = torch.tensor(target, dtype=torch.float64)
    prediction_t = X_t @ W_t.T + b_t
    loss_t = 0.5 * ((prediction_t - target_t) ** 2).sum(dim=1).mean()
    loss_t.backward()

    np.testing.assert_allclose(grad_W, W_t.grad.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(grad_b, b_t.grad.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(grad_X, X_t.grad.numpy(), rtol=1e-10, atol=1e-10)

    print("4. 批次公式与 PyTorch autograd 一致：")
    print("   G.T @ X      ->", grad_W.shape)
    print("   G.sum(axis=0)->", grad_b.shape)
    print("   G @ W        ->", grad_X.shape)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    check_vector_gradient()
    check_jacobian()
    check_single_sample()
    check_batch_and_autograd()
    print("\n全部检查通过。先用形状推导，再用数值差分和自动求导核对。")
