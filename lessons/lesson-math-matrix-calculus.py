"""常用向量与矩阵求导公式：代入例题，并用中心差分核对。"""
from __future__ import annotations

import numpy as np


def numerical_gradient(function, value):
    """可选核对：每次只改一个输入元素，输出为标量。"""
    step = 1e-5
    result = np.zeros_like(value)
    for index in np.ndindex(value.shape):
        plus, minus = value.copy(), value.copy()
        plus[index] += step
        minus[index] -= step
        result[index] = (function(plus) - function(minus)) / (2 * step)
    return result


def numerical_jacobian(function, value):
    """输出为向量：行对应输出、列对应输入。"""
    step = 1e-5
    result = np.zeros((function(value).size, value.size))
    for column in range(value.size):
        plus, minus = value.copy(), value.copy()
        plus[column] += step
        minus[column] -= step
        result[:, column] = (function(plus) - function(minus)) / (2 * step)
    return result


def check_scalar(name, function, value, formula_result):
    numeric = numerical_gradient(function, value)
    np.testing.assert_allclose(formula_result, numeric, rtol=1e-6, atol=1e-7)
    print(f"{name}\n  函数值：{function(value)}\n  公式代入得到的梯度：\n{formula_result}\n")


def check_vector(name, function, value, formula_result):
    numeric = numerical_jacobian(function, value)
    np.testing.assert_allclose(formula_result, numeric, rtol=1e-6, atol=1e-7)
    print(f"{name}\n  输出：{function(value)}\n  公式代入得到的雅可比：\n{formula_result}\n")


def main():
    np.set_printoptions(precision=4, suppress=True)
    a = np.array([2.0, -3.0])
    x = np.array([1.0, 4.0])
    check_scalar("V1：aᵀx+c → a", lambda x: a @ x + 5, x, a)

    x = np.array([2.0, -1.0, 3.0])
    check_scalar("V2：xᵀx → 2x", lambda x: x @ x, x, 2*x)

    B = np.array([[1.0, 0.0], [0.0, 2.0]])
    x = np.array([2.0, 1.0])
    check_scalar("V3（对称）：xᵀBx → 2Bx", lambda x: x @ B @ x, x, 2*B @ x)
    B = np.array([[1.0, 2.0], [0.0, 3.0]])
    x = np.array([1.0, 2.0])
    check_scalar("V3（一般）：xᵀBx → (B+Bᵀ)x", lambda x: x @ B @ x, x, (B+B.T) @ x)

    A = np.array([[1.0, 2.0], [0.0, 1.0]])
    b = np.array([1.0, 1.0])
    x = np.array([1.0, 2.0])
    check_scalar("V4：sum((Ax-b)²) → 2Aᵀ(Ax-b)", lambda x: np.sum((A @ x-b)**2), x, 2*A.T @ (A @ x-b))

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_scalar("M1：sum(A) → 全 1", np.sum, A, np.ones_like(A))
    check_scalar("M2：sum(A⊙A) → 2A", lambda A: np.sum(A*A), A, 2*A)
    C = np.array([[2.0, -1.0], [0.0, 3.0]])
    check_scalar("M3：sum(C⊙A) → C", lambda A: np.sum(C*A), A, C)

    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0, 5.0])
    check_scalar("M4：xᵀAy → xyᵀ", lambda A: x @ A @ y, A, np.outer(x, y))

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[1.0, 0.0], [0.0, 2.0]])
    C = np.ones((2, 2))
    check_scalar("M5：sum((AB-C)²) → 2(AB-C)Bᵀ", lambda A: np.sum((A @ B-C)**2), A, 2*(A @ B-C) @ B.T)

    A = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]])
    B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    C = np.array([[1.0, 2.0], [3.0, 4.0]])
    G = C  # f=sum(C⊙P)，M3 给出 f 对 P 的梯度。
    check_scalar("M6 左输入：P=AB，∂f/∂A=GBᵀ", lambda A: np.sum(C*(A @ B)), A, G @ B.T)
    check_scalar("M6 右输入：P=AB，∂f/∂B=AᵀG", lambda B: np.sum(C*(A @ B)), B, A.T @ G)

    A = np.array([[1.0, 2.0], [-1.0, 3.0]])
    b = np.array([0.0, 1.0])
    x = np.array([2.0, 3.0])
    check_vector("J1：Ax+b → A", lambda x: A @ x+b, x, A)
    x = np.array([2.0, 3.0])
    check_vector("J2：逐元素平方 → diag(2x)", lambda x: x*x, x, np.diag(2*x))
    check_vector("J2：逐元素指数 → diag(exp(x))", np.exp, x, np.diag(np.exp(x)))
    check_vector("J2：逐元素对数 → diag(1/x)", np.log, x, np.diag(1/x))

    a = np.array([1.0, -2.0])
    x = np.array([2.0, 1.0])
    check_scalar("组合：3xᵀx+aᵀx+7 → 6x+a", lambda x: 3*(x @ x)+a @ x+7, x, 6*x+a)
    a = np.array([1.0, 2.0])
    check_scalar("链式规则：(aᵀx)² → 2(aᵀx)a", lambda x: (a @ x)**2, x, 2*(a @ x)*a)
    print("所有公式代入结果都与中心差分一致。")


if __name__ == "__main__":
    main()
