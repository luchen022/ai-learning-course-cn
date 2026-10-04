"""普通数学函数的向量与矩阵求导：逐元素偏导和中心差分核对。"""
from __future__ import annotations

import numpy as np

EPS = 1e-5


def gradient_by_difference(function, value):
    """标量函数：每次只改一个输入元素，其他元素固定。"""
    gradient = np.zeros_like(value, dtype=float)
    for index in np.ndindex(value.shape):
        plus, minus = value.copy(), value.copy()
        plus[index] += EPS
        minus[index] -= EPS
        gradient[index] = (function(plus) - function(minus)) / (2 * EPS)
    return gradient


def jacobian_by_difference(function, value):
    """多个输出：输入与输出都按行展开，行对应输出、列对应输入。"""
    jacobian = np.zeros((function(value).size, value.size))
    for column, index in enumerate(np.ndindex(value.shape)):
        plus, minus = value.copy(), value.copy()
        plus[index] += EPS
        minus[index] -= EPS
        jacobian[:, column] = ((function(plus) - function(minus)) / (2 * EPS)).ravel()
    return jacobian


def vector_scalar(x):
    return x[0] ** 2 + 2 * x[1] ** 2


def vector_output(x):
    return np.array([x[0] ** 2 + x[1], x[0] * x[1]])


def matrix_scalar(A):
    a, b, c, d = A.ravel()
    return a**2 + 3 * b**2 + c * d


def matrix_square(A):
    return A @ A


def main():
    np.set_printoptions(precision=4, suppress=True)
    x = np.array([2.0, 1.0])
    gradient = np.array([2 * x[0], 4 * x[1]])
    np.testing.assert_allclose(gradient, gradient_by_difference(vector_scalar, x), atol=1e-7)
    delta = np.array([0.1, 0.0])
    print("1. f(x)=x₁²+2x₂²，梯度：", gradient)
    print("   ∇f·Δx：", gradient @ delta, "；实际变化：", vector_scalar(x + delta) - vector_scalar(x))

    x = np.array([2.0, 3.0])
    jacobian = np.array([[2 * x[0], 1], [x[1], x[0]]])
    np.testing.assert_allclose(jacobian, jacobian_by_difference(vector_output, x), atol=1e-7)
    print("2. F(x)=[x₁²+x₂,x₁x₂]，雅可比：\n", jacobian)
    print("   JΔx：", jacobian @ delta, "；实际变化：", vector_output(x + delta) - vector_output(x))

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    a, b, c, d = A.ravel()
    gradient = np.array([[2 * a, 6 * b], [d, c]])
    np.testing.assert_allclose(gradient, gradient_by_difference(matrix_scalar, A), atol=1e-7)
    H = np.array([[0.0, 0.1], [0.0, 0.0]])
    print("3. f(A)=a²+3b²+cd，矩阵梯度：\n", gradient)
    print("   对应位置乘积之和：", np.sum(gradient * H), "；实际变化：", matrix_scalar(A + H) - matrix_scalar(A))

    jacobian = np.array([[2*a, c, b, 0], [b, a+d, 0, b], [c, 0, a+d, c], [0, c, b, 2*d]])
    np.testing.assert_allclose(jacobian, jacobian_by_difference(matrix_square, A), atol=1e-7)
    np.testing.assert_allclose((jacobian @ H.ravel()).reshape(2, 2), A @ H + H @ A)
    print("4. F(A)=A @ A，展开后的雅可比：\n", jacobian)
    print("   导数作用于 H：\n", A @ H + H @ A)
    print("   实际变化：\n", matrix_square(A + H) - matrix_square(A))

    def composite(x):
        u, v = x[0] + x[1], x[0] - x[1]
        return u**2 + 3 * v**2

    x = np.array([2.0, 1.0])
    u, v = x[0] + x[1], x[0] - x[1]
    gradient = np.array([2*u + 6*v, 2*u - 6*v])
    np.testing.assert_allclose(gradient, gradient_by_difference(composite, x), atol=1e-7)
    print("5. u=x₁+x₂，v=x₁−x₂，f=u²+3v²，梯度：", gradient)
    print("所有手推导数都与中心差分一致。")


if __name__ == "__main__":
    main()
