import numpy as np
import math


def i(x):
    Q = np.array([[1, 0], [0, 1]])
    computed = x.transpose().dot(Q).dot(x)
    derivative = np.array([2 * x[0], 2 * x[1]])
    hessian = np.array([[2, 0], [0, 2]])
    return computed, derivative, hessian


def ii(x):
    Q = np.array([[5, 0], [0, 1]])
    computed = x.transpose().dot(Q).dot(x)
    derivative = np.array([10 * x[0], 2 * x[1]])
    hessian = np.array([[10, 0], [0, 2]])
    return computed, derivative, hessian


def iii(x):
    Q = np.array([[4, -1.7], [-1.7, 2]])
    computed = x.transpose().dot(Q).dot(x)
    derivative = np.array([4.6 * x[0] + 0.3 * x[1], 2.3 * x[0] + 0.6 * x[1]])
    hessian = np.array([[4.6, 0.3], [2.3, 0.6]])
    return computed, derivative, hessian


def rosenbrock(x):
    computed = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    derivative = np.array(
        [-400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])
    hessian = np.array([
        [1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
        [-400 * x[0], 200]
    ])
    return computed, derivative, hessian


def linear(x):
    a = np.array([1, 0])
    computed = a.transpose().dot(x)
    derivative = a
    hessian = np.array([[0, 0], [0, 0]])
    return computed, derivative
