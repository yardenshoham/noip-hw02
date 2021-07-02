import numpy as np
import math


def i(x, compute_hessian=False):
    Q = np.array([[1, 0], [0, 1]])
    computed = x.transpose().dot(Q).dot(x)
    derivative = np.array([2 * x[0], 2 * x[1]])
    if compute_hessian:
        hessian = np.array([[2, 0], [0, 2]])
    else:
        hessian = None
    return computed, derivative, hessian


def ii(x, compute_hessian=False):
    Q = np.array([[5, 0], [0, 1]])
    computed = x.transpose().dot(Q).dot(x)
    derivative = np.array([10 * x[0], 2 * x[1]])
    if compute_hessian:
        hessian = np.array([[10, 0], [0, 2]])
    else:
        hessian = None
    return computed, derivative, hessian


def iii(x, compute_hessian=False):
    Q = np.array([[4, -1.7], [-1.7, 2]])
    computed = x.transpose().dot(Q).dot(x)
    derivative = np.array([4.6 * x[0] + 0.3 * x[1], 2.3 * x[0] + 0.6 * x[1]])
    if compute_hessian:
        hessian = np.array([[4.6, 0.3], [2.3, 0.6]])
    else:
        hessian = None
    return computed, derivative, hessian


def rosenbrock(x, compute_hessian=False):
    computed = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    derivative = np.array(
        [-400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])
    if compute_hessian:
        hessian = np.array([
            [1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
            [-400 * x[0], 200]
        ])
    else:
        hessian = None
    return computed, derivative, hessian


def linear(x, compute_hessian=False):
    a = np.array([1, 0])
    computed = a.transpose().dot(x)
    derivative = a
    if compute_hessian:
        hessian = np.array([[0, 0], [0, 0]])
    else:
        hessian = None
    return computed, derivative, hessian


def qp(x, compute_hessian=False):
    computed = x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2
    derivative = np.array([2 * x[0], 2 * x[1], 2 * (x[2] + 1)])
    if compute_hessian:
        hessian = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    else:
        hessian = None
    return computed, derivative, hessian


def qp_ineq0(x):
    return -x[0], np.array([-1, 0, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def qp_ineq1(x):
    return -x[1], np.array([0, -1, 0]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def qp_ineq2(x):
    return -x[2], np.array([0, 0, -1]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def lp(x, compute_hessian=False):
    computed = -x[0] - x[1]
    derivative = np.array([-1, -1])
    if compute_hessian:
        hessian = np.array([[0, 0], [0, 0]])
    else:
        hessian = None
    return computed, derivative, hessian


def lp_ineq0(x):
    return -x[0] - x[1] + 1, np.array([-1, -1]), np.array([[0, 0], [0, 0]])


def lp_ineq1(x):
    return x[1] - 1, np.array([0, 1]), np.array([[0, 0], [0, 0]])


def lp_ineq2(x):
    return x[0] - 2, np.array([1, 0]), np.array([[0, 0], [0, 0]])


def lp_ineq3(x):
    return -x[1], np.array([0, -1]), np.array([[0, 0], [0, 0]])
