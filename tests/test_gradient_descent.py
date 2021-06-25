import unittest
import unconstrained_min
import examples
import utils
import numpy as np


class TestGradientDescent(unittest.TestCase):

    def test_quad_min(self):
        for f in [examples.i, examples.ii, examples.iii]:
            x0 = np.array([1, 1])
            step_size = 0.1
            max_iter = 100
            param_tol = 1e-8
            obj_tol = 1e-12
            x_min, success, path = unconstrained_min.gradient_descent(
                f, x0, step_size, obj_tol, param_tol, max_iter)

            utils.plot(f, path)
            print(f"x_min={x_min} success={success}")

    def test_rosenbrock_min(self):
        f = examples.rosenbrock
        x0 = np.array([2, 2])
        step_size = 0.001
        max_iter = 10000
        param_tol = 1e-8
        obj_tol = 1e-7
        x_min, success, path = unconstrained_min.gradient_descent(
            f, x0, step_size, obj_tol, param_tol, max_iter)
        utils.plot(f, path)
        print(f"x_min={x_min} success={success}")

    def test_lin_min(self):
        f = examples.linear
        x0 = np.array([2, 2])
        step_size = 0.001
        max_iter = 10000
        param_tol = 1e-8
        obj_tol = 1e-7
        x_min, success, path = unconstrained_min.gradient_descent(
            f, x0, step_size, obj_tol, param_tol, max_iter)
        utils.plot(f, path)
        print(f"x_min={x_min} success={success}")


if __name__ == '__main__':
    unittest.main()
