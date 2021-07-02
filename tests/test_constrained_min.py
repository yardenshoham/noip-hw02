import unittest
import constrained_min
import examples
import utils
import numpy as np


class TestConstrainedMin(unittest.TestCase):
    def test_qp(self):
        x_min, path = constrained_min.interior_pt(examples.qp,
                                                  [examples.qp_ineq0, examples.qp_ineq1,
                                                   examples.qp_ineq2],
                                                  np.array([[1, 1, 1]]),
                                                  np.array([1]),
                                                  np.array([0.1, 0.2, 0.7]))
        print(f"x_min={x_min}")
        utils.plot_constrained_qp(path, x_min)

    def test_lp(self):
        x_min, path = constrained_min.interior_pt(examples.lp,
                                                  [examples.lp_ineq0, examples.lp_ineq1,
                                                   examples.lp_ineq2, examples.lp_ineq3],
                                                  None,
                                                  None,
                                                  np.array([0.5, 0.75]))
        print(f"x_min={x_min}")
        utils.plot_constrained_lp(path, x_min)


if __name__ == '__main__':
    unittest.main()
