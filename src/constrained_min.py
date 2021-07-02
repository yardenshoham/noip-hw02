import numpy as np
import unconstrained_min
import utils


def objective_function(t, f, ineqs):
    def phi(x):
        m = [np.log(-ineq(x)[0]) for ineq in ineqs]
        return -np.sum(m)

    def df_phi(x): return -np.sum(ineq(x)[1] / ineq(x)[0] for ineq in ineqs)

    def hessian_phi(x):
        h_phi = np.zeros((x.shape[0], x.shape[0]))
        for ineq in ineqs:
            computed, df, hessian = ineq(x)
            df = df[:, np.newaxis]
            h_phi += df @ df.T / (computed ** 2) - hessian / computed
        return h_phi

    def res(x, compute_hessian=False):
        o_computed, o_df, o_hessian = f(x, compute_hessian)
        computed = t * o_computed + phi(x)
        df = t * o_df + df_phi(x)
        if compute_hessian:
            hessian = t * o_hessian + hessian_phi(x)
        else:
            hessian = None
        return computed, df, hessian
    return res


def constrained_newton_dir(hessian, df, eq_constraints_mat):
    if len(eq_constraints_mat.shape) == 1:
        eq_constraints_mat = eq_constraints_mat[:, np.newaxis]
    if len(df.shape) == 1:
        df = df[:, np.newaxis]
    rows, columns = eq_constraints_mat.shape
    kkt_matrix = np.block(
        [[hessian, eq_constraints_mat.T], [eq_constraints_mat, np.zeros((rows, rows))]])
    rhs = np.vstack((-df, np.zeros((rows, 1))))
    return np.linalg.solve(kkt_matrix, rhs)[0:columns].reshape((columns,))


def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    t = 1
    mu = 10
    m = len(ineq_constraints)
    x_min = x0
    path = [x0]
    while m / t > 1e-5:
        the_function = objective_function(t, func, ineq_constraints)
        if eq_constraints_mat is None:
            x_min, success, line_search_path = unconstrained_min.line_search(
                "nt", the_function, x_min, step_size=0.1, obj_tol=1e-12, param_tol=1e-8, max_iter=1000)
            path.extend(line_search_path)
        else:
            x_prev = x_min
            f_prev, df_prev, hessian_prev = the_function(x_min, True)
            i = 0
            success = False
            step_alpha = 1
            while i < 1000:
                selected_dir = constrained_newton_dir(
                    hessian_prev, df_prev, eq_constraints_mat)

                x_next = x_prev + step_alpha * selected_dir
                f_next, df_next, hessian_next = the_function(x_next, True)

                # Wolfe
                while np.isnan(f_next) or f_next > f_prev + np.linalg.norm(1e-4 * step_alpha * df_prev * selected_dir):
                    step_alpha *= 0.2
                    x_next = x_prev + step_alpha * selected_dir
                    f_next, df_next, hessian_next = the_function(x_next, True)

                step_length = np.linalg.norm(x_next - x_prev)
                if step_length < 1e-8:
                    success = True
                    print(
                        f"Success! A small enough distance between two consecutive iteration locations was achieved.")
                    break
                path.append(x_next)
                change = abs(f_next - f_prev)
                if change < 1e-12:
                    success = True
                    print(
                        f"Success! A small enough change in objective function values, between two consecutive iterations was achieved.")
                    break
                i += 1
                utils.report(i, x_next, f_next, step_length, change)
                f_prev = f_next
                df_prev = df_next
                x_prev = x_next
                hessian_prev = hessian_next
            x_min = x_next
        t *= mu

    return x_min, path
