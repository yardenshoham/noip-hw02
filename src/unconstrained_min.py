import math
import utils
import numpy as np


def line_search(dir_selection_method, f, x0, step_size, obj_tol, param_tol, max_iter, init_step_len=1.0, slope_ratio=1e-4, back_track_factor=0.2):
    x_prev = x0
    f_prev, df_prev, hessian_prev = f(x0, dir_selection_method != 'gd')
    path = [x_prev]
    i = 0
    success = False
    step_alpha = step_size if dir_selection_method == 'gd' else init_step_len
    while i < max_iter:
        if dir_selection_method == 'gd':
            selected_dir = -df_prev
        else:
            selected_dir = newton_dir(hessian_prev, df_prev)

        x_next = x_prev + step_alpha * selected_dir
        f_next, df_next, hessian_next = f(x_next, dir_selection_method != 'gd')

        if dir_selection_method == 'bfgs':
            selected_dir, hessian_next = bfgs_dir(
                hessian_prev, x_next, x_prev, df_next, df_prev)

        # Wolfe
        while dir_selection_method != 'gd' and (np.isnan(f_next) or f_next > f_prev + np.linalg.norm(slope_ratio * step_alpha * df_prev * selected_dir)):
            step_alpha *= back_track_factor
            x_next = x_prev + step_alpha * selected_dir
            if dir_selection_method == 'bfgs':
                f_next, df_next, _ = f(x_next, False)
            else:
                f_next, df_next, hessian_next = f(
                    x_next, dir_selection_method != 'gd')

        step_length = np.linalg.norm(x_next - x_prev)
        if step_length < param_tol:
            success = True
            print(
                f"Success! A small enough distance (<{param_tol}) between two consecutive iteration locations was achieved.")
            break
        path.append(x_next)
        change = abs(f_next - f_prev)
        if change < obj_tol:
            success = True
            print(
                f"Success! A small enough change (<{obj_tol}) in objective function values, between two consecutive iterations was achieved.")
            break
        i += 1
        utils.report(i, x_next, f_next, step_length, change)
        f_prev = f_next
        df_prev = df_next
        x_prev = x_next
        hessian_prev = hessian_next
    return x_next, success, path


def newton_dir(hessian_prev, df_prev):
    return np.linalg.solve(hessian_prev, -df_prev)


def bfgs_dir(B_k, x_kplus1, x_k, nabla_f_x_kplus1, nabla_f_x_k):
    s_k = (x_kplus1 - x_k).reshape(-1, 1)
    y_k = (nabla_f_x_kplus1 - nabla_f_x_k).reshape(-1, 1)
    B_kplus1 = B_k - (B_k @ s_k @ s_k.T @ B_k.T) / \
        (s_k.T @ B_k @ s_k) + (y_k @ y_k.T) / (y_k.T @ s_k)
    return newton_dir(B_k, nabla_f_x_k), B_kplus1
