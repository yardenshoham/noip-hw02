import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def report(i, x_curr, f_curr, step_length, change):
    print(
        f"Iteration {i}: f({x_curr})={f_curr}, step length={step_length}, change={change}")


def plot(func, path):
    x_1 = np.array([x1 for x1, _ in path])
    x_2 = np.array([x2 for _, x2 in path])
    points = np.array([p for p in path])

    X1, X2 = np.meshgrid(np.linspace(-1.2 * abs(max(x_1)), 1.2 * abs(max(x_1))),
                         np.linspace(-1.2 * abs(max(x_2)), 1.2 * abs(max(x_2))))
    F = np.empty(shape=(X1.shape[0], X1.shape[1]))

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = np.array([X1[i][j], X2[i][j]])
            F[i][j], _, _ = func(x, False)

    fig, ax = plt.subplots()
    cp = ax.contour(X1, X2, F)
    ax.clabel(cp, inline=1, fontsize=10)

    ax.scatter(x_1, x_2, s=0.8)
    plt.savefig(f"{func.__name__}.png")
    plt.clf()
    plt.cla()


def plot_iterations(func, path):
    func_values = [func(x, False)[0] for x in path]
    plt.plot(func_values)
    plt.savefig(f"{func.__name__}_iterations.png")
    plt.clf()
    plt.cla()


def plot_constrained_qp(path, x_min):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('feasible region')
    ax.view_init(elev=19, azim=-29)

    # path
    xs, ys, zs = [], [], []
    for x, y, z in path:
        xs.append(x)
        ys.append(y)
        zs.append(z)
    ax.plot(xs, ys, zs, label='path')
    rounded_x = np.around(x_min[0], 2)
    rounded_y = np.around(x_min[1], 2)
    rounded_z = np.around(x_min[2], 2)
    ax.scatter(x_min[0], x_min[1], x_min[2],
               label=f'final candidate ({rounded_x}, {rounded_y}, {rounded_z})')

    # feasible region
    xs, ys, zs = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ax.plot_trisurf(xs, ys, zs, alpha=0.3)

    ax.legend()
    # plt.show()
    plt.savefig(f"plot_constrained_qp.png")
    plt.clf()
    plt.cla()


def plot_constrained_lp(path, x_min):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title('feasible region')

    # path
    xs, ys = [], []
    for x, y in path:
        xs.append(x)
        ys.append(y)
    ax.plot(xs, ys, label='path')
    rounded_x = np.around(x_min[0], 2)
    rounded_y = np.around(x_min[1], 2)
    ax.scatter(x_min[0], x_min[1],
               label=f'final candidate ({rounded_x}, {rounded_y})')

    # feasible region
    x = np.arange(0, 2, 0.01)
    y = -x + 1
    ax.fill_between(x, y, 1, alpha=0.3)

    ax.legend()
    plt.savefig(f"plot_constrained_lp.png")
    plt.clf()
    plt.cla()
