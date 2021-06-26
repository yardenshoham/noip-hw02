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
            F[i][j], _, _ = func(x)

    fig, ax = plt.subplots()
    cp = ax.contour(X1, X2, F)
    ax.clabel(cp, inline=1, fontsize=10)

    ax.scatter(x_1, x_2, s=0.8)
    plt.savefig(f"{func.__name__}.png")
    plt.clf()
    plt.cla()


def plot_iterations(func, path):
    func_values = [func(x)[0] for x in path]
    plt.plot(func_values)
    plt.savefig(f"{func.__name__}_iterations.png")
    plt.clf()
    plt.cla()
