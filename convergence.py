import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from src.p2d.solvers.poisson import PoissonIrregularDomain_2d
from src.p2d.utils.shapes import flower


def test_convergence(res_list=(32, 64, 128, 256)):
    # 1. Manufactured solution: u(x,y) = sin(pi*x)*sin(pi*y), f = Delta u
    x, y = sp.symbols("x y")
    u_sym = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    f_sym = sp.diff(u_sym, x, 2) + sp.diff(u_sym, y, 2)

    u_exact_fn = sp.lambdify((x, y), u_sym, "numpy")
    f_fn = sp.lambdify((x, y), f_sym, "numpy")

    mu = lambda x, y: np.ones_like(x)
    k = lambda x, y: np.zeros_like(x)
    phi = lambda x, y: flower(x, y, r0=0.5)

    l2_errors = []
    linf_errors = []

    print(f"{'=' * 65}")
    print(f"{'N':>6} | {'L_inf Error':>12} | {'L_inf Rate':>10} | {'L_2 Error':>12} | {'L_2 Rate':>10}")
    print(f"{'-' * 65}")

    for idx, N in enumerate(res_list):
        solver = PoissonIrregularDomain_2d(
            xrange=(-1, 1),
            yrange=(-1, 1),
            nx=N,
            ny=N,
            alpha=u_exact_fn,
            phi=phi,
            mu=mu,
            k=k,
            f=f_fn,
            g=u_exact_fn,
            reinit=False
        )

        u_num = solver.solve(solve_where="both")
        u_ex = u_exact_fn(solver.X, solver.Y)

        mask = ~np.isnan(u_num)
        dx, dy = solver.dx, solver.dy
        diff = np.abs(u_ex[mask] - u_num[mask])

        linf = np.max(diff)
        l2 = np.sqrt(np.sum(diff**2 * dx * dy))

        l2_errors.append(l2)
        linf_errors.append(linf)

        if idx == 0:
            print(f"{N:6d} | {linf:12.4e} | {'-':>10} | {l2:12.4e} | {'-':>10}")
        else:
            rate_linf = np.log2(linf_errors[idx - 1] / linf)
            rate_l2 = np.log2(l2_errors[idx - 1] / l2)
            print(f"{N:6d} | {linf:12.4e} | {rate_linf:10.2f} | {l2:12.4e} | {rate_l2:10.2f}")

    print(f"{'=' * 65}")

    # Plot convergence
    N_arr = np.array(res_list)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.loglog(N_arr, l2_errors, "o-", label=r"$L^2$", base=2, lw=2)
    ax.loglog(N_arr, linf_errors, "s-", label=r"$L^\infty$", base=2, lw=2)
    ax.loglog(N_arr, l2_errors[0] * (N_arr[0] / N_arr)**2, "--k", label=r"$\mathcal{O}(h^2)$", base=2)

    ax.set_xlabel(r"Grid resolution $N$", fontsize=13)
    ax.set_ylabel(r"Error", fontsize=13)
    ax.set_title("Poisson Solver Convergence (Both Inside & Outside)", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=12, frameon=False)
    fig.savefig("convergence_plot.png", dpi=300)
    print("Saved convergence plot to convergence_plot.png")


if __name__ == "__main__":
    test_convergence()
