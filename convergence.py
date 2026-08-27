import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from src.p2d.solvers.poisson import PoissonIrregularDomain_2d
from src.p2d.utils.shapes import flower


def test_convergence(res_list=(32, 64, 128, 256), target_res=128):
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

    u_ex_target = None
    u_num_target = None
    solver_target = None

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

        if N == target_res:
            u_ex_target = u_ex
            u_num_target = u_num
            solver_target = solver

        if idx == 0:
            print(f"{N:6d} | {linf:12.4e} | {'-':>10} | {l2:12.4e} | {'-':>10}")
        else:
            rate_linf = np.log2(linf_errors[idx - 1] / linf)
            rate_l2 = np.log2(l2_errors[idx - 1] / l2)
            print(f"{N:6d} | {linf:12.4e} | {rate_linf:10.2f} | {l2:12.4e} | {rate_l2:10.2f}")

    print(f"{'=' * 65}")

    # 4-panel plot: Exact solution, Numerical solution, Absolute error, and Convergence plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 8.5), constrained_layout=True)

    x_t, y_t = solver_target.X, solver_target.Y
    phi_t = solver_target.phi
    vmin = min(u_ex_target.min(), np.nanmin(u_num_target))
    vmax = max(u_ex_target.max(), np.nanmax(u_num_target))

    # Top Left: Exact Solution
    pc0 = axs[0, 0].pcolormesh(x_t, y_t, u_ex_target, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
    axs[0, 0].contour(x_t, y_t, phi_t, levels=[0], colors="k", linewidths=1.5)
    axs[0, 0].set_title(rf"Exact Solution ($N_x=N_y={target_res}$)", fontsize=13)
    axs[0, 0].set_xlabel(r"$x$", fontsize=12)
    axs[0, 0].set_ylabel(r"$y$", fontsize=12)
    axs[0, 0].set_aspect("equal")
    cbar0 = fig.colorbar(pc0, ax=axs[0, 0], fraction=0.046, pad=0.04)
    cbar0.set_label(r"$u_{\mathrm{exact}}$", fontsize=11)

    # Top Right: Numerical Solution
    pc1 = axs[0, 1].pcolormesh(x_t, y_t, u_num_target, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
    axs[0, 1].contour(x_t, y_t, phi_t, levels=[0], colors="k", linewidths=1.5)
    axs[0, 1].set_title(rf"Numerical Solution ($N_x=N_y={target_res}$)", fontsize=13)
    axs[0, 1].set_xlabel(r"$x$", fontsize=12)
    axs[0, 1].set_ylabel(r"$y$", fontsize=12)
    axs[0, 1].set_aspect("equal")
    cbar1 = fig.colorbar(pc1, ax=axs[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_label(r"$u_h$", fontsize=11)

    # Bottom Left: Absolute Error
    err_target = np.abs(u_ex_target - u_num_target)
    pc2 = axs[1, 0].pcolormesh(x_t, y_t, err_target, cmap="inferno", shading="auto")
    axs[1, 0].contour(x_t, y_t, phi_t, levels=[0], colors="cyan", linewidths=1.5)
    axs[1, 0].set_title(rf"Absolute Error $|u - u_h|$ ($N_x=N_y={target_res}$)", fontsize=13)
    axs[1, 0].set_xlabel(r"$x$", fontsize=12)
    axs[1, 0].set_ylabel(r"$y$", fontsize=12)
    axs[1, 0].set_aspect("equal")
    cbar2 = fig.colorbar(pc2, ax=axs[1, 0], fraction=0.046, pad=0.04, format="%.1e")
    cbar2.set_label(r"$|u - u_h|$", fontsize=11)

    # Bottom Right: Convergence Plot
    N_arr = np.array(res_list)
    axs[1, 1].loglog(N_arr, l2_errors, "o-", label=r"$L^2$", base=2, lw=2)
    axs[1, 1].loglog(N_arr, linf_errors, "s-", label=r"$L^\infty$", base=2, lw=2)
    axs[1, 1].loglog(N_arr, l2_errors[0] * (N_arr[0] / N_arr)**2, "--k", label=r"$\mathcal{O}(h^2)$", base=2)
    axs[1, 1].set_xlabel(r"Grid resolution $N$", fontsize=12)
    axs[1, 1].set_ylabel(r"Error", fontsize=12)
    axs[1, 1].set_title("Poisson Solver Convergence", fontsize=13)
    axs[1, 1].grid(True, which="both", alpha=0.3)
    axs[1, 1].legend(fontsize=11, frameon=False)

    fig.savefig("convergence_plot.png", dpi=300)
    print("Saved convergence plot to convergence_plot.png")


if __name__ == "__main__":
    test_convergence()
