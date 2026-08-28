import numpy as np
import scipy.special as sp_special
import matplotlib.pyplot as plt

from p2d.solvers.heat import HeatIrregularDomain_2d
from p2d.utils.shapes import circle


def test_convergence(
    dt_list=(0.004, 0.002, 0.001, 0.0005, 0.00025),
    target_dt=0.001,
    nx=128,
    ny=128,
    trange=(0.0, 0.04),
    save_fig="figs/convergence_heat.png",
):
    """
    Test 2nd-order temporal convergence of the Crank-Nicolson heat equation solver
    on an irregular domain with Dirichlet boundary conditions.

    The continuous problem inside a disk of radius R is:
        du/dt - Delta u + u = 0,    (x, y) in Omega
        u = 0,                      (x, y) on Gamma (boundary at r = R)

    The analytical solution is given by the decaying radial Bessel mode:
        u(r, t) = exp(-lambda * t) * J_0(k * r)
    where:
        - z_0 is the first positive zero of the Bessel function J_0(z) (z_0 ~= 2.4048)
        - k = z_0 / R ensures u(R, t) = 0 on Gamma
        - lambda = k^2 + 1 (accounting for diffusion and reaction coefficients)

    Because the spatial grid is fixed and fine, we compare against a high-resolution
    reference solution (dt_ref = 6.25e-5) on the same spatial discretization to isolate
    the temporal discretization error and demonstrate second-order convergence:
        ||u_dt - u_ref|| = O(dt^2)
    in both L_2 and L_inf norms.
    """
    R = 0.6
    z0 = sp_special.jn_zeros(0, 1)[0]
    k_val = z0 / R

    phi = lambda x, y: circle(x, y, 0.0, 0.0, R)
    alpha = lambda x, y: np.zeros_like(x)
    u0 = lambda x, y: sp_special.j0(k_val * np.sqrt(x**2 + y**2))

    tmin, tmax = trange

    # Compute high-resolution reference solution
    dt_ref = 0.0000625
    solver = HeatIrregularDomain_2d(
        xrange=(-1, 1),
        yrange=(-1, 1),
        nx=nx,
        ny=ny,
        alpha=alpha,
        phi=phi,
        f=0.0,
        g=0.0,
        mu=1.0,
    )

    solver.solve(u0, dt=dt_ref, trange=(tmin, tmax), solve_where="both")
    u_ref = solver.sol[-1, :, :].copy()

    mask = solver.phi < 0
    dx, dy = solver.dx, solver.dy

    l2_errors = []
    linf_errors = []
    u_target = None

    print(f"{'=' * 65}")
    print(f"{'dt':>8} | {'L_inf Error':>12} | {'L_inf Rate':>10} | {'L_2 Error':>12} | {'L_2 Rate':>10}")
    print(f"{'-' * 65}")

    for idx, dt in enumerate(dt_list):
        solver.solve(u0, dt=dt, trange=(tmin, tmax), solve_where="both")
        u_num = solver.sol[-1, :, :]

        if dt == target_dt:
            u_target = u_num.copy()

        diff = np.abs(u_ref[mask] - u_num[mask])
        linf = np.max(diff)
        l2 = np.sqrt(np.sum(diff**2 * dx * dy))

        l2_errors.append(l2)
        linf_errors.append(linf)

        if idx == 0:
            print(f"{dt:8.5f} | {linf:12.4e} | {'-':>10} | {l2:12.4e} | {'-':>10}")
        else:
            rate_linf = np.log2(linf_errors[idx - 1] / linf)
            rate_l2 = np.log2(l2_errors[idx - 1] / l2)
            print(f"{dt:8.5f} | {linf:12.4e} | {rate_linf:10.2f} | {l2:12.4e} | {rate_l2:10.2f}")

    print(f"{'=' * 65}")

    # 4-panel plot: Reference solution, Numerical solution, Absolute error, and Convergence plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 8.5), constrained_layout=True)

    x_t, y_t = solver.X, solver.Y
    phi_t = solver.phi

    u_ref_plot = np.where(mask, u_ref, np.nan)
    u_target_plot = np.where(mask, u_target, np.nan)
    err_target = np.where(mask, np.abs(u_ref - u_target), np.nan)

    vmin = np.nanmin(u_ref_plot)
    vmax = np.nanmax(u_ref_plot)

    # Top Left: Reference Solution
    pc0 = axs[0, 0].pcolormesh(x_t, y_t, u_ref_plot, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
    axs[0, 0].contour(x_t, y_t, phi_t, levels=[0], colors="k", linewidths=1.5)
    axs[0, 0].set_title(rf"Exact ($t={tmax:.2f}, \Delta t={dt_ref}$)", fontsize=13)
    axs[0, 0].set_xlabel(r"$x$", fontsize=12)
    axs[0, 0].set_ylabel(r"$y$", fontsize=12)
    axs[0, 0].set_aspect("equal")
    cbar0 = fig.colorbar(pc0, ax=axs[0, 0], fraction=0.046, pad=0.04)
    cbar0.set_label(r"$u_{\mathrm{ref}}$", fontsize=11)

    # Top Right: Numerical Solution
    pc1 = axs[0, 1].pcolormesh(x_t, y_t, u_target_plot, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
    axs[0, 1].contour(x_t, y_t, phi_t, levels=[0], colors="k", linewidths=1.5)
    axs[0, 1].set_title(rf"Numerical ($\Delta t={target_dt}$)", fontsize=13)
    axs[0, 1].set_xlabel(r"$x$", fontsize=12)
    axs[0, 1].set_ylabel(r"$y$", fontsize=12)
    axs[0, 1].set_aspect("equal")
    cbar1 = fig.colorbar(pc1, ax=axs[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_label(r"$u_h$", fontsize=11)

    # Bottom Left: Absolute Error
    pc2 = axs[1, 0].pcolormesh(x_t, y_t, err_target, cmap="inferno", shading="auto")
    axs[1, 0].contour(x_t, y_t, phi_t, levels=[0], colors="cyan", linewidths=1.5)
    axs[1, 0].set_title(rf"$|u - u_h|$ ($\Delta t={target_dt}$)", fontsize=13)
    axs[1, 0].set_xlabel(r"$x$", fontsize=12)
    axs[1, 0].set_ylabel(r"$y$", fontsize=12)
    axs[1, 0].set_aspect("equal")
    cbar2 = fig.colorbar(pc2, ax=axs[1, 0], fraction=0.046, pad=0.04, format="%.1e")
    cbar2.set_label(r"$|u - u_h|$", fontsize=11)

    # Bottom Right: Convergence Plot
    dt_arr = np.array(dt_list)
    axs[1, 1].loglog(dt_arr, l2_errors, "o-", label=r"$L^2$", base=2, lw=2)
    axs[1, 1].loglog(dt_arr, linf_errors, "s-", label=r"$L^\infty$", base=2, lw=2)
    axs[1, 1].loglog(dt_arr, l2_errors[0] * (dt_arr / dt_arr[0])**2, "--k", label=r"$\mathcal{O}(\Delta t^2)$", base=2)
    axs[1, 1].set_xlabel(r"Time step $\Delta t$", fontsize=12)
    axs[1, 1].set_ylabel(r"Error", fontsize=12)
    axs[1, 1].set_title("Convergence", fontsize=13)
    axs[1, 1].grid(True, which="both", alpha=0.3)
    axs[1, 1].legend(fontsize=11, frameon=False)

    fig.savefig(save_fig, dpi=300)
    print(f"Saved convergence plot to {save_fig}")


if __name__ == "__main__":
    test_convergence()
