import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from p2d.solvers.heat import HeatIrregularDomain_2d
from p2d.utils.shapes import circle, rectangle, flower

# Setup PDE parameters
mu = lambda x, y: np.ones_like(x)
f = lambda x, y: np.zeros_like(x)
g = lambda x, y: np.zeros_like(x)
alpha = 1.0
u0 = lambda x, y: np.zeros_like(x)

tmin, tmax = 0.0, 0.5
dt = 0.005

# Define 3 different interface geometries
shapes = [
    ("Circle", lambda x, y: circle(x, y, x0=0.0, y0=0.0, r=1.0)),
    ("Square", lambda x, y: rectangle(x, y, 0.0, 0.0, 2.0, 2.0)),
    ("Flower", lambda x, y: flower(x, y, r0=1.0, amplitude=0.35, n=5)),
]

# Solve on each domain (both inside and outside the interface)
results = []
for name, phi in shapes:
    solver = HeatIrregularDomain_2d(
        xrange=(-2, 2),
        yrange=(-2, 2),
        nx=128,
        ny=128,
        alpha=alpha,
        phi=phi,
        mu=mu,
        f=f,
        g=g,
    )
    solver.solve(u0, dt, trange=(tmin, tmax), solve_where="both")
    results.append((name, solver.X, solver.Y, solver.phi, solver.sol))

# Select 3 time steps: initial, intermediate, and end
num_steps = results[0][4].shape[0]
time_indices = [0, num_steps // 8, num_steps - 1]
times = [time_indices[0] * dt, time_indices[1] * dt, (num_steps - 1) * dt]

# Determine common color scaling across all panels
vmin = min(np.nanmin(res[4][time_indices, :, :]) for res in results)
vmax = max(np.nanmax(res[4][time_indices, :, :]) for res in results)

# Create 3x3 (9 panel) figure
fig, axs = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True, sharex=True, sharey=True)

for row_idx, (name, x, y, phi_grid, sol) in enumerate(results):
    for col_idx, (t_idx, t_val) in enumerate(zip(time_indices, times)):
        ax = axs[row_idx, col_idx]
        u_snapshot = sol[t_idx, :, :]

        # Unified solution field (inside and outside)
        pc = ax.pcolormesh(x, y, u_snapshot, cmap="coolwarm", shading="auto", vmin=vmin, vmax=vmax)

        # Interface zero level set
        ax.contour(x, y, phi_grid, levels=[0], colors="black", linewidths=2)

        ax.set_aspect("equal")
        ax.tick_params(labelsize=12)

        # Column titles on the top row
        if row_idx == 0:
            ax.set_title(f"$t = {t_val:.3f}$", fontsize=16)

        # Row labels / y-axis on the left column
        if col_idx == 0:
            ax.set_ylabel(f"{name}\n" + r"$y$", fontsize=14)

        # x-axis labels on the bottom row
        if row_idx == 2:
            ax.set_xlabel(r"$x$", fontsize=14)

interface_line = Line2D([], [], color="black", lw=2, label=r"$\phi=0$")
axs[0, 0].legend(handles=[interface_line], loc="upper right", fontsize=12, frameon=True)

cbar = fig.colorbar(pc, ax=axs, pad=0.02, shrink=0.85)
cbar.set_label(r"$u(x, y, t)$", fontsize=14)
cbar.ax.tick_params(labelsize=12)

output_path = "figs/example_heat.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Heat simulation completed and saved to {output_path}")
