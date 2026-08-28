import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from p2d.solvers.poisson import PoissonIrregularDomain_2d
from p2d.utils.shapes import circle, rectangle, flower

# Setup PDE parameters
mu = lambda x, y: np.ones_like(x)
k = lambda x, y: np.zeros_like(x)
f = lambda x, y: np.cos(2 * np.pi * x / 2) * np.sin(2 * np.pi * y / 2)
g = lambda x, y: np.zeros_like(x)
alpha = 0.0

# Define 3 different interface geometries
shapes = [
    ("Circle", lambda x, y: circle(x, y, x0=0.0, y0=0.0, r=1.0)),
    ("Rectangle", lambda x, y: rectangle(x, y, 0.0, 0.0, 2.0, 2.0)),
    ("Flower", lambda x, y: flower(x, y, r0=1.0, amplitude=0.35, n=5)),
]

# Solve on each domain (both inside and outside the interface)
results = []
for name, phi in shapes:
    solver = PoissonIrregularDomain_2d(
        xrange=(-2, 2),
        yrange=(-2, 2),
        nx=256,
        ny=256,
        alpha=alpha,
        phi=phi,
        mu=mu,
        k=k,
        f=f,
        g=g,
    )
    u = solver.solve(solve_where="both")
    results.append((name, solver.X, solver.Y, solver.phi, u))

# Determine common color scaling across panels
vmin = min(np.nanmin(res[4]) for res in results)
vmax = max(np.nanmax(res[4]) for res in results)

# Plot solution on a 3-panel figure
fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True, sharey=True)

for ax, (name, x, y, phi_grid, u) in zip(axs, results):
    pc = ax.pcolormesh(x, y, u, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
    ax.contour(x, y, phi_grid, levels=[0], colors="k", linewidths=2)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_title(name, fontsize=16)
    ax.tick_params(labelsize=12)

axs[0].set_ylabel(r"$y$", fontsize=14)

interface_line = Line2D([], [], color="k", lw=2, label=r"$\phi=0$")
axs[0].legend(handles=[interface_line], loc="upper right", fontsize=12, frameon=True)

cbar = fig.colorbar(pc, ax=axs, pad=0.02, shrink=0.85)
cbar.set_label(r"$u$", fontsize=14)
cbar.ax.tick_params(labelsize=12)

output_path = "figs/example_poisson.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Poisson simulation completed and saved to {output_path}")