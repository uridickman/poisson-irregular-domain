import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from p2d.solvers.heat import HeatIrregularDomain_2d
from p2d.utils.shapes import *

# Setup PDE parameters
mu = lambda x, y: np.ones_like(x)
f = lambda x, y: np.zeros_like(x)
g = lambda x, y: np.zeros_like(x)
phi = lambda x, y: rectangle(x,y,0.0,0.0,2.0,2.0)
alpha = 1.0
u0 = lambda x, y: np.zeros_like(x)

tmin, tmax = 0.0, 0.5
dt = 0.005

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

x, y = solver.X, solver.Y
phi_grid = solver.phi

solver.solve(u0, dt, trange=(0.0, 1.0), solve_where="both")

# Select 3 time steps: initial, halfway, and end
num_steps = solver.sol.shape[0]
time_indices = [0, num_steps // 8, num_steps - 1]
times = [tmin, tmin + (num_steps // 8) * dt, tmax]

# Plotting
vmin = np.nanmin(solver.sol[time_indices,:,:])
vmax = np.nanmax(solver.sol[time_indices,:,:])

fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True, sharey=True)

for ax, idx, t_val in zip(axs, time_indices, times):
    u_snapshot = solver.sol[idx, :, :]
    
    # Unified solution field (inside and outside)
    pc = ax.pcolormesh(x, y, u_snapshot, cmap="coolwarm", shading="auto", vmin=vmin, vmax=vmax)
    
    # Interface zero level set
    ax.contour(x, y, phi_grid, levels=[0], colors="black", linewidths=2)
    
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title(f"$t = {t_val:.3f}$", fontsize=16)

axs[0].set_ylabel(r"$y$", fontsize=14)

interface_line = Line2D([], [], color="black", lw=2, label=r"$\phi=0$")
axs[0].legend(handles=[interface_line], loc="upper right", fontsize=12, frameon=True)

cbar = fig.colorbar(pc, ax=axs, pad=0.02, shrink=0.85)
cbar.set_label(r"$u(x, y, t)$", fontsize=14)
cbar.ax.tick_params(labelsize=12)

fig.savefig("figs/example_heat.png", dpi=300, bbox_inches="tight")
print("Heat simulation completed and saved to heat_fig.png")
