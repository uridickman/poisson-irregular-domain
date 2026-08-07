import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from p2d.poisson import PoissonIrregularDomain_2d
from p2d.shapes import *

mu = lambda x, y: np.ones_like(x)
k =  lambda x, y: np.zeros_like(x)
f =  lambda x, y: np.cos(2*np.pi*x / 2)*np.sin(2*np.pi*y / 2)
g =  lambda x, y: np.zeros_like(x)
# phi = lambda x,y: rectangle(x,y,0.0,0.0,1.5,1.5)

def phi(x, y):
    c1 = circle(x, y,  0.0,  -0.4, 0.45)
    c2 = circle(x, y,  0.45, 0.15, 0.6)
    c3 = circle(x, y, -0.60, 0.20, 0.70)

    main = np.minimum.reduce([c1,c2,c3])

    island = circle(x, y, 1.25, -1.4, 0.35)

    return np.minimum(main, island)

solver = PoissonIrregularDomain_2d(
    xrange=(-2,2),
    yrange=(-2,2),
    nx=64,
    ny=64,
    alpha=0.0,
    phi=phi,
    mu=mu,
    k=k,
    f=f,
    g=g
)

x,y = solver.X, solver.Y
phi = solver.phi

u_outside = solver.solve(solve_inside=False)
u_inside = solver.solve(solve_inside=True)

## PLOT SOLUTION

vmin = np.minimum(u_inside.min(),u_outside.min())
vmax = np.minimum(u_inside.max(),u_outside.max())

fig, axs = plt.subplots(1,2,figsize=(11, 5), constrained_layout=True, sharey=True)

# Solution field
pc = axs[0].pcolormesh(x, y, u_outside,cmap="viridis",shading="auto",vmin=vmin,vmax=vmax)
axs[1].pcolormesh(x, y, u_inside,cmap="viridis",shading="auto",vmin=vmin,vmax=vmax)

# Zero level set
axs[0].contour(x, y, phi,levels=[0],colors="k",linewidths=2)
axs[1].contour(x, y, phi,levels=[0],colors="k",linewidths=2)

interface = Line2D([], [], color="k", lw=2, label=r"$\phi=0$")

axs[1].legend(handles=[interface], fontsize=14, frameon=False)

axs[0].set_xlabel(r"$x$", fontsize=14)
axs[1].set_xlabel(r"$x$", fontsize=14)
axs[0].set_ylabel(r"$y$", fontsize=14)

axs[0].set_title("Solve outside", fontsize=16)
axs[1].set_title("Solve inside", fontsize=16)

for ax in axs.flatten():
    ax.set_aspect("equal")
    ax.tick_params(labelsize=12)

cbar = fig.colorbar(pc, ax=axs[1], pad=0.02)
cbar.set_label(r"$u$", fontsize=14)
cbar.ax.tick_params(labelsize=12)

fig.savefig("fig.png", dpi=300, bbox_inches="tight")