from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np

from p2d.solvers.heat import HeatIrregularDomain_2d
from p2d.utils.shapes import flower


def create_heat_gif(
    tmin: float = 0.0,
    tmax: float = 0.3,
    dt: float = 0.005,
    nx: int = 256,
    ny: int = 256,
    fps: int = 20,
    output_path: str = "figs/animation_heat.gif",
):
    """
    Solve the heat equation on a flower-shaped domain and export an animated GIF.

    Parameters:
        tmin (float): Start time.
        tmax (float): End time.
        dt (float): Time step size.
        nx (int): Spatial resolution in x.
        ny (int): Spatial resolution in y.
        fps (int): Frames per second in the output GIF.
        output_path (str): Destination path for the GIF.
    """
    
    mu = lambda x, y: np.ones_like(x)
    f = lambda x, y: np.zeros_like(x)
    g = lambda x, y: np.zeros_like(x)
    alpha = 1.0
    u0 = lambda x, y: np.zeros_like(x)

    # phi = lambda x, y: circle(x, y, x0=0.0,y0=0.0,r=1.0)
    phi = lambda x, y: flower(x, y, r0=1.0, amplitude=0.35, n=5)

    solver = HeatIrregularDomain_2d(
        xrange=(-2, 2),
        yrange=(-2, 2),
        nx=nx,
        ny=ny,
        alpha=alpha,
        phi=phi,
        mu=mu,
        f=f,
        g=g,
    )

    print("Solving heat equation on flower domain...")
    solver.solve(u0, dt, trange=(tmin, tmax), solve_where="both")

    x, y = solver.X, solver.Y
    phi_grid = solver.phi
    sol = solver.sol
    num_frames = sol.shape[0]

    times = np.linspace(tmin, tmax, num_frames)
    vmin = 0.0
    vmax = alpha

    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    pc = ax.pcolormesh(
        x, y, sol[0, :, :], cmap="coolwarm", shading="auto", vmin=vmin, vmax=vmax
    )
    ax.contour(x, y, phi_grid, levels=[0], colors="black", linewidths=2)

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$y$", fontsize=13)
    ax.tick_params(labelsize=11)

    title = ax.set_title(rf"$t = {times[0]:.3f}$", fontsize=14)

    interface_line = Line2D([], [], color="black", lw=2, label=r"$\phi=0$")
    ax.legend(handles=[interface_line], loc="upper right", fontsize=11, frameon=True)

    cbar = fig.colorbar(pc, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label(r"$u(x, y, t)$", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    def update(frame_idx):
        pc.set_array(sol[frame_idx, :, :].ravel())
        title.set_text(rf"$t = {times[frame_idx]:.3f}$")
        return pc, title

    print(f"Rendering animation ({num_frames} frames)...")
    anim = FuncAnimation(fig, update, frames=num_frames, blit=False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    print(f"GIF successfully saved to {output_path}")


if __name__ == "__main__":
    create_heat_gif()
