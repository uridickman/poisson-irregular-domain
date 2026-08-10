# p2d: Poisson solver on an Irregular Domain

*Author*: Uri Dickman

Solves the Poisson equation on an irregular domain on a Cartesian grid based on the methods from Gibou et al. (2002), *J. Comp. Phys.*

### Problem statement

```math
\begin{align*}
ku-\nabla\cdot(\mu\nabla u)&=f,\quad x,y \in \Omega \\
u &= \alpha,\quad x,y \in \Gamma \\
u &= g,\quad x,y \in \partial \Omega
\end{align*}
```

where $f$ and $g$ are functions of $x$ and $y$, $\alpha$ is a constant, and $k$ and $\mu$ may be spatially varying.
The domain $\Omega=\Omega^\pm$, so the solution $u(x,y)$ is only defined **either** inside or outside the interface $\Gamma$, where
```math
\begin{equation*}
\begin{cases}
\Omega^-&: \{x,y\in\mathbb{R} \space | \space \phi(x,y)<0\} \\
\Gamma&: \{x,y\in\mathbb{R} \space | \space \phi(x,y)=0\} \\
\Omega^+&: \{x,y\in\mathbb{R} \space | \space \phi(x,y)>0\}
\end{cases}

\end{equation*}
```

### Methods

The level-set function defining the interface is re-initialized to be a signed distance function using 5th order WENO scheme with the Godunov flux-preserving method. The re-initialization solves
```math
\phi_t+S(\phi_0)(|\nabla\phi|-1)=0
```
with constant extrapolation on the wall boundaries, since $\phi$ is only relevant near the interface boundary.
The re-initialization procedure is run with a pseudo-timestep $\Delta t$ using the Total Variation Diminishing 3-step Runga-Kutta (TVD RK3) method until $\max_{x,y}|\phi(t_{n})-\phi(t_{n-1})|<\text{NTOL}$, or until the maximum number of iterations is reached.

Then, a linear system is constructed to solve the Poisson equation on the domain defined by the wall and interface boundaries.
The linear system is symmetric because linear extrapolation is used to determine the values of ghost-nodes across the interface boundary.
Overall, the scheme is second-order in space.

### Usage

Create a virtual environment:
```bash
uv venv /path/to/venv --python 3.14
source /path/to/venv/bin/activate
uv pip install -e .
```

Run the example:
```bash
python example.py
```