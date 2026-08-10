# p2d: Poisson solver on an Irregular Domain

*Author*: Uri Dickman

Solves the Poisson equation on an irregular domain based on the methods from Gibou et al. (2002), *J. Comp. Phys.*

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