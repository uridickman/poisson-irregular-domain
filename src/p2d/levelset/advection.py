import numpy as np
from numpy.typing import NDArray
from numba import njit
from typing import Tuple
from ..utils.time import TimeManager,TimeIntegrator
from .hj import *

def reinitialize_phi(
    phi0        : NDArray,
    dt          : float,
    dx          : float,
    dy          : float,
    max_iter    : int       = 50,
    tol         : float     = 1e-3
) -> NDArray:
    trange = (0.0, max_iter * dt)
    
    Nx, Ny = phi0.shape

    # Only compare with previous phi for convergence on
    # a band of approximately 10 grid points
    band_width = 10.0 * max(dx, dy)
    band = np.abs(phi0) < band_width

    # initialize phi to phi0
    phi = np.zeros((Nx + 6, Ny + 6))
    constant_extrapolation(phi, phi0)

    # compute the sign of phi
    S = sign(phi0, eps=min(dx, dy))
    S_ext = np.zeros_like(phi)
    constant_extrapolation(S_ext, S)
    
    integrator = TVD_RK3_Godunov(dt, dx, dy, n_direction=1)
    tm = TimeManager(trange=trange,integrator=integrator)

    while not tm.done():

        phi_old = phi.copy()

        phi_next = tm.advance(phi_old,Vn=S_ext,f=S_ext)

        # keep ghost cells consistent with the newly advanced interior
        constant_extrapolation(phi_next, phi_next[3:-3, 3:-3])

        err = np.max(np.abs(phi_next[3:-3, 3:-3][band] - phi_old[3:-3, 3:-3][band]))

        phi = phi_next

        if err < tol:
            print(f"Reinitialization converged in {tm.step} iterations.")
            return phi[3:-3, 3:-3]

    print(f"Reinitialization failed to converge in {tm.num_steps} iterations.")
    print(f"Final error = {err:e}")

    return phi[3:-3, 3:-3]


