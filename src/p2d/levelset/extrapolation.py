import numpy as np
from numpy.typing import NDArray
from numba import njit
from typing import Tuple
from ..solvers.time import TimeManager,TimeIntegrator
from .hj import *

@njit(cache=True)
def _partial_ux(u,u_interface,phi,Nx,Ny,dx,dy):
    out = np.zeros((Nx,Ny))
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            
            phi_ij = phi[i,j]
            phi_ip1j = phi[i+1,j]
            phi_im1j = phi[i-1,j]

            u_ij = u[i,j]
            u_ip1j = u[i+1,j]
            u_im1j = u[i-1,j]

            # CASE 1: all nodes on same side
            if phi_ij * phi_ip1j > 0 and phi_ij * phi_im1j > 0: 
                if np.abs(phi_ip1j) <= np.abs(phi_im1j):
                    ux = (u_ip1j - u_ij) / dx
                else:
                    ux = (u_ij - u_im1j) / dx

            # CASE 2: i-1 and i same side, i+1 opposite
            elif phi_im1j * phi_ij > 0:
                if np.abs(phi_ij) > dx**2:
                    ux = (u_ij - uim1j) / dx
                else:
                    theta = np.abs(phi_ij) / (np.abs(phi_ij) + np.abs(phi_ip1j))
                    ux = (u_interface - u_ij) / theta / dx

            # CASE 3: i and i+1 same side, i-1 opposite
            elif phi_ip1j * phi_ij > 0:
                if np.abs(phi_ij) > dx**2:
                    ux = (u_ip1j - u_ij) / dx
                else:
                    theta = np.abs(phi_ij) / (np.abs(phi_ij) + np.abs(phi_im1j))
                    ux = (u_ij - u_interface) / theta / dx

            # CASE 4: i is one side, both neighbors opposite
            else:
                ux = 0.0
                # theta_m = np.abs(phi_ij) / (np.abs(phi_ij) + np.abs(phi_im1j))
                # theta_p = np.abs(phi_ij) / (np.abs(phi_ij) + np.abs(phi_ip1j))

                # if (2 - theta_p - theta_m) < dx:
                #     if np.abs(phi_ip1j) > np.abs(phi_im1j):
                #         ux = (u_interface - u_ij) / theta_p / dx
                #     else:
                #         ux = (u_ij - u_interface) / theta_m / dx
                # else:
                #     ux = 0

            out[i,j] = ux
        
    return out

@njit(cache=True)
def _partial_uy(u,u_interface,phi,Nx,Ny,dx,dy):
    out = np.zeros((Nx,Ny))
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            
            phi_ij = phi[i,j]
            phi_ijp1 = phi[i,j+1]
            phi_ijm1 = phi[i,j-1]

            u_ij = u[i,j]
            u_ijp1 = u[i,j+1]
            u_ijm1 = u[i,j-1]

            # CASE 1: all nodes on same side
            if phi_ij * u_ijp1 > 0 and phi_ij * u_ijm1 > 0:
                if np.abs(phi_ijp1) <= np.abs(phi_ijm1):
                    uy = (u_ijp1 - u_ij) / dy
                else:
                    uy = (u_ij - u_ijm1) / dy

            # CASE 2: j-1 and hj same side, j+1 opposite
            elif u_ijm1 * phi_ij > 0:
                if np.abs(phi_ij) > dy**2:
                    uy = (u_ij - uijm1) / dy
                else:
                    theta = np.abs(phi_ij) / (np.abs(phi_ij) + np.abs(phi_ijp1))
                    uy = (u_interface - u_ij) / theta / dy

            # CASE 3: j and j+1 same side, j-1 opposite
            elif phi_ijp1 * phi_ij > 0:
                if np.abs(phi_ij) > dy**2:
                    uy = (u_ijp1 - u_ij) / dy
                else:
                    theta = np.abs(phi_ij) / (np.abs(phi_ij) + np.abs(phi_im1j))
                    uy = (u_ij - u_interface) / theta / dy

            # CASE 4: j is one side, both neighbors opposite
            else:
                uy = 0.0
                # theta_m = np.abs(phi_ij) / (np.abs(phi_ij) + np.abs(phi_ijm1))
                # theta_p = np.abs(phi_ij) / (np.abs(phi_ij) + np.abs(phi_ijp1))

                # if (2 - theta_p - theta_m) < dy:
                #     if np.abs(phi_ip1j) > np.abs(phi_im1j):
                #         uy = (u_interface - u_ij) / theta_p / dy
                #     else:
                #         uy = (u_ij - u_interface) / theta_m / dy
                # else:
                #     uy = 0

            out[i,j] = uy
        
    return out

@njit(cache=True)
def _godunov_extrapolation_Euler_step(
    u_prev      : NDArray,
    phi         : NDArray,
    S           : NDArray,
    u_interface : float,
    dt          : float,
    dx          : float,
    dy          : float,
    Nx          : int,
    Ny          : int
) -> NDArray:

    phi_plus_minus_x = _WENO5_2d_step(phi, dx, 0)
    phi_plus_minus_y = _WENO5_2d_step(phi, dy, 1)

    phi_x = _godunov_partial_vectorized_signed(phi_plus_minus_x[0], phi_plus_minus_x[1], S)
    phi_y = _godunov_partial_vectorized_signed(phi_plus_minus_y[0], phi_plus_minus_y[1], S)

    grad_phi = np.sqrt(phi_x**2 + phi_y**2) + 1e-12

    u_x = _partial_ux(u_prev, u_interface, phi, Nx, Ny, dx, dy)
    u_y = _partial_uy(u_prev, u_interface, phi, Nx, Ny, dx, dy)

    u_new = u_prev - S * dt * (phi_x * u_x + phi_y * u_y) / grad_phi

    constant_extrapolation(u_new, u_new[3:-3, 3:-3].copy())

    return u_new