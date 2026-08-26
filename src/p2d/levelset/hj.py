import numpy as np
from numpy.typing import NDArray
from numba import njit
from typing import Tuple
from ..utils.time import TimeManager,TimeIntegrator


@njit(cache=True)
def constant_extrapolation(
    phi_extrap  : NDArray,
    phi         : NDArray
) -> None:
    """Extrapolate the boundary ghost nodes by padding the boundary with the boundary value.
    WENO requires 3 nodes on each side, so pad 3 on each side.

    Args:
        phi_extrap (NDArray): N+3 x N+3 array in which to place the extrapolated phi
        phi (NDArray): N x N array containing the level set function values 
    """
    
    phi_extrap[3:-3, 3:-3] = phi

    for i in range(3):
        phi_extrap[i, 3:-3]      = phi[0, :]
        phi_extrap[-(i+1), 3:-3] = phi[-1, :]
        phi_extrap[3:-3, i]      = phi[:, 0]
        phi_extrap[3:-3, -(i+1)] = phi[:, -1]


@njit(cache=True)
def sign(
    phi : NDArray,
    eps : float =1e-3
) -> NDArray:
    """Smoothed function Sign(phi), given by
    
    S(ϕ) = ϕ / √(ϕ^2+ε^2)

    Args:
        phi (NDArray): level-set function
        eps (float, optional): small value. Defaults to 1e-3.

    Returns:
        NDArray: smoothed Sign function
    """
    
    return phi / np.sqrt(phi*phi + eps*eps)


@njit(cache=True)
def _WENO5_2d_step(
    phi         : NDArray,
    dx          : np.float64,
    direction   : int
) -> Tuple[NDArray, NDArray]:
    """Computes a 5th order WENO step for computing ∂ϕ+/∂x and ∂ϕ-/∂x
    required by the Godunov scheme.

    Args:
        phi (NDArray): level-set function
        dx (np.float64): grid spacing
        direction (int): direction to compute partials. 0 for x, 1 for y.

    Raises:
        ValueError: raised when direction is neither 0 nor 1.or

    Returns:
        Tuple[NDArray, NDArray]: ∂ϕ+/∂x and ∂ϕ-/∂x
    """
    
    Nx,Ny = phi.shape
    partial_phi_plus = np.zeros_like(phi)
    partial_phi_minus = np.zeros_like(phi)

    if direction == 0:
        for j in range(Ny):
            phi_slice = phi[:,j]
            partial_phi_plus[3:-3,j] = _WENO_partial_phi_vectorized(phi_slice,dx,True)
            partial_phi_minus[3:-3,j] = _WENO_partial_phi_vectorized(phi_slice,dx,False)
    elif direction == 1:
        for i in range(Nx):
            phi_slice = phi[i,:]
            partial_phi_plus[i,3:-3] = _WENO_partial_phi_vectorized(phi_slice,dx,True)
            partial_phi_minus[i,3:-3] = _WENO_partial_phi_vectorized(phi_slice,dx,False)
    else:
        raise ValueError(f"direction = {direction} is not defined.")

    return partial_phi_plus,partial_phi_minus


@njit(cache=True)
def _WENO_partial_phi_vectorized(
    phi     : NDArray,
    dx      : float,
    is_plus : bool
) -> NDArray:
    """Computes either ∂ϕ+/∂x or ∂ϕ-/∂x using 5th order WENO.

    Args:
        phi (_type_): level-set function
        dx (_type_): grif spacing
        is_plus (bool): if is_plus, compute ∂ϕ+/∂x, otherwise compute ∂ϕ-/∂x.

    Returns:
        NDArray: array containing either ∂ϕ+/∂x or ∂ϕ-/∂x
    """
    
    idx = np.arange(3,len(phi)-3,dtype=np.int64)
    d1 = (phi[idx-2] - phi[idx-3]) / dx
    d2 = (phi[idx-1] - phi[idx-2]) / dx
    d3 = (phi[idx]   - phi[idx-1]) / dx
    d4 = (phi[idx+1] - phi[idx])   / dx
    d5 = (phi[idx+2] - phi[idx+1]) / dx
    d6 = (phi[idx+3] - phi[idx+2]) / dx

    if is_plus:
        _,d5,d4,d3,d2,d1 = (d1,d2,d3,d4,d5,d6)

    d_max = d1*d1
    d_max = np.maximum(d_max, d2*d2)
    d_max = np.maximum(d_max, d3*d3)
    d_max = np.maximum(d_max, d4*d4)
    d_max = np.maximum(d_max, d5*d5)

    u1x = d1/3-7/6*d2+11/6*d3
    u2x = -d2/6+5/6*d3+d4/3
    u3x = d3/3+5/6*d4-d5/6

    S1 = 13/12*(d1-2*d2+d3)**2+1/4*(d1-4*d2+3*d3)**2
    S2 = 13/12*(d2-2*d3+d4)**2+1/4*(d2-d4)**2
    S3 = 13/12*(d3-2*d4+d5)**2+1/4*(3*d3-4*d4+d5)**2

    eps = 1e-6*d_max + 1e-99
    alpha1 = 0.1/(S1+eps)**2
    alpha2 = 0.6/(S2+eps)**2
    alpha3 = 0.3/(S3+eps)**2
    alpha_tot = alpha1 + alpha2 + alpha3

    w1 = alpha1/alpha_tot
    w2 = alpha2/alpha_tot
    w3 = alpha3/alpha_tot

    return w1*u1x + w2*u2x + w3*u3x

@njit(cache=True)
def _godunov_vectorized(
    phi_plus    : NDArray,
    phi_minus   : NDArray,
    Vn          : NDArray
) -> NDArray:
    
    phi_x2_V_pos = np.maximum(np.maximum(phi_minus,0.0)**2,np.minimum(phi_plus,0.0)**2)
    phi_x2_V_neg = np.maximum(np.minimum(phi_minus,0.0)**2,np.maximum(phi_plus,0.0)**2)
    
    phi_x2 = np.where(Vn > 0.0, phi_x2_V_pos, np.where(Vn < 0.0, phi_x2_V_neg, 0.0))
    
    return phi_x2

@njit(cache=True)
def _godunov_Euler_step(
    phi_prev        : NDArray,
    Vn              : NDArray,
    f               : NDArray,
    dt              : float,
    dx              : float,
    dy              : float,
    n_direction     : int
) -> NDArray:
    
    phi_plus_minus_x = _WENO5_2d_step(phi_prev, dx, 0)
    phi_plus_minus_y = _WENO5_2d_step(phi_prev, dy, 1)
    
    phi_x2 = _godunov_vectorized(phi_plus_minus_x[0], phi_plus_minus_x[1], Vn)
    phi_y2 = _godunov_vectorized(phi_plus_minus_y[0], phi_plus_minus_y[1], Vn)
    
    grad = np.sqrt(phi_x2 + phi_y2)

    phi_new = phi_prev - n_direction * dt * Vn * (grad)

    if f is not None:
        phi_new += n_direction * dt * f

    constant_extrapolation(phi_new, phi_new[3:-3,3:-3].copy())
    
    return phi_new


class TVD_RK3_Godunov(TimeIntegrator):
    def __init__(self,Vn,f,dt,dx,dy,n_direction=1):
        """Implements the TVD RK3 scheme, computed as a Runge-Kutta weighted sum of Euler steps.

        Args:
            Vn (NDArray): Vn
            f (NDArray): f(x,y)
            dt (float): pseudo-timestep
            dx (float): grid spacing in x
            dy (float): grid spacing in x
        """
        super().__init__(dt=dt)
        self.Vn = Vn
        self.f = f
        self.dx = dx
        self.dy = dy
        self.n_direction = n_direction

        self.args = args = (self.Vn,self.f,self.dt,self.dx,self.dy,self.n_direction)
    
    def step(self,phi_prev):
        phi_np1 = self.godunov_Euler_step(phi_prev,*self.args)
        phi_np2 = self.godunov_Euler_step(phi_np1,*self.args)
        phi_np1h = 0.75*phi_prev + 0.25*phi_np2
        phi_np3h = self.godunov_Euler_step(phi_np1h,*self.args)
        phi_out = (phi_prev + 2*phi_np3h) / 3
        constant_extrapolation(phi_out, phi_out[3:-3,3:-3].copy())
        
        return phi_out


    @staticmethod
    def godunov_Euler_step(
        phi_prev        : NDArray,
        Vn              : NDArray,
        f               : NDArray,
        dt              : float,
        dx              : float,
        dy              : float,
        n_direction     : int
    ) -> NDArray:
        return _godunov_Euler_step(phi_prev,Vn,f,dt,dx,dy,n_direction)


