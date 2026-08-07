import numpy as np
from numba import njit

def reinitialize_phi(phi0, dt, dx, dy, max_iter=50, tol=1e-6):
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

    for n_iter in range(max_iter):

        phi_old = phi.copy()

        phi = TVD_RK3_step(phi, S_ext, dt, dx, dy, n_direction=1)

        phi_inner = phi[3:-3, 3:-3]
        phi_old_inner = phi_old[3:-3, 3:-3]

        err = np.max(np.abs(phi_inner[band] - phi_old_inner[band]))

        if err < tol:
            print(f"Reinitialization converged in {n_iter+1} iterations.")
            return phi_inner

    print(f"Reinitialization failed to converge in {max_iter} iterations.")
    print(f"Final error = {err:e}")

    return phi[3:-3, 3:-3]


@njit(cache=True)
def constant_extrapolation(phi_extrap, phi):
    phi_extrap[3:-3, 3:-3] = phi

    for i in range(3):
        phi_extrap[i, 3:-3]      = phi[0, :]
        phi_extrap[-(i+1), 3:-3] = phi[-1, :]
        phi_extrap[3:-3, i]      = phi[:, 0]
        phi_extrap[3:-3, -(i+1)] = phi[:, -1]


@njit(cache=True)
def sign(phi, eps=1e-3):
    return phi / np.sqrt(phi*phi + eps*eps)


@njit(cache=True)
def WENO5_2d_step(phi:np.ndarray,dx:np.float64,direction:int):
    Nx,Ny = phi.shape
    partial_phi_plus = np.zeros_like(phi)
    partial_phi_minus = np.zeros_like(phi)

    if direction == 0:
        for j in range(Ny):
            phi_slice = phi[:,j]
            partial_phi_plus[3:-3,j] = WENO_partial_phi_vectorized(phi_slice,dx,True)
            partial_phi_minus[3:-3,j] = WENO_partial_phi_vectorized(phi_slice,dx,False)
    elif direction == 1:
        for i in range(Nx):
            phi_slice = phi[i,:]
            partial_phi_plus[i,3:-3] = WENO_partial_phi_vectorized(phi_slice,dx,True)
            partial_phi_minus[i,3:-3] = WENO_partial_phi_vectorized(phi_slice,dx,False)
    else:
        raise ValueError(f"direction = {direction} is not defined.")

    return partial_phi_plus,partial_phi_minus


@njit(cache=True)
def WENO_partial_phi_vectorized(phi,dx,is_plus):
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
def godunov_reinit_vectorized(phi, S, direction, dx):
    phi_plus_minus = WENO5_2d_step(phi, dx, direction)
    phi_plus = phi_plus_minus[0]
    phi_minus = phi_plus_minus[1]
    
    phi_x2_S_pos = np.maximum(np.maximum(phi_minus,0.0)**2,np.minimum(phi_plus,0.0)**2)
    phi_x2_S_neg = np.maximum(np.minimum(phi_minus,0.0)**2,np.maximum(phi_plus,0.0)**2)
    
    phi_x2 = np.where(S > 0, phi_x2_S_pos, phi_x2_S_neg)
    
    return phi_x2

@njit(cache=True)
def Euler_godunov_step(phi_prev,S,dt,dx,dy,n_direction):
    phi_x2 = godunov_reinit_vectorized(phi_prev,S,0,dx)
    phi_y2 = godunov_reinit_vectorized(phi_prev,S,1,dy)
    grad = np.sqrt(phi_x2 + phi_y2)
    phi_new = phi_prev - n_direction*dt*S*(grad-1.0)
    constant_extrapolation(phi_new, phi_new[3:-3,3:-3].copy())
    return phi_new
        
        
def TVD_RK3_step(phi_prev,S,dt,dx,dy,n_direction):
    phi_np1 = Euler_godunov_step(phi_prev,S,dt,dx,dy,n_direction)
    phi_np2 = Euler_godunov_step(phi_np1,S,dt,dx,dy,n_direction)
    phi_np1h = 0.75*phi_prev + 0.25*phi_np2
    phi_np3h = Euler_godunov_step(phi_np1h,S,dt,dx,dy,n_direction)
    phi_out = (phi_prev + 2*phi_np3h) / 3
    constant_extrapolation(phi_out, phi_out[3:-3,3:-3].copy())
    
    return phi_out