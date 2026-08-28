import numpy as np
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import splu

from typing import List, Tuple, Union, Callable
from numpy.typing import NDArray

# Custom
from ..utils.shapes import *
from ..levelset import levelset as ls
from ..utils.bcs import BCType, WallBC, WallType
from .time import TimeManager, TimeIntegrator, CrankNicolson
from .basesolver import BaseSolver


class HeatIrregularDomain_2d(BaseSolver):
    """Class to construct a linear system to solve the heat equation on an irregular domain with
    Dirichlet Boundary conditions on the wall and boundary from Gibou et al. (2002).
    Can solve inside or outside, as jump conditions are not supported.
    
        ∂u/∂t - ∇.(μ∇u) = f,   x ϵ Ω
        u = α,                 x ϵ Γ
        u = g,                 x ϵ ∂Ω
        u(x, 0) = u0,          x ϵ Ω

    Args:
        xrange (tuple[int,int] | tuple[float,float], optional): domain in x. Defaults to (0,1).
        yrange (tuple[int,int] | tuple[float,float], optional): domain in y. Defaults to (0,1).
        nx (int, optional): number of grid points in x. Defaults to 32.
        ny (int, optional): number of grid points in y. Defaults to 32.
        alpha (callable | float, optional): value of u on the Dirichlet boundary. Defaults to 0.0.
        phi (callable | NDArray | float, optional): level-set function. Defaults to np.zeros((32, 32)).
        f (callable | NDArray | float, optional): forcing term. Defaults to 0.0.
        g (callable | NDArray | float, optional): boundary condition on the wall. Defaults to 0.0.
        mu (callable | NDArray | float, optional): diffusion coefficient. Defaults to 1.0.
    """
    def __init__(
        self,
        xrange   : tuple[float, float] | tuple[int, int] = (0, 1),
        yrange   : tuple[float, float] | tuple[int, int] = (0, 1),
        nx       : int   = 32,
        ny       : int   = 32,
        alpha    : callable | float = 0.0,
        phi      : callable | NDArray | float = 0.0,
        mu       : callable | NDArray | float = 1.0,
        f        : callable | NDArray | float = 0.0,
        g        : callable | NDArray | float = 0.0,
    ):
        super().__init__(xrange,yrange,nx,ny,alpha,phi,f,g,mu)


    def __repr__(self):
        return "HeatIrregularDomain_2d"


    def apply_boundary_conditions(self, u: NDArray, nodes=None) -> NDArray:
        if nodes is None:
            if not hasattr(self, "nodes") or self.nodes is None:
                return u
            nodes = self.nodes

        node_ids = {node: idx for idx, node in enumerate(nodes)}

        for j in range(self.ny):
            if (self.nx - 1, j) in node_ids:
                row = node_ids[(self.nx - 1, j)]
                u[row] = self.wall_boundary_conditions[WallType.RIGHT].val[j]
            if (0, j) in node_ids:
                row = node_ids[(0, j)]
                u[row] = self.wall_boundary_conditions[WallType.LEFT].val[j]

        for i in range(self.nx):
            if (i, self.ny - 1) in node_ids:
                row = node_ids[(i, self.ny - 1)]
                u[row] = self.wall_boundary_conditions[WallType.TOP].val[i]
            if (i, 0) in node_ids:
                row = node_ids[(i, 0)]
                u[row] = self.wall_boundary_conditions[WallType.BOTTOM].val[i]

        return u


    def compute_lhs_rhs(self, solve_inside=False):
        """Constructs the spatial discretization matrix A and boundary/forcing vector rhs
        for the system du/dt = A*u + rhs.
        """
        phi = self.phi
        mu = self.mu
        f = self.f

        interior_nodes = [
            (i, j)
            for i in range(1, self.nx - 1)
            for j in range(1, self.ny - 1)
            if self._is_physical(self.phi, i, j, inside=solve_inside)
        ]
        
        left_nodes = [
            (0, j)
            for j in range(self.ny)
            if self._is_physical(self.phi, 0, j, inside=solve_inside)
        ]

        right_nodes = [
            (self.nx - 1, j)
            for j in range(self.ny)
            if self._is_physical(self.phi, self.nx - 1, j, inside=solve_inside)
        ]

        bottom_nodes = [
            (i, 0)
            for i in range(1, self.nx - 1)
            if self._is_physical(self.phi, i, 0, inside=solve_inside)
        ]

        top_nodes = [
            (i, self.ny - 1)
            for i in range(1, self.nx - 1)
            if self._is_physical(self.phi, i, self.ny - 1, inside=solve_inside)
        ]
        
        all_nodes = interior_nodes + right_nodes + left_nodes + top_nodes + bottom_nodes
        
        node_ids = -np.ones((self.nx, self.ny), dtype=int)
        for row, (i, j) in enumerate(all_nodes):
            node_ids[i, j] = row

        N = len(all_nodes)

        A = lil_matrix((N, N), dtype=float)
        rhs = np.zeros(N)

        X = self.X
        Y = self.Y
        dx = self.dx
        dy = self.dy
        dx2 = dx**2
        dy2 = dy**2

        # For wall nodes: held fixed at Dirichlet values
        for i, j in (right_nodes + left_nodes + top_nodes + bottom_nodes):
            row = node_ids[i, j]
            A[row, row] = 0.0
            rhs[row] = 0.0
        
        min_theta_x = np.inf
        min_theta_y = np.inf

        for i, j in interior_nodes:
            phi_ij = phi[i, j]
                
            row = node_ids[i, j]
            rhs[row] = f[i, j]
            
            ghosts = self._get_ghosts(phi, i, j)
            
            right_ghost = ghosts["right"]
            left_ghost = ghosts["left"]
            top_ghost = ghosts["top"]
            bottom_ghost = ghosts["bottom"]
            
            right_wall = (i + 1 == self.nx - 1)
            left_wall = (i - 1 == 0)
            top_wall = (j + 1 == self.ny - 1)
            bottom_wall = (j - 1 == 0)
            
            mu_iph = (mu[i, j] + mu[i + 1, j]) / 2.0
            mu_imh = (mu[i, j] + mu[i - 1, j]) / 2.0
            mu_jph = (mu[i, j] + mu[i, j + 1]) / 2.0
            mu_jmh = (mu[i, j] + mu[i, j - 1]) / 2.0

            # --- Right neighbor ---
            if right_ghost:
                theta = np.abs(phi_ij) / (np.abs(phi_ij) + np.abs(phi[i + 1, j]))
                u_interface = self.alpha_fn(X[i, j] + theta * dx, Y[i, j])
                A[row, row] -= mu_iph / theta / dx2
                rhs[row] += mu_iph * u_interface / theta / dx2

                if theta < min_theta_x:
                    min_theta_x = theta

            elif right_wall:
                g_val = self.wall_boundary_conditions[WallType.RIGHT].val[j]
                A[row, row] -= mu_iph / dx2
                rhs[row] += mu_iph * g_val / dx2
            else:
                right_row = node_ids[i + 1, j]
                A[row, row] -= mu_iph / dx2
                A[row, right_row] += mu_iph / dx2

            # --- Left neighbor ---
            if left_ghost:
                theta = np.abs(phi_ij) / (np.abs(phi_ij) + np.abs(phi[i - 1, j]))
                u_interface = self.alpha_fn(X[i, j] - theta * dx, Y[i, j])
                A[row, row] -= mu_imh / theta / dx2
                rhs[row] += mu_imh * u_interface / theta / dx2

                if theta < min_theta_x:
                    min_theta_x = theta

            elif left_wall:
                g_val = self.wall_boundary_conditions[WallType.LEFT].val[j]
                A[row, row] -= mu_imh / dx2
                rhs[row] += mu_imh * g_val / dx2
            else:
                left_row = node_ids[i - 1, j]
                A[row, row] -= mu_imh / dx2
                A[row, left_row] += mu_imh / dx2

            # --- Top neighbor ---
            if top_ghost:
                theta = np.abs(phi_ij) / (np.abs(phi_ij) + np.abs(phi[i, j + 1]))
                u_interface = self.alpha_fn(X[i, j], Y[i, j] + theta * dy)
                A[row, row] -= mu_jph / theta / dy2
                rhs[row] += mu_jph * u_interface / theta / dy2

                if theta < min_theta_y:
                    min_theta_y = theta

            elif top_wall:
                g_val = self.wall_boundary_conditions[WallType.TOP].val[i]
                A[row, row] -= mu_jph / dy2
                rhs[row] += mu_jph * g_val / dy2
            else:
                top_row = node_ids[i, j + 1]
                A[row, row] -= mu_jph / dy2
                A[row, top_row] += mu_jph / dy2

            # --- Bottom neighbor ---
            if bottom_ghost:
                theta = np.abs(phi_ij) / (np.abs(phi_ij) + np.abs(phi[i, j - 1]))
                u_interface = self.alpha_fn(X[i, j], Y[i, j] - theta * dy)
                A[row, row] -= mu_jmh / theta / dy2
                rhs[row] += mu_jmh * u_interface / theta / dy2

                if theta < min_theta_y:
                    min_theta_y = theta

            elif bottom_wall:
                g_val = self.wall_boundary_conditions[WallType.BOTTOM].val[i]
                A[row, row] -= mu_jmh / dy2
                rhs[row] += mu_jmh * g_val / dy2
            else:
                bottom_row = node_ids[i, j - 1]
                A[row, row] -= mu_jmh / dy2
                A[row, bottom_row] += mu_jmh / dy2
        
        return A.tocsc(), rhs, all_nodes, min_theta_x, min_theta_y


    def initialize_solver(
        self,
        u0,
        dt=0.01,
        trange=(0.0, 1.0),
        solve_where="inside"
    ):
        
        self.solve_inside = solve_where == "inside" or solve_where == "both"
        self.solve_outside = solve_where == "outside" or solve_where == "both"

        u0_field = self._evaluate_field(u0, self.X, self.Y)

        self.tm = TimeManager(dt=dt,trange=trange)
        self.sol = np.zeros((self.tm.num_steps + 1, self.nx, self.ny), dtype=float)

        if self.solve_inside:

            self.A_in, self.rhs_in, self.nodes_in, min_tx_in, min_ty_in = self.compute_lhs_rhs(solve_inside=True)
            self.tm.add_integrator("inside",CrankNicolson(dt, self.A_in))

            self.u_in = np.zeros(len(self.nodes_in), dtype=float)
            self.matrix_to_vec(self.nodes_in,u0_field,self.u_in)
            self.u_in = self.apply_boundary_conditions(self.u_in, nodes=self.nodes_in)

            self.vec_to_matrix(self.nodes_in,self.sol[0,:,:],self.u_in)

        if self.solve_outside:
            
            self.A_out, self.rhs_out, self.nodes_out, min_tx_out, min_ty_out = self.compute_lhs_rhs(solve_inside=False)
            self.tm.add_integrator("outside",CrankNicolson(dt, self.A_out))
            
            self.u_out = np.zeros(len(self.nodes_out), dtype=float)
            self.matrix_to_vec(self.nodes_out,u0_field,self.u_out)
            self.u_out = self.apply_boundary_conditions(self.u_out, nodes=self.nodes_out)

            self.vec_to_matrix(self.nodes_out,self.sol[0,:,:],self.u_out)

        return self.u_in, self.u_out


    def solve(self, u0, dt, trange=(0.0, 1.0), solve_where="inside"):
        self.initialize_solver(u0=u0, dt=dt, trange=trange, solve_where=solve_where)

        while not self.tm.done():
            self.tm.advance_time()
            step_idx = self.tm.step_idx

            if self.solve_inside:
                self.u_in = self.tm.step_field("inside",self.u_in, self.rhs_in)
                self.u_in = self.apply_boundary_conditions(self.u_in, nodes=self.nodes_in)
                self.vec_to_matrix(self.nodes_in,self.sol[step_idx,:,:],self.u_in)
            if self.solve_outside:
                self.u_out = self.tm.step_field("outside",self.u_out, self.rhs_out)
                self.u_out = self.apply_boundary_conditions(self.u_out, nodes=self.nodes_out)
                self.vec_to_matrix(self.nodes_out,self.sol[step_idx,:,:],self.u_out)

        return self.sol