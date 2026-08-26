import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

from typing import List, Tuple, Union, Callable
from numpy.typing import NDArray

# Custom
from ..utils.shapes import *
from ..levelset import advection as lsa
from ..utils.bcs import BCType,WallBC,WallType


class PoissonIrregularDomain_2d(object):
    """Class to construct a linear system to solve the Poisson equation on an irregular domain with
    Dirichlet Boundary conditions on the wall and boundary from Gibou et al. (2002).
    Can solve inside or outside, as jump conditions are not supported.
    
        ku-∇.(μ∇u) = f,        x ϵ Ω
        u = α,                 x ϵ Γ
        u = g,                 x ϵ ∂Ω
    
    Args:
        xrange (tuple[int,int], optional): domain in x. Defaults to (0,1).
        yrange (tuple[int,int], optional): domain in y. Defaults to (0,1).
        nx (int, optional): number of grid points in x. Defaults to 32.
        ny (int, optional): number of grid points in y. Defaults to 32.
        alpha (float, optional): value of u on the dirichlet boundary. Defaults to 0.0.
        phi (callable | NDArray, optional): level-set function. Defaults to np.zeros((32, 32)).
        mu (callable | NDArray, optional): diffusion coefficient. Defaults to np.zeros((32, 32)).
        k (callable | NDArray, optional): reaction term. Defaults to np.zeros((32, 32)).
        f (callable | NDArray, optional): forcing term. Defaults to np.zeros((32, 32)).
        g (callable | NDArray, optional): boundary condition on the wall. Defaults to np.zeros((32, 32)).
    
    """
    def __init__(
        self,
        xrange   : tuple[int,int] = (0,1),
        yrange   : tuple[int,int] = (0,1),
        nx       : int   = 32,
        ny       : int   = 32,
        alpha    : float = 0.0,
        phi      : callable | NDArray = np.zeros((32, 32)),
        mu       : callable | NDArray = np.zeros((32, 32)),
        k        : callable | NDArray = np.zeros((32, 32)),
        f        : callable | NDArray = np.zeros((32, 32)),
        g        : callable | NDArray = np.zeros((32, 32)),
        reinit   : bool = True
    ):
        super().__init__()

        self.xrange, self.yrange = xrange,yrange
        self.nx, self.ny = nx, ny

        self.dx = (xrange[1] - xrange[0]) / (nx - 1)
        self.dy = (yrange[1] - yrange[0]) / (ny - 1)

        self.X, self.Y = np.meshgrid(
            np.linspace(*xrange,num=nx),
            np.linspace(*yrange,num=ny)
        )
        
        self.alpha = alpha

        self.phi = self._evaluate_field(phi,self.X,self.Y)
        if reinit:
            self.phi[:,:] = self._reinitialize_phi(self.phi,self.dx,self.dy,max_iter=500,tol=1e-4)

        self.mu = self._evaluate_field(mu,self.X,self.Y)
        self.k = self._evaluate_field(k,self.X,self.Y)
        self.f = self._evaluate_field(f,self.X,self.Y)
        self.g = self._evaluate_field(g,self.X,self.Y)

        self.wall_boundary_conditions = {
            WallType.LEFT   : WallBC(
                                bc_type   = BCType.DIRICHLET,
                                val       = self.g[0,:]
                                ),
            WallType.RIGHT  : WallBC(
                                bc_type   = BCType.DIRICHLET,
                                val       = self.g[-1,:] 
                                ),
            WallType.TOP    : WallBC(
                                bc_type   = BCType.DIRICHLET,
                                val       = self.g[:,-1] 
                                ),
            WallType.BOTTOM : WallBC(
                                bc_type   = BCType.DIRICHLET,
                                val       = self.g[:,0]
                                )
        }
        

    def __repr__(self):
        return "PoissonIrregularDomain_2d"


    @staticmethod
    def _evaluate_field(field, x, y):
        if callable(field):
            return field(x, y)
        return field


    @staticmethod
    def _reinitialize_phi(phi,dx,dy,max_iter=500,tol=1e-4):
        print("Reinitializing level-set function...")
        dt = 0.3 * np.minimum(dx,dy)
        return lsa.reinitialize_phi(
            phi,dt,dx,dy,max_iter=max_iter,tol=tol
        )


    @staticmethod
    def _get_ghosts(phi, i, j):
        s = phi[i, j]

        try:
            ghosts = {
                "top":    s * phi[i, j+1] < 0,
                "bottom": s * phi[i, j-1] < 0,
                "left":   s * phi[i-1, j] < 0,
                "right":  s * phi[i+1, j] < 0,
            }
        except IndexError:
            ghosts = {
                "top":    False,
                "bottom": False,
                "left":   False,
                "right":  False,
            }

        return ghosts


    @staticmethod
    def _is_physical(phi,i,j,inside=True):
        if inside:
            return phi[i,j] < 0
        else:
            return phi[i,j] > 0


    def compute_lhs_rhs(self,solve_inside=False):
        u_interface = self.alpha
        phi = self.phi
        mu = self.mu
        k = self.k
        f = self.f

        interior_nodes = [
            (i, j)
            for i in range(1, self.nx - 1)
            for j in range(1, self.ny - 1)
            if self._is_physical(self.phi, i, j, inside=solve_inside)
        ]
        
        left_nodes = [
            (0, j)
            for j in range(self.ny )
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

        A   = lil_matrix((N, N), dtype=float)
        rhs = np.zeros(N)

        dx = self.dx
        dy = self.dy
        dx2 = dx**2
        dy2 = dy**2

        for i,j in right_nodes:
            row = node_ids[i, j]
            A[row, row] = 1.0
            rhs[row] += self.wall_boundary_conditions[WallType.RIGHT].val[j]
        for i,j in left_nodes:
            row = node_ids[i, j]
            A[row, row] = 1.0
            rhs[row] += self.wall_boundary_conditions[WallType.LEFT].val[j]
        for i,j in top_nodes:
            row = node_ids[i, j]
            A[row, row] = 1.0
            rhs[row] += self.wall_boundary_conditions[WallType.TOP].val[i]
        for i,j in bottom_nodes:
            row = node_ids[i, j]
            A[row, row] = 1.0
            rhs[row] += self.wall_boundary_conditions[WallType.BOTTOM].val[i]
        
        for i,j in interior_nodes:
            row = node_ids[i, j]
            rhs[row] = f[i, j]
            A[row, row] = -k[i, j]
            
            ghosts = self._get_ghosts(phi,i,j)
            
            right_ghost = ghosts["right"]
            left_ghost = ghosts["left"]
            top_ghost = ghosts["top"]
            bottom_ghost = ghosts["bottom"]
            
            right_wall = i+1 == self.nx-1
            left_wall = i-1 == 0
            top_wall = j+1 == self.ny-1
            bottom_wall = j-1 == 0
            
            mu_iph = (mu[i,j] + mu[i+1,j]) / 2
            mu_imh = (mu[i,j] + mu[i-1,j]) / 2
            mu_jph = (mu[i,j] + mu[i,j+1]) / 2
            mu_jmh = (mu[i,j] + mu[i,j-1]) / 2

            if right_ghost:
                theta = np.abs(phi[i, j]) / (np.abs(phi[i, j]) + np.abs(phi[i+1, j]))
                A[row, row] -= mu_iph / theta / dx2
                rhs[row] += mu_iph * u_interface / theta / dx2
            elif right_wall:
                g = self.wall_boundary_conditions[WallType.RIGHT].val[j]
                A[row, row] -= mu_iph / dx2
                rhs[row] -= mu_iph * g / dx2
            else:
                right_row = node_ids[i+1, j]
                A[row, row] -= mu_iph / dx2
                A[row, right_row] += mu_iph / dx2
                
            
            if left_ghost:
                theta = np.abs(phi[i, j]) / (np.abs(phi[i, j]) + np.abs(phi[i-1, j]))
                A[row, row] -= mu_imh / theta / dx2
                rhs[row] += mu_imh * u_interface / theta / dx2
            elif left_wall:
                g = self.wall_boundary_conditions[WallType.LEFT].val[j]
                A[row, row] -= mu_imh / dx2
                rhs[row] -= mu_imh * g / dx2
            else:
                left_row = node_ids[i-1, j]
                A[row, row] -= mu_imh / dx2
                A[row, left_row] += mu_imh / dx2

            if top_ghost:
                theta = np.abs(phi[i, j]) / (np.abs(phi[i, j]) + np.abs(phi[i, j+1]))
                A[row, row] -= mu_jph / theta / dy2
                rhs[row] += mu_jph * u_interface / theta / dy2
            elif top_wall:
                g = self.wall_boundary_conditions[WallType.TOP].val[i]
                A[row, row] -= mu_jph / dy2
                rhs[row] -= mu_jph * g / dy2
            else:
                top_row = node_ids[i, j+1]
                A[row, row] -= mu_jph / dy2
                A[row, top_row] += mu_jph / dy2

            if bottom_ghost:
                theta = np.abs(phi[i, j]) / (np.abs(phi[i, j]) + np.abs(phi[i, j-1]))
                A[row, row] -= mu_jmh / theta / dy2
                rhs[row] += mu_jmh * u_interface / theta / dy2
            elif bottom_wall:
                g = self.wall_boundary_conditions[WallType.BOTTOM].val[i]
                A[row, row] -= mu_jmh / dy2
                rhs[row] -= mu_jmh * g / dy2
            else:
                bottom_row = node_ids[i, j-1]
                A[row, row] -= mu_jmh / dy2
                A[row, bottom_row] += mu_jmh / dy2
        
        lhs = A.tocsc()
        return lhs,rhs,all_nodes


    def solve(self,solve_inside=False):
        """Constructs and solves the linear system to solve the Poisson equation.

        Args:
            solve_inside (bool, optional): if True, solves inside the boundary. Otherwise, solve outside. Defaults to False.

        Returns:
            NDArray: the returned solution
        """
        # Construct the linear system
        lhs,rhs,nodes = self.compute_lhs_rhs(solve_inside=solve_inside)
        lu = splu(lhs)
        u = lu.solve(rhs)

        # Populate the full regular grid based on the nodes
        # used to solve the system.
        # The other nodes are set to the value on the interface
        sol = np.full((self.nx, self.ny), self.alpha, dtype=float)
        for row, (i, j) in enumerate(nodes):
            sol[i, j] = u[row]
        
        return sol