import numpy as np
from scipy.sparse.linalg import spsolve

from typing import List, Tuple, Union, Callable
from numpy.typing import NDArray

# Custom
from ..utils.shapes import *
from ..levelset import levelset as ls
from ..utils.bcs import BCType,WallBC,WallType
from .basesolver import BaseSolver


class PoissonIrregularDomain_2d(BaseSolver):
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
        alpha (callable | float, optional): value of u on the Dirichlet boundary. Defaults to 0.0.
        phi (callable | NDArray, optional): level-set function. Defaults to np.zeros((32, 32)).
        f (callable | NDArray, optional): forcing term. Defaults to np.zeros((32, 32)).
        g (callable | NDArray, optional): boundary condition on the wall. Defaults to np.zeros((32, 32)).
        mu (callable | NDArray, optional): diffusion coefficient. Defaults to np.zeros((32, 32)).
        k (callable | NDArray, optional): reaction term. Defaults to np.zeros((32, 32)).
    
    """
    def __init__(
        self,
        xrange   : tuple[float, float] | tuple[int, int] = (0, 1),
        yrange   : tuple[float, float] | tuple[int, int] = (0, 1),
        nx       : int   = 32,
        ny       : int   = 32,
        alpha    : callable | float = 0.0,
        phi      : callable | NDArray | float = 0.0,
        f        : callable | NDArray | float = 0.0,
        g        : callable | NDArray | float = 0.0,
        mu       : callable | NDArray | float = 1.0,
        k        : callable | NDArray | float = 1.0
    ):
        super().__init__(xrange,yrange,nx,ny,alpha,phi,f,g,mu,k)


    def __repr__(self):
        return "PoissonIrregularDomain_2d"


    def solve(self,solve_where="inside"):
        if solve_where == "both":
            sol = self.solve_one_side(solve_inside=True)
            sol_outside = self.solve_one_side(solve_inside=False)
            
            sol[self.phi > 0] = sol_outside[self.phi > 0]
            return sol
        else:
            return self.solve_one_side(solve_inside=(solve_where=="inside"))


    def solve_one_side(self,solve_inside=False):
        """Constructs and solves the linear system to solve the Poisson equation.

        Args:
            solve_inside (bool, optional): if True, solves inside the boundary. Otherwise, solve outside. Defaults to False.

        Returns:
            NDArray: the returned solution
        """
        # Construct the linear system
        A,rhs,nodes = self.compute_lhs_rhs(solve_inside=solve_inside)
        u = spsolve(A.tocsc(), rhs)

        # Populate the full regular grid based on the nodes
        # used to solve the system.
        # The other nodes are set to the value on the interface.nx
        sol = np.full((self.nx, self.ny), np.nan)
        self.vec_to_matrix(nodes,sol,u)
        
        return sol