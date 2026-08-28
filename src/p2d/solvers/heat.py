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

            self.A_in, self.rhs_in, self.nodes_in = self.compute_lhs_rhs(solve_inside=True)
            self.tm.add_integrator("inside",CrankNicolson(dt, self.A_in.tocsc()))

            self.u_in = np.zeros(len(self.nodes_in), dtype=float)
            self.matrix_to_vec(self.nodes_in,u0_field,self.u_in)
            self.u_in = self.apply_boundary_conditions(self.u_in, nodes=self.nodes_in)

            self.vec_to_matrix(self.nodes_in,self.sol[0,:,:],self.u_in)

        if self.solve_outside:
            
            self.A_out, self.rhs_out, self.nodes_out = self.compute_lhs_rhs(solve_inside=False)
            self.tm.add_integrator("outside",CrankNicolson(dt, self.A_out.tocsc()))
            
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