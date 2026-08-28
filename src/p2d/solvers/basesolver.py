from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Callable, Optional, Union
import numpy as np
from numpy.typing import NDArray
from ..levelset import levelset as ls
from ..utils.bcs import *

class BaseSolver(object):
    def __init__(
        self,
        xrange   : tuple[int,int] = (0,1),
        yrange   : tuple[int,int] = (0,1),
        nx       : int   = 32,
        ny       : int   = 32,
        alpha    : callable | float = 0.0,
        phi      : callable | NDArray = np.zeros((32, 32)),
        f        : callable | NDArray = np.zeros((32, 32)),
        g        : callable | NDArray = np.zeros((32, 32)),
        mu       : callable | NDArray = np.zeros((32, 32)),
        k        : callable | NDArray = np.zeros((32, 32))
        
    ):
        self.xrange, self.yrange = xrange,yrange
        self.nx, self.ny = nx, ny

        self.dx = (xrange[1] - xrange[0]) / (nx - 1)
        self.dy = (yrange[1] - yrange[0]) / (ny - 1)

        self.X, self.Y = np.meshgrid(
            np.linspace(*xrange,num=nx),
            np.linspace(*yrange,num=ny),
            indexing="ij"
        )
        
        if callable(alpha):
            self.alpha_fn = alpha
        else:
            self.alpha_fn = lambda x, y, _alpha=alpha: np.full_like(np.asarray(x, dtype=float), _alpha)

        self.phi = self._evaluate_field(phi,self.X,self.Y)
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

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def compute_lhs_rhs(self, solve_inside=False):
        pass

    @abstractmethod
    def solve(self, solve_where="inside", **kwargs):
        pass


    @staticmethod
    def _evaluate_field(field, x, y):
        if callable(field):
            res = field(x, y)
            if np.isscalar(res):
                return np.full_like(x, res, dtype=float)
            return np.asarray(res, dtype=float)
        if isinstance(field, (int, float)) or np.isscalar(field):
            return np.full_like(x, field, dtype=float)
        return np.asarray(field, dtype=float)


    @staticmethod
    def _reinitialize_phi(phi, dx, dy, max_iter=500, tol=1e-4):
        print("Reinitializing level-set function...")
        dt = 0.3 * np.minimum(dx, dy)
        return ls.reinitialize_phi(
            phi, dt, dx, dy, max_iter=max_iter, tol=tol
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
    def _is_physical(phi, i, j, inside=True):
        if inside:
            return phi[i, j] < 0
        else:
            return phi[i, j] > 0

    @staticmethod
    def vec_to_matrix(nodes,matrix,vec):
        for row, (i, j) in enumerate(nodes):
            matrix[..., i, j] = vec[row]

    @staticmethod
    def matrix_to_vec(nodes,matrix,vec):
        for row, (i, j) in enumerate(nodes):
            vec[row] =  matrix[..., i, j]