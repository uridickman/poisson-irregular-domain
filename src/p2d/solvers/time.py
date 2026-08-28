from abc import ABC, abstractmethod
from typing import Tuple, Union, Callable

import numpy as np
from numpy.typing import NDArray

from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import splu


class TimeManager:
    def __init__(
        self,
        dt: float,
        trange: Tuple[float, float] = (0.0, 1.0)
    ):
        self.tmin, self.tmax = trange
        self.dt = dt
        self.t = self.tmin
        self.step_idx = 0
        self.num_steps = int(np.round((self.tmax - self.tmin) / self.dt))
        
        self.integrators =  {}

    def add_integrator(self, name: str, integrator: TimeIntegrator):
        self.integrators[name] = integrator

    def step_field(self, name: str, u_prev, f, **kwargs):
        return self.integrators[name].step(u_prev, f, **kwargs)

    def advance_time(self):
        self.t += self.dt
        self.step_idx += 1

    def done(self):
        return self.t >= self.tmax - 1e-12 * self.dt
        
    def reset(self):
        self.t = self.tmin
        self.step = 0


class TimeIntegrator(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def step(self, u_prev, f, **kwargs):
        pass


class ForwardEuler(TimeIntegrator):
    def __init__(self, dt, A):
        super().__init__()
        self.A = A.tocsc()
        self.dt = dt
    
    def step(self, u_prev, f, **kwargs):
        return u_prev + self.dt * (self.A @ u_prev + f)


class RK3(TimeIntegrator):
    def __init__(self, dt, A):
        super().__init__()
        self.M = A.tocsc()
        self.dt = dt

    def step(self, u_prev, f, **kwargs):
        u1 = u_prev + self.dt * (self.M @ u_prev + f)
        u2 = 0.75 * u_prev + 0.25 * (u1 + self.dt * (self.M @ u1 + f))
        u3 = u2 + self.dt * (self.M @ u2 + f)
        return (u_prev + 2 * u3) / 3.0


class BackwardEuler(TimeIntegrator):
    def __init__(self, dt, A):
        super().__init__()
        self.dt = dt
        
        n, m = A.shape[0], A.shape[1]
        I = eye(n, m, format="csc")
        
        self.M = splu((I - self.dt * A).tocsc())
        
    def step(self, u_prev, f, **kwargs):
        return self.M.solve(u_prev + self.dt * f)


class CrankNicolson(TimeIntegrator):
    def __init__(self, dt, A):
        super().__init__()

        self.dt = dt

        n, m = A.shape[0], A.shape[1]
        I = eye(n, m, format="csc")

        self.M = splu((I - self.dt / 2.0 * A).tocsc())
        self.N = (I + self.dt / 2.0 * A).tocsc()

    def step(self, u_prev, f, **kwargs):
        return self.M.solve(self.N @ u_prev + self.dt * f)