from abc import ABC,abstractmethod

import numpy as np
from numpy.typing import NDArray

from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import splu

class TimeManager(object):
    def __init__(
        self,
        trange: Tuple[float, float] = (0.0, 1.0),
        integrator : str = "RK3"
    ):
        super().__init__()

        self.tmin, self.tmax = trange
        
        self.integrator = integrator
        self.dt = self.integrator.dt
        self.T = np.arange(*trange,step=self.dt)
        self.t = self.tmin
        self.step = 0
        
        self.num_steps = int((self.tmax - self.tmin) / self.dt)
        
        
    def reset(self):
        self.t = 0

    def advance(self, u_prev, **kwargs):
        u_next = self.integrator.step(u_prev, **kwargs)
        self.t += self.dt
        self.step += 1
        
        return u_next

    def done(self):
        return self.t >= self.tmax


class TimeIntegrator(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def step(self,u_prev,**kwargs):
        pass


class ForwardEuler(TimeIntegrator):
    def __init__(self, dt, A):
        super().__init__()
        self.A = A
        self.dt = dt
    
    def step(self,u_prev):
        return u_prev + self.dt * (self.A @ u_prev)


class RK3(TimeIntegrator):
    def __init__(self, dt, A):
        super().__init__()
        self.A = A
        self.dt = dt

    def step(self, u_prev):
        u1 = u_prev + self.dt * (self.A @ u_prev)
        u2 = 0.75 * u_prev + 0.25 * (u1 + self.dt * (self.A @ u1))
        u3 = u2 + self.dt * (self.A @ u2)
        return (u_prev + 2 * u3) / 3


class BackwardEuler(TimeIntegrator):
    def __init__(self, dt, A):
        super().__init__(dt=dt)

        self.dt = dt
        
        n,m = A.shape[0],A.shape[1]
        I = eye(n,m)
        
        self.A = splu(I - self.dt * A)
        
    def step(u_prev):
        return self.A.solve(u_prev)


class CrankNicolson(TimeIntegrator):
    def __init__(self, dt, A):
        super().__init__()

        self.dt = dt

        n,m = A.shape[0],A.shape[1]
        I = eye(n,m)
        
        self.A = splu(I - self.dt / 2 * A)
        self.B = I + self.dt / 2 * A

    @staticmethod
    def step(self, u_prev):
        return self.A.solve(B @ u_prev)