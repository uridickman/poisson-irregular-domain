from numba import njit
from numba.experimental import jitclass
import numpy as np


@njit(cache=True)
def circle(x,y,x0=0.0,y0=0.0,r=1.0):
    return np.sqrt((x - x0)**2 + ((y - y0))**2) - r


@njit(cache=True)
def rectangle(x,y,x0,y0,l,w):
    dx = np.abs(x - x0) - l/2
    dy = np.abs(y - y0) - w/2

    outside = np.sqrt(np.maximum(dx, 0)**2 +
                    np.maximum(dy, 0)**2)

    inside = np.minimum(np.maximum(dx, dy), 0)

    phi = outside + inside
    return phi


@njit(cache=True)
def flower(x, y, r0=1.0, amplitude=0.25, petals=8):

    theta = np.arctan2(y, x)
    r = np.sqrt(x*x + y*y)

    rb = r0 * (1.0 + amplitude*np.cos(petals*theta))

    return r - rb


@njit(cache=True)
def star(x, y, r0=1.0, amplitude=0.35, n=5):

    theta = np.arctan2(y, x)
    r = np.sqrt(x*x + y*y)

    rb = r0 * (1.0 + amplitude*np.cos(n*theta))

    return r - rb


@njit(cache=True)
def ellipse(x, y, a=1.0, b=0.6):
    px = np.abs(x)
    py = np.abs(y)

    k1 = np.sqrt((px/a)**2 + (py/b)**2)
    k2 = np.sqrt((px/(a*a))**2 + (py/(b*b))**2)

    return (k1 - 1.0) * k1 / k2

