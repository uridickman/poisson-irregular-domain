from enum import Enum
from dataclasses import dataclass
from typing import Union
from numpy.typing import NDArray

class BCType(Enum):
    DIRICHLET = 1
    NEUMANN   = 2 # Not implemented
    ROBIN     = 3 # Not implemented
    PERIODIC  = 4 # Not implemented


class WallType(Enum):
    LEFT    = 1
    RIGHT   = 2
    TOP     = 3
    BOTTOM  = 4


@dataclass
class WallBC:
    bc_type : BCType
    val     : Union[float, NDArray]