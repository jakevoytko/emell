"""Pure math functions used by the ML routines."""

from emell.computation.l1_loss import l1_loss
from emell.computation.l2_loss import l2_loss
from emell.computation.quadratic_cost import quadratic_cost
from emell.computation.relu import relu

__all__ = [
    "l1_loss",
    "l2_loss",
    "relu",
    "quadratic_cost",
]
