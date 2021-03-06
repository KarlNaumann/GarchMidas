import numpy as np
from scipy.optimize import Bounds, SR1

"""
Basinhopping Settings
-------------------------------------------------------
This file details the settings that are used in the basinhopping optimization:
The custom steplength, restrictions dictionaries, and bounds
"""

__author__ = "Karl Naumann-Woleske"
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Karl Naumann-Woleske"


class StepAsym(object):
    def __init__(self, stepsize=1, max=3, rest=True):
        """ Class that generates custom step-sizes for the parameters in the
        regular GM models - larger steps are taken for the weights w1, w2 than
        for others

        Parameters
        ----------
        stepsize : int
        max : int
        rest : bool
        """
        self.stepsize = stepsize
        self.max = max
        self.rest = rest

    def __call__(self, x):
        # Order: a,b,m,theta,w1,w2
        s = self.stepsize
        # Starting value dependence mostly in w2 - thus vary most
        x[0] += np.random.uniform(-0.03 * s, 0.03 * s)   # mu
        x[1] += np.random.uniform(-0.03 * s, 0.03 * s)   # a1
        x[2] += np.random.uniform(-0.03 * s, 0.03 * s)   # b1
        x[3] += np.random.uniform(-0.03 * s, 0.03 * s)   # gamma
        x[4] += np.random.uniform(-0.07 * s, 0.07 * s)   # m
        x[5] += np.random.uniform(-0.07 * s, 0.07 * s)   # theta
        x[6] += np.random.uniform(-self.max * s, self.max * s)  # w1
        x[7] += np.random.uniform(-self.max * s, self.max * s)  # w2

        # Maintain feasibility
        if x[1] <= 1e-8:
            x[1] = 1e-8
        if x[2] >= 1.0:
            x[2] = 1 - 1e-8
        if self.rest:
            x[6] = 1

        return x


def bounds_Simple_asym():
    """ Generate bounds for the simply asymetric case.
    Order: a, b, m, theta, w1, w2

    Returns
    -------
    rest : Bounds
    unrest : Bounds
    """
    # Restricted case lb & ub
    lb_res = [-1e15, 0, 0, -1e1, -1e15, -1e15, 1, 1]
    ub_res = [1e15, 1, 1, 1e1, 1e15, 1e15, 1, 1e15]
    # unrestricted case lb & ub
    lb_free = [-1e15, 0, 0, -1e1, -1e15, -1e15, -1e15, -1e15]
    ub_free = [1e15, 1, 1, 1e1, 1e15, 1e15, 1e15, 1e15]
    # create bounds objects
    rest = Bounds(lb_res, ub_res, keep_feasible=True)
    unrest = Bounds(lb_free, ub_free, keep_feasible=True)
    return rest, unrest


def min_arg(args, rest, disp=False):
    """ Input for the local minimisation algorithm

    Parameters
    ----------
    args : tuple
    rest : Bounds
    disp : bool

    Returns
    -------
    kwargs: dict
    """
    return {
        'args': args,
        'jac': '2-point',
        'hess': SR1(),
        'bounds': rest,
        'options': {
            'disp': disp,
            'ftol': 1e-8,
            'gtol': 1e-8
        }
    }


def print_fun(x, f, accepted):
    """ Function to visualize the basinhopping steps

    Parameters
    ----------
    x : float
    f : float
    accepted : bool
    """
    print("F", f)
    print("Acc?", accepted)
    print("X:", x)
    print("- - - - - - - - - -")
