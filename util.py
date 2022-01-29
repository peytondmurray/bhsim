"""Miscellaneous utilities for plotting black hole images."""
from typing import Any, Callable, Iterable, Tuple

import numpy as np
import numpy.typing as npt


def fast_root(
    f: Callable,
    x: npt.NDArray[float],
    y: npt.NDArray[float],
    args: Iterable[Any],
    tol: float = 1e-6,
    max_steps: int = 10,
) -> npt.NDArray[float]:
    """Find the values x0 for each y-value which f(x0_i, y_i, *args) = 0.

    Parameters
    ----------
    f : Callable
        Objective function for which roots are to be found.
    x : npt.NDArray[float]
        An array of X-values {x_i}; at least one sign-flip of f must exist for a pair of values
        {x_m, x_m+1} if a root is to be found.
    y : npt.NDArray[float]
        Y-values {y_i}; a root of the equation x0_i is calculated for each y_i.
    args : Iterable[Any]
        Any additional arguments are passed to f.
    tol : float
        Controls how close to the true zero of f each root is by the following relation:
            f(x0_i, y_i, *args) - f(x0_i_true, y_i, *args) < tol
        unless the number of max_steps is reached.
    max_steps : int
        Maximum number of iterations to run in refining the guesses of the roots.

    Returns
    -------
    npt.NDArray[float]
        Array of roots of f; the shape of the array is the same as y, as there is a one root for
        each value of y.
    """
    xmin, xmax = find_brackets(f, x, y, args, tol, max_steps)

    sign_min = np.sign(f(xmin, y, *args))
    sign_max = np.sign(f(xmax, y, *args))

    for i in range(max_steps):
        xmid = 0.5 * (xmin + xmax)
        f_mid = f(xmid, y, *args)
        sign_mid = np.sign(f_mid)

        if np.nanmax(f_mid) < tol:
            return xmid
        else:
            xmin = np.where(sign_min == sign_mid, xmid, xmin)
            xmax = np.where(sign_max == sign_mid, xmid, xmax)

    return xmid


def find_brackets(
    f: Callable,
    x: npt.NDArray[float],
    y: npt.NDArray[float],
    args: Iterable[Any],
    tol: float = 1e-6,
    max_steps: int = 10,
) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Find a pair of vectors {x_i_min} and {x_i_max}, between which the roots of f lie.

    Parameters
    ----------
    f : Callable
        Objective function whose roots are the periastron distance for a given value of alpha.
    x : npt.NDArray[float]
        An array of X-values {x_i}; at least one sign-flip of f must exist for a pair of values
        {x_m, x_m+1} if a root is to be found.
    y : npt.NDArray[float]
        Y-values {y_i}; a root of the equation x0_i is calculated for each y_i.
    args : Iterable[Any]
        Any additional arguments are passed to f; these usually include r, theta_0, N, and M.
    tol : float
        Controls how close to the true zero of f each root is by the following relation:
            f(x0_i, y_i, *args) - f(x0_i_true, y_i, *args) < tol
        unless the number of max_steps is reached.
    max_steps : int
        Maximum number of iterations to run in refining the guesses of the roots.

    Returns
    -------
    Tuple[npt.NDArray[float], npt.NDArray[float]]
        Minimum and maximum x value vectors, between which the roots of f lie.
    """
    # Calculate the objective function; find the sign flips where the function is not nan
    xx, yy = np.meshgrid(x, y)
    objective = f(xx, yy, *args)
    flip_mask = (np.diff(np.sign(objective), axis=1) != 0) & (
        ~np.isnan(objective[:, :-1])
    )
    i_at_sign_flip = np.where(
        np.any(flip_mask, axis=1), np.argmax(flip_mask, axis=1), -1
    )

    # Where a valid sign flip was found, return the value of x to the left and the right
    xmin = np.where(i_at_sign_flip != -1, x[i_at_sign_flip], np.nan)
    xmax = np.where(i_at_sign_flip != -1, x[i_at_sign_flip + 1], np.nan)
    return xmin, xmax
