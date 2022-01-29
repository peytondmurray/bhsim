"""Benchmarking routines used for optimization."""
from typing import Iterable

import numpy as np
import numpy.typing as npt

import bh
from data import Isoradial


def benchmark(th0: float, r_values: npt.NDArray[float], n_vals: Iterable[int]):
    """Run the isoradial benchmark routine.

        Run with pyinstrument or your favorite benchmarking tool to benchmark.

    Parameters
    ----------
    th0 : float
        Inclination with respect to the accretion disk normal, in degrees
    r_values : npt.NDArray[float]
        Isoradial distances
    n_vals : Iterable[int]
        Order of the calculations
    """
    alpha = np.linspace(0, 2 * np.pi, 1000)
    theta_0 = th0 * np.pi / 180

    isoradials = []
    for r in r_values:
        for n in sorted(n_vals)[::-1]:
            isoradials.append(
                Isoradial(
                    bh.reorient_alpha(alpha, n),
                    bh.impact_parameter(alpha, r, theta_0, n, 1),
                    r,
                    theta_0,
                    n,
                )
            )

    return


if __name__ == "__main__":
    benchmark(th0=80, r_values=np.arange(6, 30, 2), n_vals=[0, 1])
