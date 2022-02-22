"""Generate output figures of black hole images."""
from typing import Iterable

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import rich.progress as rp

import bh
from data import Isoradial

plt.style.use("dark_background")
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16


def main():
    """Generate a plot of the isoradials."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("S")
    ax.set_axis_off()

    alpha = np.linspace(0, 2 * np.pi, 1000)
    theta_0 = 80 * (np.pi / 180)
    # r_values = [6, 8, 10, 12, 14, 1 20, 30]
    r_values = np.arange(6, 30, 2)

    alphas, b_values = [], []
    for r in rp.track(r_values):
        color = cmocean.cm.ice(
            (r - np.min(r_values)) / (np.max(r_values) - np.min(r_values))
        )
        for n in [0, 1]:
            b_values.append(bh.impact_parameter_b(alpha, r, theta_0, n=n, M=1))
            alphas.append(bh.alpha_reorient(alpha, n))

            ax.plot(
                bh.alpha_reorient(alpha, n),
                bh.impact_parameter_b(alpha, r, theta_0, n=n, M=1),
                color=color,
            )

    plt.show()


def generate_isoradials(
    th0: float, r_vals: npt.NDArray[float], n_vals: Iterable[int], color=None, save=True
):
    """Generate a svg of the isoradials.

    Parameters
    ----------
    th0 : float
        Inclination of the observer with respect to the accretion disk normal
    r_vals : npt.NDArray[float]
        Distances from the black hole center to the isoradials to be plotted
    n_vals : Iterable[int]
        Order of the images calculated
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("S")
    ax.set_axis_off()

    alpha = np.linspace(0, 2 * np.pi, 1000)
    theta_0 = th0 * np.pi / 180

    for n in sorted(n_vals)[::-1]:
        for r in r_vals:
            if color is None:
                color = cmocean.cm.ice(
                    (r - np.min(r_vals)) / (np.max(r_vals) - np.min(r_vals))
                )

            iso = Isoradial(
                bh.reorient_alpha(alpha, n),
                bh.impact_parameter(alpha, r, theta_0, n=n, m=1),
                r,
                theta_0,
                n,
            )

            ax.plot(iso.alpha, iso.b, color=color)

    if save:
        plt.savefig(f"./isoradials-th0={th0}.svg")

    return plt


if __name__ == "__main__":
    for th0 in np.array([80]):
        generate_isoradials(th0=th0, r_vals=np.arange(6, 30, 2), n_vals=[0, 1, 2])
