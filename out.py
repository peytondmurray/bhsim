"""Generate output figures of black hole images."""
from typing import Iterable, Optional, Tuple

import cmocean.cm as ccm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mt
import numpy as np
import numpy.typing as npt
import scipy.interpolate as si

import bh
from data import Isoradial

plt.style.use("dark_background")
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16


def generate_isoradials(
    th0: float,
    r_vals: npt.NDArray[float],
    n_vals: Iterable[int],
    color: Optional[Tuple[float, float, float, float]] = None,
    cmap: Optional[matplotlib.colors.LinearSegmentedColormap] = None,
) -> matplotlib.figure.Figure:
    """Generate a svg of the isoradials.

    Parameters
    ----------
    th0 : float
        Inclination of the observer with respect to the accretion disk normal
    r_vals : npt.NDArray[float]
        Distances from the black hole center to the isoradials to be plotted
    n_vals : Iterable[int]
        Order of the images calculated
    color: Optional[Tuple(float, float, float, float)]
        RGBA tuple of the color to use to plot the isoradials; otherwise, a colormap is used to map
        isoradials of different r to different colors
    cmap: Optional[matplotlib.colors.LinearSegmentedColormap]
        Colormap to use if color is None

    Returns
    -------
    matplotlib.figure.Figure
        Image of the given black hole isoradials.
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("S")
    ax.set_axis_off()

    alpha = np.linspace(0, 2 * np.pi, 1000)
    theta_0 = th0 * np.pi / 180

    if cmap is None:
        cmap = ccm.ice

    for n in sorted(n_vals)[::-1]:
        for r in r_vals:
            if color is None:
                linecolor = cmap((r - np.min(r_vals)) / (np.max(r_vals) - np.min(r_vals)))
            else:
                linecolor = color

            iso = Isoradial(
                bh.reorient_alpha(alpha, n),
                bh.impact_parameter(alpha, r, theta_0, n=n, m=1),
                r,
                theta_0,
                n,
            )

            ax.plot(iso.alpha, iso.b, color=linecolor)

    return fig


def generate_scatter_image(
    ax: Optional[matplotlib.axes.Axes],
    alpha: npt.NDArray[float],
    r_vals: npt.NDArray[float],
    th0: float,
    n_vals: Iterable[int],
    m: float,
    cmap: Optional[matplotlib.colors.LinearSegmentedColormap] = None,
) -> matplotlib.figure.Figure:
    """Generate an image of the black hole using a scatter plot.

    Parameters
    ----------
    ax : Optional[matplotlib.axes.Axes]
        Axes on which the image is to be drawn
    alpha: npt.NDArray[float]
        Polar angles for which the flux is to be plotted
    r_vals : npt.NDArray[float]
        Distances from the black hole center to the isoradials to be plotted
    th0 : float
        Inclination of the observer with respect to the accretion disk normal
    n_vals : Iterable[int]
        Order of the images calculated
    m : float
        Mass of the black hole
    cmap : Optional[matplotlib.colors.LinearSegmentedColormap]
        Colormap to use for plotting the image

    Returns
    -------
    matplotlib.figure.Figure
        Image of the given black hole isoradials.
    """
    theta_0 = th0 * np.pi / 180
    df = bh.generate_image_data(alpha, r_vals, theta_0, n_vals, m, {"max_steps": 3})

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    else:
        fig = ax.get_figure()

    if cmap is None:
        cmap = ccm.gray

    ax.set_theta_zero_location("S")
    ax.set_axis_off()
    for n in sorted(n_vals, reverse=True):
        df_n = df.loc[df["n"] == n]
        ax.scatter(df_n["alpha"], df_n["b"], c=df_n["flux"], cmap=cmap)
    return fig


def generate_image(
    ax: Optional[matplotlib.axes.Axes],
    alpha: npt.NDArray[float],
    r_vals: npt.NDArray[float],
    th0: float,
    n_vals: Iterable[int],
    m: float,
    cmap: Optional[matplotlib.colors.LinearSegmentedColormap],
) -> matplotlib.figure.Figure:
    """Generate an image of the black hole.

    Parameters
    ----------
    ax : Optional[matplotlib.axes.Axes]
        Axes on which the image is to be drawn
    alpha: npt.NDArray[float]
        Polar angles for which the flux is to be plotted
    r_vals : npt.NDArray[float]
        Distances from the black hole center to the isoradials to be plotted
    th0 : float
        Inclination of the observer with respect to the accretion disk normal
    n_vals : Iterable[int]
        Order of the images calculated
    m : float
        Mass of the black hole
    cmap : Optional[matplotlib.colors.LinearSegmentedColormap]
        Colormap to use for plotting the image

    Returns
    -------
    matplotlib.figure.Figure
        Image of the given black hole isoradials.
    """
    theta_0 = th0 * np.pi / 180
    df = bh.generate_image_data(alpha, r_vals, theta_0, n_vals, m, {"max_steps": 3})

    if ax is None:
        _, ax = plt.subplots(figsize=(30, 30))

    ax.set_axis_off()

    minx, maxx = df["x"].min(), df["x"].max()
    miny, maxy = df["y"].min(), df["y"].max()

    xx, yy = np.meshgrid(
        np.linspace(minx, maxx, 10000),
        np.linspace(miny, maxy, 10000),
    )

    fluxgrid = np.zeros(xx.shape, dtype=float)

    for n in n_vals:
        df_n = df.loc[df["n"] == n]
        fluxgrid_n = si.griddata(
            (df_n["x"], df_n["y"]), df_n["flux"], (xx, yy), method="linear"
        )
        fluxgrid[fluxgrid == 0] += fluxgrid_n[fluxgrid == 0]

    cmap = ccm.gray

    transform = mt.Affine2D().rotate_deg_around(0, 0, -90)
    ax.imshow(
        fluxgrid,
        interpolation="bilinear",
        cmap=cmap,
        aspect="equal",
        origin="lower",
        extent=[minx, maxx, miny, maxy],
        transform=transform + ax.transData,
    )

    edges = transform.transform(
        np.array([(minx, miny), (maxx, miny), (minx, maxy), (maxx, maxy)])
    )

    ax.set_xlim(edges[:, 0].min(), edges[:, 0].max())
    ax.set_ylim(edges[:, 1].min(), edges[:, 1].max())

    return ax.get_figure()
