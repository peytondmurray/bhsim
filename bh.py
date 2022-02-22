"""A collection of expressions for equations in Luminet's paper used for generating images."""
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt
import scipy.special as sp
import sympy as sy
import sympy.physics.mechanics as sym

import util


def expr_q() -> sy.Symbol:
    """Generate a sympy expression for Q.

    Returns
    -------
    sy.Symbol
        Symbolic expression for Q
    """
    p, m = sy.symbols("P, M")
    return sy.sqrt((p - 2 * m) * (p + 6 * m))


def expr_b() -> sy.Symbol:
    """Generate a sympy expression for b, the radial coordinate in the observer's frame.

    See equation 5 in Luminet (1977) for reference.

    Note: the original paper has an algebra error here; dimensional analysis indicates that the
    square root of P**3/(P-2M) must be taken.

    Returns
    -------
    sy.Symbol
        Symbolic expression for b
    """
    p, m = sy.symbols("P, M")
    return sy.sqrt((p**3) / (p - 2 * m))


def expr_r_inv() -> sy.Symbol:
    """Generate a sympy expression for 1/r.

    See equation 13 from Luminet (1977) for reference.

    Note: this equation has an algebra error; see the docstring for expr_u.

    Returns
    -------
    sy.Symbol
        Symbolic expression for 1/r
    """
    p, m, q, u, k = sy.symbols("P, M, Q, u, k")
    sn = sy.Function("sn")

    return (1 / (4 * m * p)) * (-(q - p + 2 * m) + (q - p + 6 * m) * sn(u, k**2) ** 2)


def expr_u() -> sy.Symbol:
    """Generate a sympy expression for the argument to sn in equation 13 of Luminet (1977).

    See equation 13 from Luminet (1977) for reference.

    Note: the original paper has an algebra error here; sqrt(P/Q) should _divide_ gamma, not
    multiply it.

    Returns
    -------
    sy.Symbol
        Symbolic expression for the argument of sn
    """
    gamma, z_inf, k, p, q, n = sy.symbols("gamma, zeta_inf, k, P, Q, N")
    return sy.Piecewise(
        (gamma / (2 * sy.sqrt(p / q)) + sy.elliptic_f(z_inf, k**2), sy.Eq(n, 0)),
        (
            (gamma - 2 * n * sy.pi) / (2 * sy.sqrt(p / q))
            - sy.elliptic_f(z_inf, k**2)
            + 2 * sy.elliptic_k(k**2),
            True,
        ),
    )


def expr_gamma() -> sy.Symbol:
    """Generate a sympy expression for gamma, an angle that relates alpha and theta_0.

    See equation 10 of Luminet (1977) for reference.

    Returns
    -------
    sy.Symbol
        Symbolic expression for gamma
    """
    alpha, theta_0 = sy.symbols("alpha, theta_0")
    return sy.acos(sy.cos(alpha) / sy.sqrt(sy.cos(alpha) ** 2 + sy.tan(theta_0) ** -2))


def expr_k() -> sy.Symbol:
    """Generate a sympy expression for k; k**2 is used as a modulus in the elliptic integrals.

    See equation 12 of Luminet (1977) for reference.

    Returns
    -------
    sy.Symbol
        Symbolic expression for k
    """
    p, m, q = sy.symbols("P, M, Q")
    return sy.sqrt((q - p + 6 * m) / (2 * q))


def expr_zeta_inf() -> sy.Symbol:
    """Generate a sympy expression for zeta_inf.

    See equation 12 of Luminet (1977) for reference.

    Returns
    -------
    sy.Symbol
        Symbolic expression of zeta_inf
    """
    p, m, q = sy.symbols("P, M, Q")
    return sy.asin(sy.sqrt((q - p + 2 * m) / (q - p + 6 * m)))


def expr_ellipse() -> sy.Symbol:
    """Generate a sympy expression for an ellipse.

    In the newtonian limit, isoradials form these images.

    Returns
    -------
    sy.Symbol
        Symbolic expression for an ellipse viewed at an inclination of theta_0
    """
    r, alpha, theta_0 = sy.symbols("r, alpha, theta_0")
    return r / sy.sqrt(1 + (sy.tan(theta_0) ** 2) * (sy.cos(alpha) ** 2))


def impact_parameter(
    alpha: npt.NDArray[np.float64],
    r_value: float,
    theta_0: float,
    n: int,
    m: float,
    objective_func: Optional[Callable] = None,
    **root_kwargs
) -> npt.NDArray[np.float64]:
    """Calculate the impact parameter for each value of alpha.

    Parameters
    ----------
    alpha : npt.NDArray[np.float64]
        Polar angle in the observer's frame of reference
    r_value : float
        Isoradial distance for which the impact parameter is to be calculated
    theta_0 : float
        Inclination of the observer with respect to the accretion disk plane normal
    n : int
        Order of the calculation; n=0 corresponds to the direct image, n>0 are ghost images
    m : float
        Mass of the black hole
    objective_func : Optional[Callable]
        Objective function whose roots are the periastron distances for a given b(alpha)
    root_kwargs
        Additional arguments are passed to fast_root

    Returns
    -------
    npt.NDArray[np.float64]
        Impact parameter b for each value of alpha. If no root of the objective function for the
        periastron is found for a particular value of alpha, the value of an ellipse is used at that
        point.
    """
    if objective_func is None:
        objective_func = lambda_objective()

    ellipse = lambdify(["r", "alpha", "theta_0"], expr_ellipse())
    b = lambdify(["P", "M"], expr_b())

    p_arr = util.fast_root(
        objective_func,
        np.linspace(2.1, 50, 1000),
        alpha,
        [theta_0, r_value, n, m],
        **root_kwargs
    )
    return np.where(np.isnan(p_arr), ellipse(r_value, alpha, theta_0), b(p_arr, m))


def reorient_alpha(alpha: Union[float, npt.NDArray[float]], n: int) -> float:
    """Reorient the polar angle on the observation coordinate system.

    From Luminet's paper:

        "...the observer will detect generally two images, a direct (or primary) image at polar
        coordinates (b^(d), alpha) and a ghost (or secundary) image at (b^(g), alpha + pi)."

    This function adds pi to the polar angle for ghost images, and returns the original angle for
    direct images.

    Parameters
    ----------
    alpha : float
        Polar angle alpha in the observer's "sensor" coordinate system.
    n : int
        Order of the image which is being calculated. n=0 corresponds to the direct image, while n>0
        corresponds to ghost images.

    Returns
    -------
    float
        Reoriented polar angle
    """
    if n > 0:
        return (alpha + np.pi) % (2 * np.pi)
    return alpha


def lambdify(*args, **kwargs) -> Callable:
    """Lambdify a sympy expression from Luminet's paper.

    Luminet makes use of the sn function, which is one of as Jacobi's elliptic functions. Sympy
    doesn't (yet) support this function, so lambdifying it requires specifying the correct scipy
    routine.

    Arguments are passed diretly to sympy.lambdify; if "modules" is specified, the user must specify
    which function to call for 'sn'.

    Parameters
    ----------
    *args
        Arguments are passed to sympy.lambdify
    **kwargs
        Additional kwargs passed to sympy.lambdify

    Returns
    -------
    Callable
        Lambdified expression
    """
    kwargs["modules"] = kwargs.get(
        "modules",
        [
            "numpy",
            {
                "sn": lambda u, m: sp.ellipj(u, m)[0],
                "elliptic_f": lambda phi, m: sp.ellipkinc(phi, m),
                "elliptic_k": lambda m: sp.ellipk(m),
            },
        ],
    )
    return sy.lambdify(*args, **kwargs)


def objective() -> sy.Symbol:
    """Generate a sympy expression for the objective function.

    The objective function has roots which are periastron distances for isoradials.

    Returns
    -------
    sy.Symbol
        Symbolic expression for the objective function
    """
    r = sy.symbols("r")
    return 1 - r * expr_r_inv()


def lambda_objective() -> Callable[[float, float, float, float, int, float], float]:
    """Generate a lambdified objective function.

    Returns
    -------
    Callable[(float, float, float, float, int, float), float]
        Objective function whose roots yield periastron distances for isoradials. The function
        signature is

            s(P, alpha, theta_0, r, n, m)
    """
    s = (
        objective()
        .subs({"u": expr_u()})
        .subs({"zeta_inf": expr_zeta_inf()})
        .subs({"gamma": expr_gamma()})
        .subs({"k": expr_k()})
        .subs({"Q": expr_q()})
    )
    return lambdify(("P", "alpha", "theta_0", "r", "N", "M"), s)


def expr_fs() -> sy.Symbol:
    """Generate an expression for the flux of an accreting disk.

    See equation 15 of Luminet (1977) for reference.

    Returns
    -------
    sy.Symbol
        Sympy expression for Fs, the radiation flux of an accreting disk
    """
    m, rstar = sy.symbols("M, r^*")
    mdot = sym.dynamicsymbols("m", level=1)

    return (
        ((3 * m * mdot) / (8 * sy.pi))
        * (1 / ((rstar - 3) * rstar ** (5 / 2)))
        * (
            sy.sqrt(rstar)
            - sy.sqrt(6)
            + (sy.sqrt(3) / 3)
            * sy.ln(
                ((sy.sqrt(rstar) + sy.sqrt(3)) * (sy.sqrt(6) - sy.sqrt(3)))
                / ((sy.sqrt(rstar) - sy.sqrt(3)) * (sy.sqrt(6) + sy.sqrt(3)))
            )
        )
    )


def expr_one_plus_z() -> sy.Symbol:
    """Generate an expression for the redshift 1+z.

    See equation 19 in Luminet (1977) for reference.

    Returns
    -------
    sy.Symbol()
        Sympy expression for the redshift of the accretion disk
    """
    m, r, theta_0, alpha, b = sy.symbols("M, r, theta_0, alpha, b")
    return (1 + sy.sqrt(m / r**3) * b * sy.sin(theta_0) * sy.sin(alpha)) / sy.sqrt(
        1 - 3 * m / r
    )


def expr_f_0() -> sy.Symbol:
    """Generate an expression for the observed bolometric flux.

    Returns
    -------
    sy.Symbol

    """
    fs, opz = sy.symbols("F_s, 1+z")
    return fs / opz**4


def lambidify_f0() -> sy.Symbol:
    f0 = (
        expr_f_0()
        .subs({"F_s": expr_fs()})
        .subs({"u": expr_u()})
        .subs({"u": expr_u()})
        .subs({"u": expr_u()})
        .subs({"u": expr_u()})
        .subs({"zeta_inf": expr_zeta_inf()})
        .subs({"gamma": expr_gamma()})
        .subs({"k": expr_k()})
        .subs({"Q": expr_q()})
    )
    return lambdify([], f0)
