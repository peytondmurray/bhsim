"""Data containers for black hole simulations."""
import pandas as pd


class Isoradial:
    """Data container for isoradial curves."""

    def __init__(self, alpha, b, r, theta_0, n):
        self.alpha = alpha
        self.b = b
        self.r = r
        self.theta_0 = theta_0
        self.n = n

    def to_df(self) -> pd.DataFrame:
        """Generate a DataFrame of alpha, b for the isoradial.

        Returns
        -------
        pd.DataFrame
            alpha, b values in DataFrame format
        """
        return pd.DataFrame({"alpha": self.alpha, "b": self.b})
