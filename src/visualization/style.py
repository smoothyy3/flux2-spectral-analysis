"""Shared matplotlib style configuration for all visualisation modules.

Import and call ``apply_style()`` at the top of every plotting module to
ensure visual consistency across all figures.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

GROUP_COLORS: dict[str, str] = {
    "real": "#2c7bb6",
    "klein_4b_distilled": "#d7191c",
    "klein_4b_base": "#fdae61",
    "vae_roundtrip": "#1a9641",
    "blur": "#abd9e9",
    "downup": "#74add1",
    "jpeg": "#4575b4",
}


def get_color(group_name: str) -> str:
    """Look up the canonical colour for a group by substring match.

    Iterates over ``GROUP_COLORS`` keys and returns the colour for the
    first key that is a substring of *group_name*.  Falls back to a
    neutral grey if no match is found.

    Parameters
    ----------
    group_name : str
        Name of the image group (e.g. ``"klein_4b_distilled"``,
        ``"real"``, ``"blur_1.5"``).

    Returns
    -------
    str
        Hex colour string.
    """
    for key, color in GROUP_COLORS.items():
        if key in group_name:
            return color
    return "#888888"


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Apply the project-wide matplotlib style.

    Sets rcParams for:

    * Sans-serif font family (DejaVu Sans)
    * Font size 11
    * Figure DPI 100
    * Right and top spines removed from all new axes
    * tight_layout enabled by default

    Call this once at module level in every plotting module.
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "figure.autolayout": True,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
