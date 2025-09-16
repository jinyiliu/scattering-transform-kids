import os
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
import cmplstyle
from cmplstyle import onecol_wth, median_wth, fullpg_wth, cm2inch
cmplstyle.use_builtin_mplstyle()

plt.rcParams["figure.constrained_layout.use"] = False

def zbin_combo_label(zbin_combo: tuple[int, ...]) -> str:
    """Convert a zbin combination to a string label."""
    return " + ".join(str(z) for z in zbin_combo)


def create_zbin_combo_subplots(
        dv_length: int,
        zbin_combos: list[tuple[int, ...]],
        n_cols: int=4,
        J: int=None,
        ylabel: str=r"$s_{1/2}$",
):
    """Create subplots for each zbin combination.

    Args:
        dv_length: Length of the data vector.
        zbin_combos: List of zbin combinations.
        n_cols: Number of columns in the subplot grid.
        J: Position to separate first- and second-order scattering coefficients.
            If None, no separation line is drawn.
        ylabel: Label for the y-axis.

    Returns:
        fig: The figure object.
        axs: Flattened array of axes objects.
        dv_length_per_combo: Length of the data vector for each zbin combination.
    """
    n_combo = len(zbin_combos)
    n_rows = (n_combo + n_cols - 1) // n_cols
    dv_length_per_combo = dv_length // n_combo # data vector length for each zbincombo

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=cm2inch(onecol_wth, n_rows * (onecol_wth / n_cols) * 0.9),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(hspace=0., wspace=0.)
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i < n_combo:
            ax.set_xlim(left=0, right=dv_length_per_combo - 1)
            ax.set_xticks([])
            ax.tick_params(
                axis="y", which="major", pad=2.0, length=3.0, right=False, labelsize=7)
            ax.text(
                x=0.97, y=0.85,
                s=zbin_combo_label(zbin_combos[i]),
                ha="right",
                fontsize=6,
                transform=ax.transAxes,
            )
            if J is not None:
                ax.axvline(x=J, color="grey", lw=1.0, ls="solid", alpha=0.5, zorder=10)
                if i < n_cols: # first row
                    ax.set_title(label=r"$s_1\ \ \ \ \ \ \ \ s_2$",
                                 x=0.13, loc="left", color="grey", fontsize=8)
        else:
            ax.axis("off")

    fig.text(
        x=0.02, y=0.5, s=ylabel,
        va="center", rotation="vertical",
    )

    return fig, axs[:n_combo], dv_length_per_combo


def get_colormap(
        cmap: str,
        param_range: tuple[float, float],
) -> [plt.cm.ScalarMappable, Callable]:
    colormap_ = mpl.colormaps[cmap]
    norm = mpl.colors.Normalize(vmin=param_range[0], vmax=param_range[1])
    sm = plt.cm.ScalarMappable(cmap=colormap_, norm=norm)
    sm.set_array([])

    def colormap(param_value: float):
        return colormap_(norm(param_value))

    return sm, colormap


def add_shared_colorbar_to_figure(
        fig: mpl.figure.Figure,
        axs: np.ndarray,
        sm: plt.cm.ScalarMappable,
        param_range: tuple[float, float],
        param_texlabel: str | None=None,
):
    """Add a colorbar to the figure."""
    cbar = fig.colorbar(
        sm,
        ax=axs,
        orientation="horizontal",
        fraction=0.03,
        pad=0.05,
        aspect=30,
    )
    cbar_ticks = np.linspace(param_range[0], param_range[1], num=4)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(
        ticklabels=[f"{tick:.2f}" for tick in cbar_ticks], fontsize=7,
    )
    cbar.ax.tick_params(length=2.0)
    if param_texlabel is not None:
        cbar.set_label(param_texlabel, fontsize=8)