import torch
import matplotlib.pyplot as plt
from scatterlens.visualisation.utils import *

def plot_Cov(
        cov: torch.Tensor,
        zbin_combos: list[tuple[int,...]],
        cmap="cividis",
):
    """Plot normalised covariance matrix."""
    from scatterlens.utils import cov2corr
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    corr = cov2corr(cov)

    fig, ax = plt.subplots(figsize=cm2inch(onecol_wth, onecol_wth))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_ticks([1., 0.5, 0., -0.5, -1.])

    # Separate zbin_combos into auto, pairs, and 3+ bins
    zbin_combo_type_counts = np.array([
        0,
        len([combo for combo in zbin_combos if len(combo) == 1]),
        len([combo for combo in zbin_combos if len(combo) == 2]),
        len([combo for combo in zbin_combos if len(combo) > 2]),
    ])
    dv_length_per_zbin_combo = cov.shape[0] // len(zbin_combos)
    separation_edges = np.cumsum(zbin_combo_type_counts * dv_length_per_zbin_combo)

    for x in separation_edges[1:-1]:
        ax.axvline(x=x, color="black", lw=0.7)
        ax.axhline(y=x, color="black", lw=0.7)

    for x, s in zip(
            (separation_edges[1:] + separation_edges[:-1]) / 2,
            ["auto", "pairs", "all"]
    ):
        ax.text(x=x, y=cov.shape[1] * 1.05, s=s, ha="center", va="top")


def plot_St_coefs_cov_training(
        st_coefs: torch.Tensor,
        zbin_combos: list[tuple[int, ...]],
        n_cols: int=4,
        J: int | None=None,
        ylabel: str=r"$s_{1/2}$",
        color: str="石涅",
):
    """Plot scattering coefficients from covariance training data."""
    if st_coefs.ndim == 1:
        st_coefs = st_coefs.unsqueeze(0)

    fig, axs, dv_length_per_combo = create_zbin_combo_subplots(
        dv_length=st_coefs.shape[1], zbin_combos=zbin_combos, n_cols=n_cols, J=J)

    for i, ax in enumerate(axs):
        for stcoef in st_coefs:
            ax.plot(
                stcoef[dv_length_per_combo * i: dv_length_per_combo * (i + 1)],
                color=color,
                linestyle="solid",
                zorder=0,
                lw=0.5,
                alpha=0.05,
            )
    fig.text(
        x=0.02, y=0.5, s=ylabel,
        va="center", rotation="vertical",
    )
