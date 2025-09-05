import torch
from scatterlens.visualisation.utils import *


def plot_St_coefs_cov_training(
        st_coefs: torch.Tensor,
        zbin_combos: list[tuple[int, ...]],
        n_cols: int=4,
        J: int | None=None,
        ylabel: str=r"$S_{1/2}$",
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
