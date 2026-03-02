from scatterlens.visualisation.utils import *

def plot_Cov(
        cov: torch.Tensor,
        zbin_combos: list[tuple[int,...]],
        cmap="cividis",
        savedir: str | None=None,
        fname: str="cov.pdf",
        return_fig_ax: bool=False,
        zbin_combo_type_labels: list[str] | None=None,
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

    if zbin_combo_type_labels is not None:
        # Separate zbin_combos into auto, pairs, and 3+ bins
        zbin_combo_type_counts = np.array([
            0,
            len([combo for combo in zbin_combos if len(combo) == 1]),
            len([combo for combo in zbin_combos if len(combo) == 2]),
            len([combo for combo in zbin_combos if len(combo) > 2]),
        ])
        dv_length_per_zbin_combo = corr.shape[0] // len(zbin_combos)
        separation_edges = np.cumsum(zbin_combo_type_counts * dv_length_per_zbin_combo)

        for x in separation_edges[1:-1]:
            ax.axvline(x=x, color="black", lw=0.7)
            ax.axhline(y=x, color="black", lw=0.7)

        for x, s in zip(
                (separation_edges[1:] + separation_edges[:-1]) / 2,
                zbin_combo_type_labels,
        ):
            ax.text(x=x, y=corr.shape[1] * 1.05, s=s, ha="center", va="top")

    if savedir is not None:
        fig.savefig(
            os.path.join(savedir, fname),
            bbox_inches="tight",
        )

    if return_fig_ax:
        return fig, ax


def plot_St_coefs(
        st_coefs: torch.Tensor,
        zbin_combos: list[tuple[int, ...]],
        n_cols: int=4,
        J: int | None=None,
        ylabel: str=r"$s_{1/2}$",
        color: str="石涅",
        savedir: str | None=None,
        fname: str="st_coefs.pdf",
        return_fig_ax: bool=False,
):
    """Plot scattering coefficients from covariance training data."""
    if st_coefs.ndim == 1:
        st_coefs = st_coefs.unsqueeze(0)

    fig, axs, dv_length_per_combo = create_zbin_combo_subplots(
        dv_length=st_coefs.shape[1],
        zbin_combos=zbin_combos,
        n_cols=n_cols,
        ylabel=ylabel,
        J=J,
    )

    for i, ax in enumerate(axs):
        for st_coef in st_coefs:
            ax.plot(
                st_coef[dv_length_per_combo * i: dv_length_per_combo * (i + 1)],
                color=color,
                linestyle="solid",
                zorder=0,
                lw=0.5,
                alpha=0.05,
            )

    if savedir is not None:
        fig.savefig(
            os.path.join(savedir, fname),
            bbox_inches="tight",
        )

    if return_fig_ax:
        return fig, axs


def plot_St_coefs_vs_param(
        st_coefs: torch.Tensor,
        zbin_combos: list[tuple[int, ...]],
        param_values: torch.Tensor,
        st_coef_fid: torch.Tensor | None=None,
        param_range: tuple[float, float] | None=None,
        param_name: str="param",
        param_texlabel: str | None=None,
        n_cols: int=4,
        J: int | None=None,
        ylabel: str=r"$s_{1/2}$",
        cmap: str="viridis",
        savedir: str | None=None,
        fname: str="st_coefs_vs_{}.pdf",
        return_fig_ax: bool=False,
):
    """Plot scattering coefficients from cosmology training data."""
    if st_coefs.ndim == 1:
        st_coefs = st_coefs.unsqueeze(0)

    if st_coef_fid is not None:
        st_coefs /= st_coef_fid.clone()

    if param_range is None:
        param_range = (param_values.min().item(), param_values.max().item())

    sm, colormap = get_colormap(cmap, param_range)

    fig, axs, dv_length_per_combo = create_zbin_combo_subplots(
        dv_length=st_coefs.shape[1],
        zbin_combos=zbin_combos,
        n_cols=n_cols,
        ylabel=ylabel,
        J=J,
    )

    for i, ax in enumerate(axs):
        for st_coef, param_value in zip(st_coefs, param_values):
            ax.plot(
                st_coef[dv_length_per_combo * i: dv_length_per_combo * (i + 1)],
                color=colormap(param_value),
                zorder=0,
                linewidth=0.5,
                alpha=1.0,
            )

    add_shared_colorbar_to_figure(
        fig=fig,
        axs=axs,
        sm=sm,
        param_range=param_range,
        param_texlabel=param_texlabel,
    )

    if savedir is not None:
        fig.savefig(
            os.path.join(savedir, fname.format(param_name)),
            bbox_inches="tight",
        )

    if return_fig_ax:
        return fig, axs