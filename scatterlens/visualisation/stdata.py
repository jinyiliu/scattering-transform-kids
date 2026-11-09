import seaborn as sns

from scatterlens.visualisation.utils import *
from scatterlens.kids1000_sims import IAMocks

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
        zbin_combo_type_counts = [
            len([combo for combo in zbin_combos if len(combo) == 1]),
            len([combo for combo in zbin_combos if len(combo) == 2]),
            len([combo for combo in zbin_combos if len(combo) > 2]),
        ]
        # Drop empty categories (and their labels) so auto+cross-only works.
        labels = [
            lab for lab, n in zip(zbin_combo_type_labels, zbin_combo_type_counts)
            if n > 0
        ]
        zbin_combo_type_counts = [n for n in zbin_combo_type_counts if n > 0]

        dv_length_per_zbin_combo = corr.shape[0] // len(zbin_combos)
        separation_edges = np.cumsum(
            [0] + [n * dv_length_per_zbin_combo for n in zbin_combo_type_counts]
        )

        for x in separation_edges[1:-1]:
            ax.axvline(x=x, color="black", lw=0.7)
            ax.axhline(y=x, color="black", lw=0.7)

        for x, s in zip(
                (separation_edges[1:] + separation_edges[:-1]) / 2,
                labels,
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
        std: torch.Tensor | None=None,
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
    st_coefs = st_coefs.clone()
    if st_coefs.ndim == 1:
        st_coefs = st_coefs.unsqueeze(0)

    if st_coef_fid is not None:
        st_coefs /= st_coef_fid.clone()
        if std is not None:
            std /= st_coef_fid.clone()

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

        if std is not None:
            ax.fill_between(
                x=np.arange(dv_length_per_combo),
                y1=1 - std[
                   dv_length_per_combo * i: dv_length_per_combo * (i + 1)
                   ],
                y2=1 + std[
                   dv_length_per_combo * i: dv_length_per_combo * (i + 1)
                   ],
                color="荻色",
                alpha=1.0,
                zorder=-1,
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

def plot_IA_poly_fit(
        j1j2_indices: list[tuple[int, int | None]],
        zbin_combos: list[tuple[int, ...]],
        IA_poly_coefs: str | torch.Tensor,
        IA_scoef_tensor: torch.Tensor,
        markers: list[str] | None=None,
        savedir: str | None=None,
        fname: str="IA_poly_fit.pdf",
        return_fig_ax: bool=False,
):
    A_IA_values = np.array(IAMocks.A_IA_values)
    nrows = len(j1j2_indices)

    _n_scoef_per_zb = IA_scoef_tensor.shape[1] // len(IAMocks.zbin_combos)
    J = int((np.sqrt(8. * _n_scoef_per_zb + 1) - 1) / 2)
    _S2_pairs = [(j1, j2) for j1 in range(J) for j2 in range(j1 + 1, J)]
    coef_pos = lambda j1, j2: j1 if j2 is None else J + _S2_pairs.index((j1, j2))
    zb_pos = lambda zb_combo: IAMocks.zbin_combos.index(zb_combo) * _n_scoef_per_zb

    if isinstance(IA_poly_coefs, str):
        IA_poly_coefs = torch.load(IA_poly_coefs, weights_only=True)

    if isinstance(IA_poly_coefs, torch.Tensor):
        poly = np.array(IA_poly_coefs)
        degree = poly.shape[0] - 1
    else:
        poly = None

    fig, axes = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=cm2inch(onecol_wth, onecol_wth * 0.65 * nrows),
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.05)
    if nrows == 1:
        axes = [axes]

    A_IA_grid = np.linspace(start=-7., stop=7., num=200)

    for p, (ax, (j1, j2)) in enumerate(zip(axes, j1j2_indices)):
        ylabel = j1j2_label((j1, j2))
        y_top = 0.
        y_bottom = np.inf
        for zbin_combo, m in zip(zbin_combos, markers):
            line_label = zbin_combo_label(zbin_combo)
            _idx = zb_pos(zbin_combo) + coef_pos(j1, j2)
            y_meas = np.array(IA_scoef_tensor)[:, _idx]
            y_fit = np.vander(A_IA_grid, degree + 1, increasing=False) @ poly[:, _idx]

            y_meas *= 1 if j2 else 100
            y_fit *= 1 if j2 else 100

            c_meas = "朱柿"
            c_fit = sns.light_palette("油紫", n_colors=6)[1]
            y_top = max(y_fit) if max(y_fit) > y_top else y_top
            y_bottom = min(y_fit) if min(y_fit) < y_bottom else y_bottom
            y_range = y_top - y_bottom

            ax.plot(
                A_IA_values, y_meas,
                marker=m,
                linestyle="none",
                markersize=4.5,
                markerfacecolor=c_meas if len(zbin_combo) == 1 else "none",
                markeredgecolor="青骊",
                markeredgewidth=0.6,
                label=line_label if p == 0 else None,
                zorder=1,
            )
            ax.plot(
                A_IA_grid, y_fit,
                color=c_fit,
                linewidth=2.5,
                zorder=0,
            )

        ax.set_ylabel(ylabel)
        ax.set_xticks(ticks=[-6, -4, -2, 0, 2, 4, 6])
        ax.set_xlim(left=-7.5, right=7.5)
        ax.set_ylim(
            top=y_top + y_range * 0.13,
            bottom=y_bottom - y_range * 0.13,
        )

        if p == 0:
            ax.legend(
                loc="lower left",
                handletextpad=0.,
                bbox_to_anchor=(-0.03, 0.97),
                ncols=5,
                columnspacing=1.2,
                frameon=False,
                edgecolor="云母",
                facecolor="none",
                fancybox=True,
                prop={"family": "Calibri", "size": 8},
            )

        if p == nrows - 1:
            ax.set_xlabel(r"$A_\mathrm{IA}$")


    if savedir is not None:
        fig.savefig(
            os.path.join(savedir, fname),
            bbox_inches="tight",
        )

    if return_fig_ax:
        return fig, axes


def j1j2_label(j1j2_index: tuple[int, int | None]) -> str:
    j1, j2 = j1j2_index
    if j2 is None:
        return rf"$10^2\,s_1^{{\, j_1={j1}}}$"
    return rf"$s_2^{{\,j_1={j1},\,j_2={j2}}} \,\left/\, s_1^{{\,j_1={j1}}}\right.$"


def plot_IA_St_coefs(
        IA_poly_coefs: str | torch.Tensor,
        zbin_combos: list[tuple[int, ...]],
        cmap: str="cool",
        n_cols: int=4,
        J: int | None=None,
        ylabel: str=r"$s_{1/2}$",
        savedir: str | None=None,
        fname: str="IA_St_coefs.pdf",
):
    A_IA_values = np.linspace(-5., 5., num=100)
    A_IA_range = (A_IA_values.min(), A_IA_values.max())

    if isinstance(IA_poly_coefs, str):
        IA_poly_coefs = torch.load(IA_poly_coefs, weights_only=True)

    poly = np.array(IA_poly_coefs, dtype=float)               # (degree + 1, n_coef)
    degree = poly.shape[0] - 1
    X = np.vander(A_IA_values, degree + 1, increasing=False)  # (n_A_IA, degree + 1)
    model = X @ poly                                          # (n_A_IA, n_coef)
    model /= model[6]

    fig, axs, _n_scoef_per_zb = create_zbin_combo_subplots(
        dv_length=model.shape[1],
        zbin_combos=zbin_combos,
        n_cols=n_cols,
        ylabel=ylabel,
        J=J,
    )

    sm, colormap = get_colormap(cmap, A_IA_range)
    add_shared_colorbar_to_figure(
        fig=fig,
        axs=axs,
        sm=sm,
        param_range=A_IA_range,
        param_texlabel=r"$A_\mathrm{IA}$",
    )

    for i, ax in enumerate(axs):
        for IA_idx, A_IA in enumerate(A_IA_values):
            ax.plot(
                model[IA_idx, _n_scoef_per_zb * i: _n_scoef_per_zb * (i + 1)],
                color=colormap(A_IA),
                zorder=0,
                linewidth=0.5,
                alpha=0.5,
            )

        # ax.set_ylim(bottom=0.95, top=1.05)
        # ax.set_yticks([0.97, 1.00, 1.03])
        ax.tick_params(axis="y", labelsize=8.)

    if savedir is not None:
        fig.savefig(
            os.path.join(savedir, fname),
            bbox_inches="tight",
        )
