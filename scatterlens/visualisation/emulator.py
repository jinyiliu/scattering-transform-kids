import seaborn as sns

from scatterlens.visualisation.utils import *

def plot_LOOCV(
        st_deviation_list: list[np.ndarray],
        zbin_combos: list[tuple[int, ...]],
        st_coef_fid: torch.Tensor | np.ndarray | None=None,
        std: np.ndarray | None=None,
        param_values: torch.Tensor | None=None,
        param_range: tuple[float, float] | None=None,
        param_texlabel: str | None=None,
        cmap: str="viridis",
        n_cols: int=4,
        J: int | None=None,
        ylabel: str=r"$\mathbf{d}_\mathrm{pred} / \mathbf{d}_\mathrm{true}$",
        savedir: str | None=None,
        fname: str="LOOCV.pdf",
        return_fig_ax: bool=False,
):
    """Plot Leave-One-Out Cross-Validation results.

    Args:
        st_deviation_list: List of deviations from the true value for each
            left-out sample.
        zbin_combos:
        st_coef_fid: Fiducial scattering coefficients for the fiducial model.
        std: Standard deviation for the scattering coefficients. Used for
            comparing the deviation to the expected scatter.
        param_values: Parameter values corresponding to each left-out sample.
            If provided, the color of the lines will indicate the parameter
            values.
        param_range:
        param_texlabel:
        cmap:
        n_cols:
        J:
        ylabel:
        savedir:
        fname:
        return_fig_ax:
    """
    if std is not None and st_coef_fid is not None:
        std /= st_coef_fid.clone()

    fig, axs, dv_length_per_combo = create_zbin_combo_subplots(
        dv_length=len(st_deviation_list[0]),
        zbin_combos=zbin_combos,
        n_cols=n_cols,
        ylabel=ylabel,
        J=J,
    )

    if param_values is not None:
        if param_range is None:
            param_range = (
                param_values.min().item(), param_values.max().item()
            )
        sm, colormap = get_colormap(cmap, param_range)
        add_shared_colorbar_to_figure(
            fig=fig,
            axs=axs,
            sm=sm,
            param_range=param_range,
            param_texlabel=param_texlabel,
        )

    for i, ax in enumerate(axs):
        for st_index, st_deviation in enumerate(st_deviation_list):
            if param_values is not None:
                ax.plot(
                    st_deviation[
                    dv_length_per_combo * i: dv_length_per_combo * (i + 1)
                    ],
                    linestyle="solid",
                    lw=0.5,
                    color=colormap(param_values[st_index].item()),
                    alpha=1.0,
                    zorder=0,
                )
            else:
                # Highlight the fiducial model
                if st_index == len(st_deviation_list) - 1:
                    ax.plot(
                        st_deviation[
                            dv_length_per_combo * i: dv_length_per_combo * (i + 1)
                        ],
                        linestyle="solid",
                        lw=2.0,
                        color="鸦青",
                        alpha=1.0,
                        zorder=1,
                    )
                else:
                    ax.plot(
                        st_deviation[
                            dv_length_per_combo * i: dv_length_per_combo * (i + 1)
                        ],
                        linestyle="solid",
                        lw=0.9,
                        color="姚黄",
                        alpha=0.9,
                        zorder=0,
                    )

            ax.set_ylim(bottom=0.95, top=1.05)
            ax.set_yticks([0.97, 1.00, 1.03])
            ax.tick_params(axis="y", labelsize=8.)

            if std is not None:
                ax.fill_between(
                    x=np.arange(dv_length_per_combo),
                    y1=1 - std[
                        dv_length_per_combo * i: dv_length_per_combo * (i + 1)
                    ],
                    y2=1 + std[
                        dv_length_per_combo * i: dv_length_per_combo * (i + 1)
                    ],
                    color=sns.dark_palette("卵白", n_colors=6)[4],
                    alpha=1.0,
                    zorder=-1,
                )

    if savedir is not None:
        fig.savefig(
            os.path.join(savedir, fname),
            bbox_inches="tight",
        )

    if return_fig_ax:
        return fig, axs


def plot_StEmu_parameter_dependence(
        emulator,
        param_texlabels: list[str],
        param_ranges: list[tuple[float, float]],
        param_names: list[str],
        fid_cosmo_params: torch.Tensor,
        fid_st_coef: torch.Tensor,
        zbin_combos: list[tuple[int, int]],
        n_preds: int=10,
        std: np.ndarray | None=None,
        cmap: str | None=None,
        n_cols: int=4,
        J: int | None=None,
        ylabel: str=r"$\mathbf{d}_\mathrm{pred} / \mathbf{d}_\mathrm{fid}$",
        savedir: str | None=None,
        fname: str="Emu_pred_{}.pdf",
        return_fig_ax: bool=False,
):
    """Plot the dependence of scattering coefficients on a parameter from an
    emulator.
    """
    fig_list = []
    axs_list = []

    if std is not None:
        std = np.array(std, dtype=float) / np.array(fid_st_coef, dtype=float)

    n_combo = len(zbin_combos)
    n_rows = (n_combo + n_cols - 1) // n_cols

    for param_index, (param_name, param_texlabel, param_range) in enumerate(
            zip(param_names, param_texlabels, param_ranges)
    ):
        param_values = torch.linspace(*param_range, steps=n_preds)
        cosmo_params = torch.clone(
            torch.broadcast_to(fid_cosmo_params, size=(n_preds, 4)))
        cosmo_params[:, param_index] = param_values

        model = torch.zeros(size=(len(cosmo_params), len(fid_st_coef)))
        for i, cosmo_param in enumerate(cosmo_params):
            X = cosmo_param.unsqueeze(dim=0)
            model[i] = emulator.predict(X).squeeze() / fid_st_coef

        fig, axs, _n_scoef_per_zb = create_zbin_combo_subplots(
            dv_length=model.shape[1],
            zbin_combos=zbin_combos,
            n_cols=n_cols,
            figsize=cm2inch(median_wth, n_rows * (median_wth / n_cols) * 0.97),
            ylabel=ylabel,
            J=J,
        )

        colormap = sns.cubehelix_palette(as_cmap=True)
        if cmap is not None:
            colormap = sns.color_palette(cmap, as_cmap=True)
        norm = plt.Normalize(vmin=param_range[0], vmax=param_range[1])
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)

        add_shared_colorbar_to_figure(
            fig=fig,
            axs=axs,
            sm=sm,
            param_range=param_range,
            param_texlabel=param_texlabel,
        )

        for i, ax in enumerate(axs):
            if std is not None:
                ax.fill_between(
                    x=np.arange(_n_scoef_per_zb),
                    y1=1 - std[_n_scoef_per_zb * i: _n_scoef_per_zb * (i + 1)],
                    y2=1 + std[_n_scoef_per_zb * i: _n_scoef_per_zb * (i + 1)],
                    color=sns.dark_palette("卵白", n_colors=6)[4],
                    alpha=1.0,
                    zorder=-1,
                )
            for p_idx, param_value in enumerate(param_values):
                ax.plot(
                    model[
                        p_idx,
                        _n_scoef_per_zb * i: _n_scoef_per_zb * (i + 1)
                    ],
                    color=colormap(norm(param_value)),
                    zorder=0,
                    linewidth=1.0,
                    alpha=1.0,
                )

            ax.tick_params(axis="y", labelsize=8.)

        if savedir is not None:
            fig.savefig(
                os.path.join(savedir, fname.format(param_name)),
                bbox_inches="tight",
            )

        fig_list.append(fig)
        axs_list.append(axs)

    if return_fig_ax:
        return fig_list, axs_list




