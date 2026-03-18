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
        ylabel: str=r"$\mathbf{d}_\mathrm{LOOCV} / \mathbf{d}_\mathrm{True}$",
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
                        lw=0.9,
                        color="丹罽",
                        alpha=1.0,
                        zorder=1,
                    )
                else:
                    ax.plot(
                        st_deviation[
                            dv_length_per_combo * i: dv_length_per_combo * (i + 1)
                        ],
                        linestyle="solid",
                        lw=0.5,
                        color="法翠",
                        alpha=0.5,
                        zorder=0,
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
        cmap: str="viridis",
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

    for param_index, (param_name, param_texlabel, param_range) in enumerate(
            zip(param_names, param_texlabels, param_ranges)
    ):
        param_values = torch.linspace(*param_range, steps=n_preds)
        cosmo_params = torch.clone(
            torch.broadcast_to(fid_cosmo_params, size=(n_preds, 4)))
        cosmo_params[:, param_index] = param_values

        pred_st_coefs = []
        for cosmo_param in cosmo_params:
            X = cosmo_param.unsqueeze(dim=0)
            pred = emulator.predict(X).squeeze() / fid_st_coef
            pred_st_coefs.append(pred)

        fig, axs = plot_LOOCV(
            st_deviation_list=pred_st_coefs,
            zbin_combos=zbin_combos,
            std=std,
            param_values=param_values,
            param_range=param_range,
            param_texlabel=param_texlabel,
            cmap=cmap,
            n_cols=n_cols,
            J=J,
            ylabel=ylabel,
            savedir=savedir,
            fname=fname.format(param_name),
            return_fig_ax=True,
        )
        fig_list.append(fig)
        axs_list.append(axs)

    if return_fig_ax:
        return fig_list, axs_list




