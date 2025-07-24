import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

onecol_wth: float = 8.8
median_wth: float = 12.0
fullpg_wth: float = 18.0

plt.style.use("/data1/jliu/scattering-transform-kids/scatterlens/matplotlibrc")

def cm2inch(*args: float | int) -> float | tuple[float, ...]:
    if len(args)==1:
        return args[0] / 2.54
    else:
        return tuple(x / 2.54 for x in args)


def create_zbincombo_subplots(
        DV: np.ndarray, zbincombo_list, n_cols: int=4, J: int=None):
    mpl.rcParams["figure.constrained_layout.use"] = False

    n_zbincombo = len(zbincombo_list)
    n_rows = (n_zbincombo + n_cols - 1) // n_cols
    dv_length = len(DV) // n_zbincombo # data vector length for each zbincombo

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=cm2inch(onecol_wth, n_rows * 2.0),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(hspace=0., wspace=0.)
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        ax.set_xlim(left=0, right=dv_length - 1)
        ax.set_xticks([])
        ax.tick_params(axis="y", which="major", pad=2.0, length=3.0, right=False)
        ax.text(x=0.97, y=0.85, s=zbincombo2label(zbincombo_list[i]),
                ha="right", fontsize=6, transform=ax.transAxes)
        if i < n_zbincombo:
            if J is not None:
                ax.axvline(x=J, color="grey", lw=1.0, ls="solid", alpha=0.5, zorder=10)
                if i < n_cols: # first row
                    ax.set_title(label=r"$s_1\ \ \ \ \ \ \ \ s_2$",
                                 x=0.13, loc="left", color="grey", fontsize=8)
        else:
            ax.axis("off")

    return fig, axs, n_rows, dv_length



def zbincombo2label(zbincombo: tuple[int, int]) -> str:
    """Convert a zbincombo tuple to a string label."""
    # TODO only works for cross-zbin pairs in KiDS-1000
    if zbincombo[0] == zbincombo[1]:
        if zbincombo[0] == 0:
             return f"1 + 2 + 3 + 4 + 5"
        return f"{zbincombo[0]}"
    else:
        return f"{zbincombo[0]} + {zbincombo[1]}"


class EmulationViz:
    @staticmethod
    def LOOCV(MSE_LIST, STD=None, zbincombo_list=None, n_cols=4, J=3, savepath=None):
        """Plot Leave-One-Out Cross-Validation errors. This plot is for relative
        errors only."""
        fig, axs, n_rows, dv_length = create_zbincombo_subplots(
            DV=MSE_LIST[0], zbincombo_list=zbincombo_list, n_cols=n_cols, J=J)

        for i, ax in enumerate(axs):
            if i < len(zbincombo_list):
                for j, MSE in enumerate(MSE_LIST):
                    MSE_zbincombo = MSE[dv_length * i: dv_length * (i + 1)]
                    ax.plot(MSE_zbincombo,
                            linestyle="solid",
                            lw=0.5 if j==0 else 0.4,
                            color="red" if j==0 else "dodgerblue",
                            alpha=1.0 if j==0 else 0.5,
                            zorder=1 if j==0 else 0,)
                if STD is not None:
                    STD_zbincombo = STD[dv_length * i: dv_length * (i + 1)]
                    ax.fill_between(np.arange(len(STD_zbincombo)),
                                    1 - STD_zbincombo, 1 + STD_zbincombo,
                                    color="moccasin", alpha=1.0, lw=0.0, zorder=-1)

                ax.set_ylim(bottom=0.95, top=1.05)
                ax.set_yticks([0.97, 1.0, 1.03])
                ax.set_yticklabels(["0.97", "1.00", "1.03"], fontsize=7)

        fig.text(
            x=0.02, y=0.5, s=r"$\mathbf{d}_\mathrm{LOOCV} / \mathbf{d}_\mathrm{True}$",
            va="center", rotation="vertical", fontsize=8,
        )

        if savepath:
            fig.savefig(savepath)


    @staticmethod
    def predictions_vs_param():
        """Plot emulator predictions across cosmological parameter values."""
        pass


    @staticmethod
    def predictions_vs_param_CosmoSLICS(
            emulator, param: str,
            FID_PARAMS: torch.tensor, FID_STDV: np.ndarray, STD=None,
            zbincombo_list=None, n_preds: int = 10, n_cols=4, J=3,
            cmap: str = "cool", savepath=None,
    ):
        """Plot emulator predictions across cosmological parameter values."""
        params = ["Omega_m", "S_8", "h", "w_0"]
        param_labels = [r"$\Omega_m$", r"$S_8$", r"$h$", r"$w_0$"]
        param_ranges = [[0.1, 0.55], [0.6, 0.9], [0.6, 0.9], [-2.0, 0.5]]

        if param not in params:
            raise ValueError(f"Parameter {param} not recognized.")
        param_range = param_ranges[params.index(param)]
        param_label = param_labels[params.index(param)]

        param_values = torch.linspace(*param_range, steps=n_preds)
        cosm_params = torch.clone(
            torch.broadcast_to(FID_PARAMS, size=(n_preds, 4)))
        cosm_params[:, params.index(param)] = param_values

        preds = []
        for cosm_param in cosm_params:
            X = cosm_param.unsqueeze(dim=0)
            pred = emulator.predict(X).squeeze()
            preds.append(pred)

        # Configure colormap and normalization
        colormap = mpl.colormaps[cmap]
        norm = mpl.colors.Normalize(vmin=param_range[0], vmax=param_range[1])
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        fig, axs, n_rows, dv_length = create_zbincombo_subplots(
            DV=FID_STDV, zbincombo_list=zbincombo_list, n_cols=n_cols, J=J)

        for i, ax in enumerate(axs):
            if i < len(zbincombo_list):
                for pred, param_value in zip(preds, param_values):
                    ax.plot(pred[dv_length * i: dv_length * (i + 1)] /
                            FID_STDV[dv_length * i: dv_length * (i + 1)],
                            color=colormap(norm(param_value)),
                            linestyle="solid", zorder=0, lw=0.5, alpha=0.5)
            ax.set_ylim(bottom=0.89, top=1.11)
            ax.set_yticks([0.93, 1.0, 1.07])
            ax.set_yticklabels(["0.93", "1.00", "1.07"], fontsize=7)

            if STD is not None:
                STD_zbincombo = STD[dv_length * i: dv_length * (i + 1)]
                ax.fill_between(np.arange(len(STD_zbincombo)),
                                1 - STD_zbincombo, 1 + STD_zbincombo,
                                color="moccasin", alpha=1.0, lw=0.0, zorder=-1)

        fig.text(
            x=0.02, y=0.5,
            s=r"$\mathbf{d}_\mathrm{pred} / \mathbf{d}_\mathrm{fid}$",
            va="center", rotation="vertical", fontsize=8,
        )

        cbar = fig.colorbar(
            sm, ax=axs, orientation='horizontal', fraction=0.03, pad=0.05, aspect=30)
        cbar_ticks = np.linspace(param_range[0], param_range[1], num=4)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(
            ticklabels=[f"{tick:.2f}" for tick in cbar_ticks], fontsize=7)
        cbar.ax.tick_params(length=2.0)
        cbar.set_label(param_label, fontsize=8)

        if savepath:
            fig.savefig(savepath)



class STDataViz:
    @staticmethod
    def STDV_vs_param():
        pass

    @staticmethod
    def COV():
        pass