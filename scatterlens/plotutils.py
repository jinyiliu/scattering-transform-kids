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
    def LOOCV(MSE_LIST, STD=None, zbincombo_list=None, n_cols=4, J=3):
        """Plot Leave-One-Out Cross-Validation errors. This plot is for relative
        errors only."""
        mpl.rcParams["figure.constrained_layout.use"] = False

        n_zbincombo = len(zbincombo_list)
        dv_length = len(MSE_LIST[0]) // n_zbincombo # dv length for each zbincombo
        n_rows = (n_zbincombo + n_cols - 1) // n_cols

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
                if J is not None:
                    ax.axvline(x=J, color="grey", lw=1.0, ls="solid", alpha=0.5)
                    if i >= (n_rows - 1) * n_cols: # last row
                        ax.text(x=J / 2, y=0.935, s=r"$s_1$",
                                color="grey", ha="center", fontsize=8)
                        ax.text(x=J * (J + 1) / 2, y=0.935, s=r"$s_2$",
                                color="grey", ha="center", fontsize=8)

                ax.text(x=J**2 - 1.3, y=1.035, s=zbincombo2label(zbincombo_list[i]),
                        family="DejaVu Sans Mono", ha="right", fontsize=6)
                ax.set_ylim(bottom=0.95, top=1.05)
                ax.set_yticks([0.97, 1.0, 1.03])
                ax.set_yticklabels(["0.97", "1.00", "1.03"], fontsize=7)
                ax.tick_params(axis="y", which="major", pad=2.0, length=3.0, right=False)
                ax.set_xlim(left=0, right=J**2 - 1)
                ax.set_xticks([])
            else:
                ax.axis("off")

        fig.text(
            x=0.02, y=0.5, s=r"$\mathbf{d}_\mathrm{LOOCV} / \mathbf{d}_\mathrm{True}$",
            va="center", rotation="vertical", fontsize=8,
        )

    @staticmethod
    def predictions_vs_param():
        """Plot emulator predictions across cosmological parameter values."""
        pass



class STDataViz:
    @staticmethod
    def STDV_vs_param():
        pass

    @staticmethod
    def COV():
        pass