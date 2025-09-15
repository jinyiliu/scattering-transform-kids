import seaborn as sns
import arviz as az
from scipy.stats import gaussian_kde
from scatterlens.visualisation.utils import *

CONFIDENCE_LEVELS_1D = (0.682, 0.954)
CONFIDENCE_LEVELS_2D = (0.393, 0.864)


def plot_posterior_corner(
        samples: np.ndarray,
        contour_color: str="olive",
        param_ranges: list[tuple[float, float]] | None=None,
        param_labels: list[str] | None=None,
        param_ticks: list[tuple[float, ...]] | None=None,
        figsize: float=median_wth,
        histplot_bivariate: bool=True,
        histplot_diagonal: bool=False,
        truths: list[float] | None=None,
        savedir: str | None=None,
        fname: str="posterior_corner.pdf",
):
    """Plot corner plot of posterior distribution.

    Args:
        samples: Samples from the posterior distribution.
        contour_color: Color of the contour lines and fill.
        param_ranges: Parameter ranges for the posterior distribution.
        param_labels: Labels for the parameters. If None, will use the keys
            of param_ranges.
        param_ticks: Ticks for the parameters.
        figsize: Width of the figure in cm.
        histplot_bivariate: Whether to plot bivariate histograms.
        histplot_diagonal: Whether to plot histograms on the diagonal.
        truths: Optional truth values for the parameters.
        savedir:
        fname:
    """
    n_params = samples.shape[1]
    n_colors = len(CONFIDENCE_LEVELS_2D)
    colors = sns.light_palette(
        color=contour_color, n_colors=n_colors + 2)[-n_colors:]
    n_histbins = 50

    if param_ranges is None:
        param_ranges = np.vstack([samples.min(axis=0), samples.max(axis=0)]).T

    if param_ticks is None:
        param_ticks = [
            np.linspace(
                start=minv - (maxv - minv) / 8,
                stop=maxv + (maxv - minv) / 8,
                num=5,
            )[1:-1] for minv, maxv in param_ranges
        ]

    mpl.rcParams["figure.constrained_layout.use"] = False
    mpl.rcParams["xtick.top"] = False
    mpl.rcParams["ytick.right"] = False

    fig, axes = plt.subplots(
        figsize=cm2inch(figsize, figsize),
        ncols=n_params,
        nrows=n_params,
    )
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    axes_diag = np.diag(axes)
    axes_lower = np.tril(axes, k=-1)
    axes_upper = np.triu(axes, k=1)

    for ax in axes_upper.flatten():
        if isinstance(ax, plt.Axes):
            ax.axis("off")  # clear upper triangle axes

    for i in range(n_params):  # lower triangle
        for j in range(n_params):
            ax = axes_lower[i, j]
            if isinstance(ax, plt.Axes):
                if histplot_bivariate:
                    sns.histplot(
                        x=samples[:, j],
                        y=samples[:, i],
                        ax=ax,
                        bins=n_histbins,
                        binrange=(param_ranges[j], param_ranges[i]),
                        cmap=sns.light_palette(color="darkgrey", as_cmap=True),
                        zorder=-1,
                    )
                levels = [1 - cfl for cfl in CONFIDENCE_LEVELS_2D[::-1]]
                sns.kdeplot(
                    x=samples[:, j],
                    y=samples[:, i],
                    ax=ax,
                    levels=levels,
                    color=contour_color,
                    fill=True,  # use matplotlib.axes.Axes.contourf
                    colors=colors,
                    alpha=0.7,
                    extend="max",
                    zorder=0,
                )
                sns.kdeplot(
                    x=samples[:, j],
                    y=samples[:, i],
                    ax=ax,
                    levels=levels,
                    color=contour_color,
                    fill=False,  # use matplotlib.axes.Axes.contour
                    colors=colors,
                    zorder=0,
                )
                ax.set_xlim(param_ranges[j])
                ax.set_ylim(param_ranges[i])
                ax.set_xticks(param_ticks[j])
                ax.set_xticklabels([])
                ax.set_yticks(param_ticks[i])
                ax.set_yticklabels([])

                if truths is not None:
                    ax.scatter(truths[j], truths[i], color="red", marker=".",
                               zorder=10)

    MAX_HDI = get_marginal_MAX_HDI(samples)

    for i, ax in enumerate(axes_diag):  # diagonal
        samples_i = samples[:, i]
        if isinstance(ax, plt.Axes):
            if histplot_diagonal:
                sns.histplot(
                    samples_i,
                    ax=ax,
                    bins=n_histbins,
                    binrange=param_ranges[i],
                    element="step",
                    fill=True,
                    color="lightgrey",
                    stat="density",
                    zorder=-2,
                )
            sns.kdeplot(
                samples_i,
                ax=ax, color=contour_color,
                linewidth=1.6,
                zorder=0,
            )
            ax.margins(y=0.1)
            ax.set_ylim(bottom=0.0)
            MAX, (HDI_LEFT, HDI_RIGHT) = MAX_HDI[i]
            ax.fill_betweenx(
                y=[0, ax.get_ylim()[1]],
                x1=HDI_LEFT,
                x2=HDI_RIGHT,
                color="lightgrey",
                zorder=-1,
            )
            ax.set_title(
                label=param_labels[
                          i] + f" $={MAX:.2f}^{{+{HDI_RIGHT - MAX:.2f}}}_{{-{MAX - HDI_LEFT:.2f}}}$",
                fontsize=7,
            )
            ax.set_xlim(param_ranges[i])
            ax.set_xticks(param_ticks[i])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])

            if truths is not None:
                ax.axvline(truths[i], color="red", lw=0.6, zorder=10)

    for ax in axes.flatten():
        ax.set_xlabel("")  # clear x-axis labels
        ax.set_ylabel("")

    # Add x-tick labels and y-axis labels for the bottom row
    for i, ax in enumerate(axes[-1, :]):
        if param_labels is not None:
            ax.set_xlabel(param_labels[i])
        if param_ticks is not None:
            ax.set_xticklabels(
                [f"${tick:.2g}$" for tick in param_ticks[i]],
                fontsize=8,
            )

    # Add y-tick labels and y-axis labels for the leftmost column
    for i, ax in enumerate(axes[:, 0]):
        if i != 0:
            if param_labels is not None:
                ax.set_ylabel(param_labels[i])
            if param_ticks is not None:
                ax.set_yticklabels(
                    [f"${tick:.2g}$" for tick in param_ticks[i]],
                    fontsize=8,
                )

    if savedir:
        fig.savefig(
            os.path.join(savedir, fname),
            bbox_inches="tight",
        )


def get_marginal_MAX_HDI(samples: np.ndarray) -> list:
    """Calculate the maximum of the 1D marginal distributions with the
    highest density interval (HDI) for each parameter.

    Args:
        samples: Samples with shape (n_samples, n_params) or (n_samples,).
    """
    ret = []
    if samples.ndim == 2:
        samples = samples.T
    else:
        samples = samples[None, :]

    for samples_i in samples:
        KDE = gaussian_kde(samples_i)
        x = np.linspace(np.min(samples_i), np.max(samples_i), num=500)
        MAX = float(x[np.argmax(KDE(x))])

        IDATA = az.from_dict(
            posterior={"p": samples_i[None, :]},
        )
        HDI = az.hdi(IDATA, hdi_prob=CONFIDENCE_LEVELS_1D[0])["p"]
        ret.append([MAX, (float(HDI[0]), float(HDI[1]))])

    if samples.ndim == 2:
        return ret
    else:
        return ret[0]