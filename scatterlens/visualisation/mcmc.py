import numpy as np
import seaborn as sns
from scatterlens.mcmc import compute_quantiles
from scatterlens.visualisation.utils import *

CONFIDENCE_LEVELS_1D = (0.682, 0.954)
CONFIDENCE_LEVELS_2D = (0.393, 0.864)

# TODO: support weights in the future

def plot_posterior_corner(
        samples: np.ndarray | list[np.ndarray],
        color: str | list[str]="京绿",
        same_contour_color: bool=False,
        fill: bool=False,
        contour_kwargs: dict | None=None,
        fill_kwargs: dict | None=None,
        levels: tuple[float]=CONFIDENCE_LEVELS_2D,
        quantiles: tuple[float]=(0.16, 0.5, 0.84),
        kde_bw_adjust: float=1.0,
        kde_gridsize: int=100,
        param_ranges: list[tuple[float, float]] | None=None,
        param_labels: list[str] | None=None,
        param_ticks: list[tuple[float, ...]] | None=None,
        figsize: tuple[float] | float=median_wth,
        truths: list[float] | None=None,
        plot_samples: bool=False,
        plot_samples_kwargs: dict | None=None,
        verbose: bool=False,
        savedir: str | None=None,
        fname: str="posterior_corner.pdf",
):
    """Plot corner plot of posterior distribution.

    Notes:
        This function uses seaborn.kdeplot to plot the contour of the posterior
        distribution. The confidence levels for the contour are defined in
        CONFIDENCE_LEVELS_2D.

    Args:
        samples: Samples from the posterior distribution.
        color: Base color of the contour lines and fill.
        same_contour_color: Whether to use the same color for all confidence levels.
        fill: Whether to fill the contour.
        contour_kwargs: Additional keyword arguments for the sns.kdeplot for the contour.
        fill_kwargs: Additional keyword arguments for the sns.kdeplot for the filled contour.
        levels: Confidence levels for the contour.
        quantiles: Quantiles to display on the diagonal plots.
        kde_bw_adjust: Bandwidth adjustment for the kernel density estimation.
            Higher values lead to smoother contours.
        kde_gridsize: Gridsize for the kernel density estimation. Higher values
            lead to smoother contours but longer computation time.
        param_ranges: Parameter ranges for the posterior distribution.
        param_labels: Labels for the parameters. If None, will use the keys
            of param_ranges.
        param_ticks: Ticks for the parameters.
        figsize: Tuple of figure size in cm. If a single float is supplied, it
            will be used for both width and height.
        truths: Optional truth values for the parameters.
        plot_samples: Whether to plot samples as hexbin in the lower triangle.
        plot_samples_kwargs: Additional keyword arguments for the hexbin plot
            of the samples.
        verbose:
        savedir:
        fname:
    """
    # TODO: verbose option to print out the quantiles
    if samples.ndim == 1:
        raise NotImplementedError(
            "1D samples are not supported."
            "Please provide 2D samples with shape (n_samples, n_params)."
        )

    if not all(0. < level < 1. for level in levels):
        raise ValueError("Confidence levels must be between 0 and 1")

    if not all(0. < quantile < 1. for quantile in quantiles):
        raise ValueError("Quantiles must be between 0 and 1")

    n_params = samples.shape[1]
    n_colors = len(CONFIDENCE_LEVELS_2D)
    colors = sns.light_palette(
        color=color, n_colors=n_colors + 2)[-n_colors:]

    fill_kwargs = fill_kwargs or {}
    contour_kwargs = contour_kwargs or {}
    plot_samples_kwargs = plot_samples_kwargs or {}

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

    if isinstance(figsize, float):
        figsize = (figsize,) * 2

    fig, axes = plt.subplots(
        figsize=cm2inch(*figsize),
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
                if plot_samples:
                    if param_ranges is not None:
                        extent = (*param_ranges[j], *param_ranges[i])
                    else:
                        extent = None
                    ax.hexbin(
                        x=samples[:, j],
                        y=samples[:, i],
                        extent=extent,
                        zorder=0,
                        linewidths=0.05,
                        **plot_samples_kwargs,
                    )
                levels = [1 - cfl for cfl in CONFIDENCE_LEVELS_2D[::-1]]
                if verbose:
                    print(
                        "KDE plotting for parameters",
                        param_labels[j] if param_labels else f"param_{j}",
                        "and",
                        param_labels[i] if param_labels else f"param_{i}",
                    )
                sns.kdeplot( # posterior contour plot
                    x=samples[:, j],
                    y=samples[:, i],
                    ax=ax,
                    levels=levels,
                    color=color,
                    bw_adjust=kde_bw_adjust,
                    gridsize=kde_gridsize,
                    fill=False,
                    colors=[color] * len(levels) if same_contour_color else colors,
                    zorder=2,
                    **contour_kwargs,
                )
                if fill:
                    sns.kdeplot(
                        x=samples[:, j],
                        y=samples[:, i],
                        ax=ax,
                        levels=levels,
                        color=color,
                        bw_adjust=kde_bw_adjust,
                        gridsize=kde_gridsize,
                        fill=True, # use matplotlib.axes.Axes.contourf
                        colors=colors,
                        alpha=0.7,
                        extend="max",
                        zorder=1,
                        **fill_kwargs,
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

    qvalues = compute_quantiles(samples, quantiles=quantiles)

    for i, ax in enumerate(axes_diag):  # diagonal
        samples_i = samples[:, i]
        if isinstance(ax, plt.Axes):
            sns.kdeplot(
                samples_i,
                ax=ax,
                bw_adjust=kde_bw_adjust,
                gridsize=kde_gridsize,
                color=color,
                linewidth=1.6,
                zorder=0,
            )
            ax.margins(y=0.1)
            ax.set_ylim(bottom=0.0)
            qlow, qmid, qhigh = qvalues[i]
            if verbose:
                print(
                    param_labels[i] if param_labels else f"param_{i}",
                    f"= {qmid:.3f} {{+{qmid - qlow:.3f}}} {{-{qhigh - qmid:.3f}}}",
                )
            # TODO: make this an optional feature
            ax.fill_betweenx(
                y=[0, ax.get_ylim()[1]],
                x1=qlow,
                x2=qhigh,
                color="lightgrey",
                zorder=-1,
            )
            # TODO: make this an optional feature
            ax.set_title(
                label=param_labels[i] + f" $={qmid:.2f}^{{+{qmid - qlow:.2f}}}_{{-{qhigh - qmid:.2f}}}$",
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

    return fig, axes

