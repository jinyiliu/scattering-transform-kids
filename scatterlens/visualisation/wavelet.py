from scatterlens.visualisation.utils import *
from scatterlens.wavelets import Morlet2D, get_Gaussian_profile, get_freq_multipole_conversion_pair

def plot_Morlet_profile(
        js: list[int],
        pixel_length: float,
        freq_samples: np.ndarray,
        sigma_Gaussian: float | None=None,
        colors: list[str] | None=None,
):
    """Plot the Morlet wavelet profiles in Fourier space.

    Args:
        js: List of j scales to plot.
        pixel_length: Length of a pixel in arcmin.
        freq_samples: Frequency samples in units of pixel^-1.
        sigma_Gaussian: Standard deviation of the Gaussian to compare with.
        colors: Colors for different j scales.
    """
    fig, ax = plt.subplots(figsize=cm2inch(onecol_wth, 6.))

    for j_index, j in enumerate(js):
        ax.plot(
            freq_samples,
            Morlet2D.get_profile(j, freq_samples),
            label=f"$j=$ {j}",
            color=colors[j_index] if colors else None,
            linewidth=0.5 if j==1 else 2.0,
        )

    gaussian_profile = get_Gaussian_profile(
        sigma=sigma_Gaussian,
        pixel_length=pixel_length,
        freq_samples=freq_samples,
    )
    ax.plot(freq_samples, gaussian_profile, color="black")

    ax.legend()
    ax.set_xlabel(r"$\mathrm{Frequency}$ [pixel$^{-1}$]")
    ax.set_ylabel("Profile")
    ax.set_xlim(left=0, right=0.24)
    ax.set_ylim(bottom=-0.07, top=1.05)
    ax.set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(top=False)

    second_ax = ax.secondary_xaxis(
        location="top",
        functions=get_freq_multipole_conversion_pair(pixel_length),
    )
    second_ax.set_xlabel(r"Multiple $\ell$")
    second_ax.set_xticks(
        ticks=[1000, 2000, 3000, 4000, 5000],
    )
    second_ax.minorticks_on()
    second_ax.tick_params(axis="x", pad=1.)

    return fig, ax