from scatterlens.visualisation.utils import *
from scatterlens.wavelets import Morlet2D, get_Gaussian_profile, get_freq_multipole_conversion_pair

def plot_Morlet_profile(
        js: list[int],
        pixel_length: float,
        freq_samples: np.ndarray,
        sigma_Gaussian: float | None=None,
        colors: list[str] | None=None,
        base_value_k0=3. / 4. * np.pi,
        base_value_sigma=0.8,
):
    """Plot the Morlet wavelet profiles in Fourier space.

    Args:
        js: List of j scales to plot.
        pixel_length: Length of a pixel in arcmin.
        freq_samples: Frequency samples in units of pixel^-1.
        sigma_Gaussian: Standard deviation of the Gaussian to compare with.
        colors: Colors for different j scales.
        base_value_k0: Base value for Morlet2D.k0(j).
        base_value_sigma: Base value for Morlet2D.sigma(j).
    """
    fig, ax = plt.subplots(figsize=cm2inch(onecol_wth, 6.))

    for j_index, j in enumerate(js):
        ax.plot(
            freq_samples,
            Morlet2D.get_profile(j, freq_samples, base_value_k0, base_value_sigma),
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


def plot_Morlet_wavelets(wavelets: torch.Tensor):
    """Plot the 2D Morlet wavelets in real space.

    Args:
        wavelets: Wavelets in Fourier space with shape (J, L, M, N). This can be
            generated using `Morlet2D.gen_filter_bank()` function and choose the
            key "psi" from the returned dictionary.
    """
    from numpy.fft import ifft2

    J, L, M, N = wavelets.shape

    fig, axs = plt.subplots(
        nrows=J - 1, ncols=L,
        figsize=cm2inch(median_wth, median_wth)
    )
    fig.set_figwidth(cm2inch(median_wth))

    for j in range(J):
        if j == 0:
            continue
        for l in range(L):
            f = wavelets[j, l]  # wavelet in Fourier space
            filter_c = np.fft.fftshift(ifft2(f))
            filter_r = filter_c.real
            filter_i = filter_c.imag

            ax = axs[j - 1, l]
            ax.imshow(
                filter_r,
                cmap="RdBu_r",
                vmin=-filter_r.max(),
                vmax=filter_r.max(),
            )
            ax.axis("off")
            ax.set_title(r"$j={}$, $l={}$".format(j, l), fontsize=7)

    return fig, axs