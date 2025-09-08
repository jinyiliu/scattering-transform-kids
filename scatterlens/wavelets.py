import os
import torch
import numpy as np
from scipy.fft import fft2
from numpy.typing import NDArray
from kymatio.scattering2d import filter_bank


class Wavelet2D(object):
    def __init__(self, M: int, N: int, J: int, L: int):
        self.M = M
        self.N = N
        self.J = J # number of scales
        self.L = L # number of angles

    def gen_filter_bank(
            self, dtype: torch.dtype, savedir: str | os.PathLike | None=None,
            overwrite=False):
        """Define the funtion in child class."""
        raise NotImplementedError


    @staticmethod
    def dilation(j: int) -> int:
        dilation_factor = 2 # binary dilation
        return dilation_factor ** j


class Morlet2D(Wavelet2D):
    def __init__(
            self, M: int, N: int, J: int, L: int,
            base_value_k0=3. / 4. * np.pi, base_value_sigma=0.8):
        super().__init__(M=M, N=N, J=J, L=L)
        self.base_value_k0 = base_value_k0
        self.base_value_sigma = base_value_sigma

    def morlet_2d(self, j: int, l: int) -> NDArray[np.complex64]:
        """Return the 2D profile of the Morlet wavelet in real space."""
        psi = filter_bank.morlet_2d(
            M=self.M,
            N=self.N,
            sigma=self.sigma(j, base_value=self.base_value_sigma),
            xi=self.k0(j, base_value=self.base_value_k0),
            theta=(int(self.L - self.L / 2 - 1) - l) / self.L * np.pi,
            slant=4. / self.L,
            offset=0.,
        )
        return psi

    def gabor_2d(self) -> NDArray[np.complex64]:
        phi = filter_bank.gabor_2d(
            M=self.M,
            N=self.N,
            sigma=self.sigma(self.J - 1, base_value=self.base_value_sigma),
            xi=0.,
            theta=0.,
            slant=1.,
            offset=0.,
        )
        return phi

    def gen_filter_bank(
            self, dtype: torch.dtype, savedir: str | os.PathLike | None=None,
            overwrite=False) -> dict[str, torch.Tensor]:
        """Generate filter bank in Fourier space."""
        if not dtype in (torch.float32, torch.float64):
            raise AssertionError("dtype must be torch.float32 or torch.float64")

        # Band-pass filters
        psi = torch.zeros(size=(self.J, self.L, self.M, self.N), dtype=dtype)

        for j in range(self.J):
            for l in range(self.L):
                psi_signal = self.morlet_2d(j, l)
                psi_signal_fourier = np.real(fft2(psi_signal))

                # The default precision in numpy is float64
                if dtype==torch.float32:
                    psi_signal_fourier = psi_signal_fourier.astype(np.float32)

                psi[j, l] = torch.from_numpy(psi_signal_fourier)

        # Low-pass filter
        phi_signal = self.gabor_2d()
        phi_signal_fourier =  np.real(fft2(phi_signal))
        if dtype==torch.float32:
            phi_signal_fourier = phi_signal_fourier.astype(np.float32)
        phi = torch.from_numpy(phi_signal_fourier)

        filters = {"psi": psi, "phi": phi}

        if savedir:
            fname = f"Morlet2Dfilters_M{self.M}N{self.N}J{self.J}L{self.L}_{str(dtype).split('.')[-1]}.pt"
            torch.save(filters, os.path.join(savedir, fname))

        return filters

    @staticmethod
    def get_profile(
            j: int,
            freq_samples: np.ndarray,
            base_value_k0=3. / 4. * np.pi,
            base_value_sigma=0.8,
    ) -> np.ndarray:
        """Get the Morlet profile in Fourier space.

        Args:
            j:
            freq_samples: Frequency samples in units of pixel^-1.
            base_value_k0: Base value for Morlet2D.k0(j).
            base_value_sigma: Base value for Morlet2D.sigma(j).
        """
        k_samples = freq_samples * 2 * np.pi
        sigma = Morlet2D.sigma(j, base_value=base_value_sigma)
        k0 = Morlet2D.k0(j, base_value=base_value_k0)
        beta = np.exp(- (k0 * sigma) ** 2 / 2)
        low_pass_window = beta * np.exp(- (k_samples * sigma) ** 2)
        profile = np.exp(- ((k_samples - k0) * sigma) ** 2 / 2) - low_pass_window
        return profile

    @staticmethod
    def sigma(j, base_value=0.8):
        """Sigma of the Gaussian envelope in pixels."""
        return base_value * Wavelet2D.dilation(j)

    @staticmethod
    def k0(j, base_value=3. / 4. * np.pi):
        """Central frequency of the Morlet wavelet in unit of pixel^-1."""
        return base_value / Wavelet2D.dilation(j)

    @staticmethod
    def j2scale(j, pixel_length, base_value_k0=3. / 4. * np.pi):
        """Convert `j` to angular scale with the same unit as `pixel_length`."""
        return pixel_length / Morlet2D.k0(j, base_value=base_value_k0) * 2


def get_Gaussian_profile(
        sigma: float,
        pixel_length: float,
        freq_samples: np.ndarray,
) -> np.ndarray:
    """Get the Gaussian profile in Fourier space.

    Args:
        sigma: Standard deviation of the Gaussian.
        pixel_length: Length of a pixel in the same unit as `sigma`.
        freq_samples: Frequency samples in units of pixel^-1.
    """
    sigma_in_pixel = sigma / pixel_length
    k_samples = freq_samples * 2 * np.pi
    return np.exp(- (k_samples * sigma_in_pixel)**2 / 2)


def get_freq_multipole_conversion_pair(pixel_length: float):
    """Get the conversion pair between multipole and frequency.

    Args:
        pixel_length: Length of a pixel in arcmin.
    """
    def multiple2freq_closure(multiple: float | np.ndarray) -> float | np.ndarray:
        return multiple2freq(multiple, pixel_length)

    def freq2multiple_closure(freq: float | np.ndarray) -> float | np.ndarray:
        return freq2multiple(freq, pixel_length)

    return freq2multiple_closure, multiple2freq_closure


def freq2multiple(freq: float | np.ndarray, pixel_length: float) -> float | np.ndarray:
    """Convert frequency to multiple.

    Args:
        freq: Frequency samples in units of pixel^-1.
        pixel_length: Length of a pixel in arcmin.
    """
    return freq * (2 * 60 * 180) / pixel_length


def multiple2freq(multiple: float | np.ndarray, pixel_length: float) -> float | np.ndarray:
    """Convert multiple to frequency.

    Args:
        multiple:
        pixel_length: Length of a pixel in arcmin.
    """
    return multiple * pixel_length / (2 * 60 * 180)