import torch
import numpy as np
from typing import Callable

from scatterlens.emulator import Emulator

class Model:
    def __init__(
            self,
            emulator: Emulator,
            systematics: list[str] | None=None,
            IA_kwargs: dict | None=None,
            baryon_kwargs: dict | None=None,
    ):
        """Forward model the statistics with systematic effects.

        Args:
            emulator:
        """
        self.emulator = emulator

        if systematics:
            if not any(syst in ("IA", "baryon") for syst in systematics):
                raise ValueError
        self.systematics = systematics

        if IA_kwargs:
            assert "savepath" in IA_kwargs
            self.IA_kwargs = IA_kwargs
            self.IA_poly_coefs = np.array(
                torch.load(IA_kwargs["savepath"], weights_only=True)
            )
            self.IA_fid = self._get_IA_fid()

        if baryon_kwargs:
            assert "savepath" in baryon_kwargs
            self.baryon_kwargs = baryon_kwargs
            self.baryon_poly_coefs = np.array(
                torch.load(baryon_kwargs["savepath"], weights_only=True)
            )
            self.baryon_fid = self._get_baryon_fid()


    def predict(self, params) -> np.ndarray:
        dv = self.emulator.predict(params[:4])
        if self.systematics:
            for syst, param in zip(self.systematics, params[4:]):
                match syst:
                    case "IA":
                        dv *= self.IA(param)
                    case "baryon":
                        dv *= self.baryon(param)
                    case _:
                        pass
        return dv



    def IA(self, A_IA) -> np.ndarray:
        """Intrinsic alignment model.

        Args:
            A_IA: Amplitude of the intrinsic alignment signal.
        """
        degree = self.IA_poly_coefs.shape[0] - 1
        X = np.vander(x=[A_IA], N=degree + 1, increasing=False)
        IA_scaling = (X @ self.IA_poly_coefs).squeeze(0) / self.IA_fid
        return IA_scaling

    def baryon(self, b_bary) -> np.ndarray:
        """Baryonic feedback model.

        Args:
            b_bary: Baryonic feedback parameter.
        """
        degree = self.baryon_poly_coefs.shape[0] - 1
        X = np.vander(x=[b_bary], N=degree + 1, increasing=False)
        baryon_scaling = (X @ self.baryon_poly_coefs).squeeze(0) / self.baryon_fid
        return baryon_scaling

    def _get_IA_fid(self):
        A_IA = 0.
        degree = self.IA_poly_coefs.shape[0] - 1
        X = np.vander(x=[A_IA], N=degree + 1, increasing=False)
        IA_fid = (X @ self.IA_poly_coefs).squeeze(0)
        return IA_fid

    def _get_baryon_fid(self):
        b_bary = 0.
        degree = self.baryon_poly_coefs.shape[0] - 1
        X = np.vander(x=[b_bary], N=degree + 1, increasing=False)
        baryon_fid = (X @ self.baryon_poly_coefs).squeeze(0)
        return baryon_fid


    @staticmethod
    def _load_poly_coefs(kwargs) -> torch.Tensor:
        assert "savepath" in kwargs
        return torch.load(kwargs["savepath"], weights_only=True)


    @staticmethod
    def mbias_correction(
            mbias_uncertainties_zbin,
            mean_mbias_zbin,
            zbin_combos,
            galaxy_weighting_zbin=None,
    ):
        """
        Perform a multiplicative bias correction to the scattering coefficients.

        Args:
            mbias_uncertainties_zbin: Nusance parameter for m-bias uncertainty
                in each redshift bin.
            mean_mbias_zbin: The mean m-bias in each redshift bin.
            zbin_combos:
            galaxy_weighting_zbin:
        """
        pass