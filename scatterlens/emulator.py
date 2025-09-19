import torch
import numpy as np
from copy import deepcopy
from typing import override
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import LeaveOneOut

# Part of this code is copied from Lars.
# Source: https://github.com/ljabbo/kids-persistent-homology


class Emulator:
    def __init__(
            self,
            training_set: dict[str, torch.Tensor],
            input_scaler=None,
            target_scaler=None,
            regressor_type=GaussianProcessRegressor,
            **regressor_args,
    ):
        assert "input" in training_set.keys()
        assert "target" in training_set.keys()

        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

        self.regressor_type = regressor_type
        self.regressor_args = regressor_args
        self.regressor = regressor_type(**regressor_args)
        self.is_fitted = False

        self._training_set = {
            "input": np.array(training_set["input"]),
            "target": np.array(training_set["target"]),
        }

        self.training_set = deepcopy(self._training_set)

        for scaler, key in zip(
                [self.input_scaler, self.target_scaler], ["input", "target"]
        ):
            if scaler is not None:
                self.training_set[key] = scaler.fit_transform(self._training_set[key])

        self.n_samples = self.training_set["input"].shape[0]
        self.n_parameters = self.training_set["input"].shape[1]
        self.n_features = self.training_set["target"].shape[1]


    def fit(self):
        self.regressor.fit(self.training_set["input"], self.training_set["target"])
        self.is_fitted = True

    def predict(self, X):
        return self._predict(self.regressor, X)

    def validation(
            self,
            norm_by_true: bool=True,
            zero_is_perfect: bool=False,
    ) -> list[np.ndarray]:
        """Perform leave-one-out cross-validation on the training set.

        Args:
            norm_by_true: If True, returns residual error normalised by true
                value. If False, returns absolute residual error.
            zero_is_perfect: If True, zero represents perfect prediction. If
                False, one represents perfect prediction.
        """
        loo = LeaveOneOut()
        deviation_list = []

        for i, (train_index, test_index) in enumerate(loo.split(self.training_set["input"])):
            temp_regressor = self.regressor_type(**self.regressor_args)
            temp_regressor.fit(
                self.training_set["input"][train_index],
                self.training_set["target"][train_index],
            )
            pred = self._predict(temp_regressor, X=self._training_set["input"][test_index])[0]
            true = self._training_set["target"][test_index][0]
            deviation = pred - true

            if norm_by_true:
                deviation /= true
                if not zero_is_perfect:
                    deviation += 1.0

            deviation_list.append(deviation)

        return deviation_list

    def _predict(self, regressor, X):
        if self.input_scaler is not None:
            X = self.input_scaler.transform(X)

        Y = regressor.predict(X)
        if self.target_scaler is not None:
            Y = self.target_scaler.inverse_transform(Y)

        return Y


class S1S2Emulator(Emulator):
    def __init__(
            self,
            training_set: dict[str, torch.Tensor],
            J: int,
            input_scaler=None,
            target_scaler=None,
            per_zbincombo_emulator: bool=False,
            regressor_type=GaussianProcessRegressor,
            **regressor_args,
    ):
        """Construct separate S1, S2 emulators.

        Args:
            training_set:
            J: Number of scattering scales. This is used to divide training
                target data vector into S1, S2 components.
            per_zbincombo_emulator: If True, trains separate S1, S2 emulators
                for each redshift bin combination. Defaults to False.
            regressor_type:
            **regressor_args:
        """
        super().__init__(
            training_set, input_scaler, target_scaler, regressor_type, **regressor_args
        )

        self.J = J
        self.n_S1 = J
        self.n_S2 = J * (J + 1) // 2
        self.per_zbincombo_emulator = per_zbincombo_emulator

        if self.n_features % (self.n_S1 + self.n_S2) == 0:
            self.n_zbincombo = self.n_features // (self.n_S1 + self.n_S2)
        else:
            raise ValueError("Invalid J value provided.")

        self.regressors = self._create_regressors()

    @override
    def fit(self):
        self.regressors = self._fit(self.regressors, self.training_set)

    @override
    def predict(self, X):
        return self._predict(regressors=self.regressors, X=X)

    @override
    def validation(
            self,
            norm_by_true: bool=True,
            zero_is_perfect: bool=False,
    ) -> list[np.ndarray]:
        loo = LeaveOneOut()

        deviation_list = []

        for i, (train_index, test_index) in enumerate(
                loo.split(self.training_set["input"])):
            temp_regressors = self._create_regressors()
            temp_training_set = {
                "input": self.training_set["input"][train_index],
                "target": self.training_set["target"][train_index],
            }
            temp_regressors = self._fit(temp_regressors, temp_training_set)
            pred = self._predict(temp_regressors, X=self._training_set["input"][test_index])[0]
            true = self._training_set["target"][test_index][0]
            deviation = pred - true

            if norm_by_true:
                deviation /= true
                if not zero_is_perfect:
                    deviation += 1.0

            deviation_list.append(deviation)

        return deviation_list


    def _create_regressors(self) -> list[BaseEstimator] | list[list[BaseEstimator]]:
        """Instantiate regressors based on current configuration."""
        if self.per_zbincombo_emulator:
            regressors = [
                [
                    self.regressor_type(**self.regressor_args)
                    for combo in range(self.n_zbincombo)
                ]
                for _ in range(2)
            ]
        else:
            regressors = [
                self.regressor_type(**self.regressor_args) for _ in range(2)
            ]
        return regressors


    def _fit(self, regressors, training_set):
        # Reshape the target feature vector into size
        # (n_samples, n_zbincombo, n_S1 + n_S2)
        target_array = training_set["target"].reshape(
            -1, self.n_zbincombo, self.n_S1 + self.n_S2
        )
        n_samples = target_array.shape[0]
        if self.per_zbincombo_emulator:
            trained_regressors = [
                [
                    regressor.fit(
                        training_set["input"], target_array[:, i, :self.n_S1]
                    ) for i, regressor in enumerate(regressors[0])
                ],
                [
                    regressor.fit(
                        training_set["input"], target_array[:, i, self.n_S1:]
                    ) for i, regressor in enumerate(regressors[1])
                ],
            ]
        else:
            S1S2 = [
                target_array[:, :, :self.n_S1].reshape(n_samples, -1),
                target_array[:, :, self.n_S1:].reshape(n_samples, -1),
            ]
            trained_regressors = [
                regressor.fit(training_set["input"], target)
                for target, regressor in zip(S1S2, regressors)
            ]
        return trained_regressors


    def _predict(self, regressors, X):
        # FIXME only works for one set of input parameters
        if self.input_scaler is not None:
            X = self.input_scaler.transform(X)

        if self.per_zbincombo_emulator:
            S1 = np.vstack([
                regressor.predict(X)[0] for regressor in regressors[0]
            ]).reshape(len(X), self.n_zbincombo, self.n_S1)
            S2 = np.vstack([
                regressor.predict(X)[0] for regressor in regressors[1]
            ]).reshape(len(X), self.n_zbincombo, self.n_S2)
        else:
            S1 = regressors[0].predict(X).reshape(len(X), self.n_zbincombo, self.n_S1)
            S2 = regressors[1].predict(X).reshape(len(X), self.n_zbincombo, self.n_S2)

        Y = np.concatenate((S1, S2), axis=-1).reshape(len(X), self.n_features)

        if self.target_scaler is not None:
            Y = self.target_scaler.inverse_transform(Y)

        return Y


class PerZbincomboEmulator(Emulator):
    def __init__(
            self,
            training_set: dict[str, torch.Tensor],
            J: int,
            input_scaler=None,
            target_scaler=None,
            regressor_type=GaussianProcessRegressor,
            **regressor_args,
    ):
        super().__init__(
            training_set, input_scaler, target_scaler, regressor_type, **regressor_args,
        )

        self.J = J
        self.n_S1 = J
        self.n_S2 = J * (J + 1) // 2

        if self.n_features % (self.n_S1 + self.n_S2) == 0:
            self.n_zbincombo = self.n_features // (self.n_S1 + self.n_S2)
        else:
            raise ValueError("Invalid J value provided.")

        self.regressors = [
            regressor_type(**regressor_args) for _ in range(self.n_zbincombo)
        ]

    @override
    def fit(self):
        self._fit(self.regressors, self.training_set)

    @override
    def predict(self, X):
        return self._predict(self.regressors, X)

    @override
    def validation(
            self,
            norm_by_true: bool=True,
            zero_is_perfect: bool=False,
    ) -> list[np.ndarray]:
        loo = LeaveOneOut()

        deviation_list = []

        for i, (train_index, test_index) in enumerate(
                loo.split(self.training_set["input"])):
            temp_regressors = [
                self.regressor_type(**self.regressor_args)
                for _ in range(self.n_zbincombo)
            ]
            temp_training_set = {
                "input": self.training_set["input"][train_index],
                "target": self.training_set["target"][train_index],
            }
            temp_regressors = self._fit(temp_regressors, temp_training_set)
            pred = self._predict(temp_regressors, X=self._training_set["input"][test_index])[0]
            true = self._training_set["target"][test_index][0]
            deviation = pred - true

            if norm_by_true:
                deviation /= true
                if not zero_is_perfect:
                    deviation += 1.0

            deviation_list.append(deviation)

        return deviation_list




    def _fit(self, regressors, training_set):
        target_array = training_set["target"].reshape(
            -1, self.n_zbincombo, self.n_S1 + self.n_S2
        )
        trained_regressors = [
            regressor.fit(
                training_set["input"], target_array[:, i, :]
            ) for i, regressor in enumerate(regressors)
        ]
        return trained_regressors

    def _predict(self, regressors, X):
        if self.input_scaler is not None:
            X = self.input_scaler.transform(X)

        Y = np.hstack([
            regressor.predict(X)[0] for regressor in regressors
        ])[None, :]

        if self.target_scaler is not None:
            Y = self.target_scaler.inverse_transform(Y)
        return Y




class PerFeatureEmulator(Emulator):
    def __init__(
            self,
            training_set: dict[str, torch.Tensor],
            input_scaler=None,
            target_scaler=None,
            regressor_type=GaussianProcessRegressor,
            **regressor_args,
    ):
        super().__init__(
            training_set, input_scaler, target_scaler, regressor_type, **regressor_args
        )

        self.regressors = [
            regressor_type(**regressor_args) for _ in range(self.n_features)
        ]

    @override
    def fit(self):
        self.regressors = self._fit(self.regressors, self.training_set)
        self.is_fitted = True

    @staticmethod
    def _fit(regressors, training_set):
        trained_regressors = [
            regressor.fit(
                training_set["input"], training_set["target"][:, i]
            )
            for i, regressor in enumerate(regressors)
        ]
        return trained_regressors

    @override
    def predict(self, X):
        return self._predict(self.regressors, X)

    @override
    def _predict(self, regressors, X):
        # FIXME only works for one set of input parameters
        if self.input_scaler is not None:
            X = self.input_scaler.transform(X)

        Y = np.array([[
            regressor.predict(X)[0] for regressor in regressors
        ]])
        if self.target_scaler is not None:
            Y = self.target_scaler.inverse_transform(Y)

        return Y

    @override
    def validation(
            self,
            norm_by_true: bool=True,
            zero_is_perfect: bool=False,
    ) -> list[np.ndarray]:
        loo = LeaveOneOut()

        deviation_list = []

        for i, (train_index, test_index) in enumerate(
                loo.split(self.training_set["input"])):
            temp_regressors = [
                self.regressor_type(**self.regressor_args)
                for _ in range(self.n_features)
            ]
            temp_training_set = {
                "input": self.training_set["input"][train_index],
                "target": self.training_set["target"][train_index],
            }
            temp_regressors = self._fit(temp_regressors, temp_training_set)
            pred = self._predict(temp_regressors, X=self._training_set["input"][test_index])[0]
            true = self._training_set["target"][test_index][0]
            deviation = pred - true

            if norm_by_true:
                deviation /= true
                if not zero_is_perfect:
                    deviation += 1.0

            deviation_list.append(deviation)

        return deviation_list


def get_emulation_uncertainty_cov_diag(
        deviation_list: list[np.ndarray],
        exclude_indices: list[int] | None=None,
) -> np.ndarray:
    """Estimate diagonal of emulation uncertainty covariance matrix.

    Args:
        deviation_list: List of deviation (pred - true) arrays from LOOCV.
        exclude_indices: List of indices to exclude from covariance
            estimation. This can be used to exclude outliers.
    """
    if exclude_indices is not None:
        deviation_list = np.delete(deviation_list, exclude_indices, axis=0)

    cov_diag = np.eye(len(deviation_list[0])) * np.mean(np.array(deviation_list)**2, axis=0)
    return cov_diag