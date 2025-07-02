class CompoundScaler:
    def __init__(self, *scalers):
        """Applies multiple scalers to the dataset in a specified sequential order.

        Args:
            scalers: a squence of initiliased scalers.
        """
        if not scalers:
            raise ValueError("No scalers provided")

        for scaler in scalers:
            check_scaler(scaler)
        self.scalers = list(scalers)

    def fit(self, X):
        _ = self.fit_transform(X)

    def transform(self, X):
        for scaler in self.scalers:
            X = scaler.transform(X)
        return X

    def fit_transform(self, X):
        for scaler in self.scalers:
            X = scaler.fit_transform(X)
        return X

    def inverse_transform(self, X):
        for scaler in self.scalers[::-1]:
            X = scaler.inverse_transform(X)
        return X


def check_scaler(scaler):
    assert hasattr(scaler, "fit") and callable(scaler.fit)
    assert hasattr(scaler, "transform") and callable(scaler.transform)
    assert hasattr(scaler, "fit_transform") and callable(scaler.fit_transform)
    assert hasattr(scaler, "inverse_transform") and callable(scaler.inverse_transform)
