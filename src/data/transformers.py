from typing import Any, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.ndimage import shift
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

f_transform_oil = FunctionTransformer(lambda x: np.log1p(x))
f_transform_cat = FunctionTransformer(lambda x: x.astype("category"))
f_transform_sin_dateinyear = FunctionTransformer(lambda x: np.sin(x / 365 * 2 * np.pi))
f_transform_cos_dateinyear = FunctionTransformer(lambda x: np.cos(x / 365 * 2 * np.pi))
f_transform_sin_month = FunctionTransformer(lambda x: np.sin(x / 12 * 2 * np.pi))
f_transform_cos_month = FunctionTransformer(lambda x: np.cos(x / 12 * 2 * np.pi))


class CustomTransformerMixin(TransformerMixin):
    def inverse_transform(self, X: Any):
        return X


class CustomPipeline(Pipeline):
    def _transform_type(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.Series):
            Xout = X.to_numpy()[:, None]
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                Xout = X[:, None]
            elif X.ndim == 2:
                assert X.shape[1] == 1
                Xout = X
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        # logger.info(f"Xout: {Xout.shape}")

        return Xout

    def fit(self, X: Union[pd.Series, np.ndarray], y: Any = None, **fit_params):
        Xout = self._transform_type(X)
        return super().fit(Xout, y, **fit_params)

    def transform(self, X: Union[pd.Series, np.ndarray], **kwargs) -> ndarray:
        Xout = self._transform_type(X)

        return super().transform(Xout, **kwargs)

    def fit_transform(self, X: Union[pd.Series, np.ndarray], y: Any = None, **fit_params) -> ndarray:
        Xout = self._transform_type(X)

        return super().fit_transform(Xout, y, **fit_params)

    def inverse_transform(self, X):
        out = X
        for _, step in self.steps[::-1]:
            out = step.inverse_transform(out)

        return out


class TransformerClip(CustomTransformerMixin):
    def __init__(self, low_q: float = 0.1, up_q: float = 0.9) -> None:
        super().__init__()

        self._low_p, self._up_p = low_q, up_q

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X) -> np.ndarray:
        X_low = np.quantile(X, self._low_p)
        X_up = np.quantile(X, self._up_p)

        out = np.clip(X, a_min=X_low, a_max=X_up)

        return out


class TransformerLog(CustomTransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X) -> np.ndarray:
        out = np.log(X + 1)

        return out

    def inverse_transform(self, X: Any):
        out = np.exp(X) - 1

        return out


class TransformerShift(CustomTransformerMixin):
    def __init__(self, period: int = 1) -> None:
        self.first_item = None

        self._period = period

    def fit(self, X, *args, **kwargs):
        if isinstance(X, pd.Series):
            self.first_item = X.iloc[0]
        elif isinstance(X, np.ndarray):
            self.first_item = X[0]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        shifted = shift(X[:, 0], self._period, cval=np.nan)
        out = X[:, 0] - shifted

        return out[:, None]

    def inverse_transform(self, X: Union[pd.Series, np.ndarray]):
        if isinstance(X, pd.Series):
            X = X.to_numpy()

        assert self.first_item is not None
        X = np.insert(X, 0, self.first_item)

        out = X.cumsum()

        return out[:, None]


def make_pipeline_sale(has_shift: bool = True, low_q: float = 0.1, up_q: float = 0.9, shift_period: int = 1):
    steps = [
        ("clip", TransformerClip(low_q, up_q)),
        ("log", TransformerLog()),
        ("scale", MinMaxScaler(feature_range=(-1, 1))),
    ]

    if has_shift is True:
        steps.insert(2, ("shift", TransformerShift(period=shift_period)))

    return CustomPipeline(steps=steps)
