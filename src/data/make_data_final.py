import math
from itertools import product
from random import shuffle
from typing import Any, Iterator, Union

import numpy as np
import pandas as pd
from loguru import logger

from . import transformers, utils

LEN_TGT = 16
MAX_TRAINING_DATE = pd.Timestamp("2017-08-15")
ADF_STATIONARY_P = 0.05

cat_features = [
    "store_nbr",
    # 'date_type_local', 'date_name_local',
    # 'locale_regional', 'date_type_regional', 'locale_name_regional',
    # 'date_type_national', 'date_name_national'
]

COLS_EXOGENOUS = [
    "month_sin",
    "month_cos",
    "dayinyear_sin",
    "dayinyear_cos",
    "sales_lag",
    #   "onpromotion",
    # "dcoilwtico",
    *cat_features,
]
COLS_ENDOGENOUS = "sales_train"


def gen_segments(s: pd.Series, shift: int) -> Iterator[tuple[Any, Any]]:
    """Generate data segment (no zero within data segment)

    Args:
        s (pd.Series): _description_

    Yields:
        Iterator[pd.Series]: _description_
    """
    idx_start_nonzero_part = idx_last_nonzero = None
    is_zero_partition = True

    for i, v in s.items():
        if is_zero_partition is True:
            if v == 0:
                continue

            idx_start_nonzero_part = idx_last_nonzero = i
            is_zero_partition = False
        else:
            if v != 0:
                idx_last_nonzero = i

                if i == s.index.max():
                    yield idx_start_nonzero_part, i

                continue

            if (i - idx_last_nonzero).days >= math.floor(shift * 0.25):  # type: ignore
                yield idx_start_nonzero_part, idx_last_nonzero

                is_zero_partition = True
                idx_last_nonzero = idx_start_nonzero_part = None

            elif i == s.index.max():
                yield idx_start_nonzero_part, i


class EcuardoSales:
    def __init__(
        self,
        paths: dict,
        diff_order: int = 0,
        low_q: float = 0.1,
        up_q: float = 0.9,
        is_rolling_cv: bool = False,
    ) -> None:
        self._diff_order = diff_order
        self._rolling_cv = is_rolling_cv
        self._low_q, self._up_q = low_q, up_q

        self.meta = utils.Meta(paths)

        ## dataset
        self.df_train_raw = pd.read_csv(paths["processed"]["train"], parse_dates=["date"], index_col="id")
        self.df_val_raw = pd.read_csv(paths["processed"]["val"], parse_dates=["date"], index_col="id")
        self.df_test_raw = pd.read_csv(paths["processed"]["test"], parse_dates=["date"], index_col="id")

        self.df_train_raw = self._transform_exogeneous(self.df_train_raw)
        self.df_val_raw = self._transform_exogeneous(self.df_val_raw)
        self.df_test_raw = self._transform_exogeneous(self.df_test_raw)

    def _transform_exogeneous(self, df: pd.DataFrame, col_name: str = "date"):
        # Transform date
        df.loc[:, "month_sin"] = transformers.f_transform_sin_month.fit_transform(pd.DatetimeIndex(df[col_name]).month)
        df.loc[:, "month_cos"] = transformers.f_transform_cos_month.fit_transform(pd.DatetimeIndex(df[col_name]).month)
        df.loc[:, "dayinyear_sin"] = transformers.f_transform_sin_dateinyear.fit_transform(
            pd.DatetimeIndex(df[col_name]).day
        )
        df.loc[:, "dayinyear_cos"] = transformers.f_transform_cos_dateinyear.fit_transform(
            pd.DatetimeIndex(df[col_name]).day
        )

        # Transform oil
        assert "dcoilwtico" in df

        df["dcoilwtico"] = transformers.f_transform_oil.fit_transform(df["dcoilwtico"])

        # Transform category
        for c in cat_features:
            if c not in df:
                logger.error(f"col '{c}' not in dataframe")
                continue

            df[c] = df[c].astype("category")

        return df

    def gen_Xy_trainval(self, df: Union[pd.DataFrame, None] = None, deterministic: bool = False) -> Iterator[tuple]:
        if df is None:
            df = pd.concat((self.df_train_raw, self.df_val_raw)).sort_values(by="date")
        else:
            df = self._transform_exogeneous(df)

        families = df["family"].unique().tolist()
        stores = df["store_nbr"].unique().tolist()
        if not deterministic:
            shuffle(stores)
            shuffle(families)

        for store, family in product(stores, families):
            if self.meta.check_zero_pair(store, family):
                continue

            # Declare transformers
            shift = self.meta.get_shift(store, family)
            if shift is None:
                shift = LEN_TGT
            p = self.meta.get_adf_test(store, family)
            if p is None:
                continue
            has_shift = p > ADF_STATIONARY_P
            pipe_sales = transformers.make_pipeline_sale(has_shift, low_q=self._low_q, up_q=self._up_q)

            s = df[(df["family"] == family) & (df["store_nbr"] == store)].set_index("date")

            for start, end in gen_segments(s["sales"], shift):
                if (end - start).days <= shift + LEN_TGT:
                    continue

                d = s[(s.index >= start) & (s.index <= end)]

                # Add lagged value
                d.loc[:, "sales_lag"] = d["sales"].shift(shift)
                d = d[~d["sales_lag"].isna()]

                # Start iterating over segment
                start_train = d.index.min()
                end_train = start_train + pd.Timedelta(days=min((d.index.max() - d.index.min()).days - LEN_TGT, shift))
                end_val = end_train + pd.Timedelta(days=LEN_TGT)

                while start_train != end_train and end_train != end_val:
                    df_train = d[(d.index >= start_train) & (d.index <= end_train)].copy()
                    df_val = d[(d.index >= end_train) & (d.index <= end_val)].copy()

                    # Transform sales_lagged
                    df_train["sales_lag"].iloc[1:] = pipe_sales.fit_transform(df_train["sales_lag"]).squeeze()[1:]
                    df_val["sales_lag"].iloc[1:] = pipe_sales.fit_transform(df_val["sales_lag"]).squeeze()[1:]

                    # Transform sales
                    df_train["sales_train"] = np.nan
                    df_train["sales_train"].iloc[1:] = pipe_sales.fit_transform(df_train["sales"]).squeeze()[1:]
                    df_train = df_train[~df_train["sales_train"].isna()]

                    Xtrain, ytrain = df_train[COLS_EXOGENOUS], df_train[COLS_ENDOGENOUS]
                    Xval, yval = df_val[COLS_EXOGENOUS], df_val["sales"]

                    yield store, family, Xtrain, ytrain, Xval, yval

                    start_train = start_train + pd.Timedelta(days=LEN_TGT)
                    end_train = end_val
                    end_val = min(end_val + pd.Timedelta(days=LEN_TGT), end)

    def gen_Xy_train_entire(self, df: Union[pd.DataFrame, None] = None) -> Iterator[tuple]:
        if df is None:
            df = pd.concat((self.df_train_raw, self.df_val_raw)).sort_values(by="date")
        else:
            df = self._transform_exogeneous(df)

        families = df["family"].unique().tolist()
        stores = df["store_nbr"].unique().tolist()

        for store, family in product(stores, families):
            if self.meta.check_zero_pair(store, family):
                continue

            # Declare transformers
            shift = self.meta.get_shift(store, family)
            if shift is None:
                shift = LEN_TGT
            p = self.meta.get_adf_test(store, family)
            if p is None:
                continue
            has_shift = p > ADF_STATIONARY_P
            pipe_sales = transformers.make_pipeline_sale(has_shift, low_q=self._low_q, up_q=self._up_q)

            s = df[(df["family"] == family) & (df["store_nbr"] == store)].set_index("date")

            for start, end in gen_segments(s["sales"], shift):
                d = s[(s.index >= start) & (s.index <= end)]

                # Add lagged value
                d.loc[:, "sales_lag"] = d["sales"].shift(shift).copy()
                d = d[~d["sales_lag"].isna()]

                if len(d) <= 1:
                    continue

                # Transform sales_lagged
                d["sales_lag"].iloc[1:] = pipe_sales.fit_transform(d["sales_lag"]).squeeze()[1:]

                # Transform sales
                d["sales_train"] = np.nan
                d["sales_train"].iloc[1:] = pipe_sales.fit_transform(d["sales"]).squeeze()[1:]
                d = d[~d["sales_train"].isna()]

                Xtrain, ytrain = d[COLS_EXOGENOUS], d[COLS_ENDOGENOUS]

                yield store, family, Xtrain, ytrain

    def gen_Xy_test(self) -> Iterator[tuple]:
        df = pd.concat((self.df_train_raw, self.df_val_raw, self.df_test_raw)).sort_values(by="date")

        families = df["family"].unique().tolist()
        stores = df["store_nbr"].unique().tolist()

        for store, family in product(stores, families):
            if self.meta.check_zero_pair(store, family):
                continue

            # Declare transformers
            shift = self.meta.get_shift(store, family)
            if shift is None:
                shift = LEN_TGT
            p = self.meta.get_adf_test(store, family)
            if p is None:
                continue
            has_shift = p > ADF_STATIONARY_P
            pipe_sales = transformers.make_pipeline_sale(has_shift, low_q=self._low_q, up_q=self._up_q)

            d = df[(df["family"] == family) & (df["store_nbr"] == store)]

            # Add lagged value and transform
            d.loc[:, "sales_lag"] = d["sales"].shift(shift).copy()
            d = d[~d["sales_lag"].isna()]
            d["sales_lag"].iloc[1:] = pipe_sales.fit_transform(d["sales_lag"]).squeeze()[1:]

            tmp = d[~d["sales"].isna()].set_index("date")
            for start, end in gen_segments(tmp["sales"], shift):
                continue
            pipe_sales.fit(d[(d["date"] >= start) & (d["date"] <= end)]["sales"])

            if has_shift is True:
                d = d[d["date"] >= MAX_TRAINING_DATE]
                pipe_sales.steps[2][1].first_item = d[d["date"] == MAX_TRAINING_DATE]["sales"]
            else:
                d = d[d["date"] > MAX_TRAINING_DATE]
            Xtest = d[COLS_EXOGENOUS]

            yield store, family, Xtest, pipe_sales
