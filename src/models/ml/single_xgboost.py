import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn import preprocessing
from sklearn.metrics import mean_squared_log_error
from tqdm.contrib.itertools import product

from src import utils


def sin_transformer(period):
    return preprocessing.FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return preprocessing.FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def get_specific_df(df: pd.DataFrame, store: int, family: str):
    df = df[(df["store_nbr"] == store) & (df["family"] == family)]
    df = df.sort_values(by="date", ascending=True).set_index(keys=["date"])

    return df


def get_X_y(df):
    df = df[df["sales"] != 0]

    X = df[["dayinyear_sin", "dayinyear_cos", "month_sin", "month_cos"]]
    y = df["sales"]

    return X, y


def add_datetime_encode(df):
    dt = pd.DatetimeIndex(df["date"])

    df["month_sin"] = sin_transformer(12).fit_transform(dt.month)
    df["month_cos"] = cos_transformer(12).fit_transform(dt.month)
    df["dayinmonth_sin"] = sin_transformer(31).fit_transform(dt.day)
    df["dayinmonth_cos"] = cos_transformer(31).fit_transform(dt.day)
    df["dayinyear_sin"] = sin_transformer(365).fit_transform(dt.day)
    df["dayinyear_cos"] = cos_transformer(365).fit_transform(dt.day)

    return df


def run_single_xgboost():
    paths = utils.load_conf()["PATH"]

    path_train = paths["processed"]["train"]
    path_val = paths["processed"]["val"]
    path_raw_test = paths["raw"]["test"]

    df_train = pd.read_csv(path_train, parse_dates=["date"])
    df_val = pd.read_csv(path_val, parse_dates=["date"])
    df_test = pd.read_csv(path_raw_test, parse_dates=["date"], index_col="id")

    df_train = add_datetime_encode(df_train)
    df_val = add_datetime_encode(df_val)
    df_test = add_datetime_encode(df_test)

    pred, tgt = [], []
    special = set()
    dict_regressors = {}

    df_train = df_train[df_train["sales"] != 0]

    for store, family in product(df_train["store_nbr"].unique(), df_train["family"].unique()):
        key = f"{store}-{family}"

        df_train_ = get_specific_df(df_train, store, family)
        df_val_ = get_specific_df(df_val, store, family)

        Xtrain, ytrain = get_X_y(df_train_)
        if len(Xtrain) == 0:
            special.add(key)
            continue

        Xval, yval = get_X_y(df_val_)

        xg_regressor = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50)
        xg_regressor.fit(Xtrain, ytrain, eval_set=[(Xtrain, ytrain), (Xval, yval)], verbose=False)

        pred_ = xg_regressor.predict(Xval)
        dict_regressors[key] = xg_regressor

        pred.extend(pred_)
        tgt.extend(yval)

    out = []
    for r in df_test.itertuples():
        key = f"{r.store_nbr}-{r.family}"
        if key not in dict_regressors:
            out.append(0)
        else:
            regressor = dict_regressors[key]

            X = np.array([r.dayinyear_sin, r.dayinyear_cos, r.month_sin, r.month_cos])[None, :]
            pred = regressor.predict(X)

            out.append(pred.item())

    msle_val = mean_squared_log_error(tgt, pred)
    logger.info(f"MSLE val: {msle_val:.5f}")

    df_test["sales"] = out
    df_test["sales"].to_csv("submission.csv")
