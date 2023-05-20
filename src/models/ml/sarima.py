import json
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm.contrib.itertools import product as t_product

from src import utils

warnings.filterwarnings("ignore")


def _find_best_param(series: pd.Series, max_p: int = 7, max_q: int = 7) -> tuple[int, int, float]:
    """Return pair of p, q and AIC score

    Args:
        series (pd.Series): sales of each product - store pair

    Returns:
        tuple[int, int, float]: tuple of p, q and AIC
    """

    best_p, best_q = -1, -1
    best_aic = 10e10

    for p, q in t_product(range(max_p), range(max_q)):
        try:
            model = SARIMAX(series, order=(p, 0, q), simple_differencing=False).fit(disp=False)
        except Exception:
            continue

        aic = model.aic

        if aic < best_aic:
            best_aic = aic
            best_p, best_q = p, q

    return best_p, best_q, best_aic


def run_sarima(n_split: int = 0, max_split: int = 0):
    """Run SARIMA with each combinations of stores and products

    Args:
        n_split (int, optional): order of split to run in parallel. Defaults to 0.
    """

    conf = utils.load_conf()

    path_log = (
        Path(conf["LOG"].replace("<model>", "sarima"))
        / datetime.now().strftime(r"%Y%m%d_%H%M%S")
        / f"{n_split:02d}.json"
    )
    path_log.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    path_processed_train = conf["PATH"]["processed"]["train"]
    path_processed_val = conf["PATH"]["processed"]["val"]

    df_train = pd.read_csv(path_processed_train, parse_dates=["date"])
    df_val = pd.read_csv(path_processed_val, parse_dates=["date"])

    # Run SARIMA model
    result = {}

    if max_split == 0:
        loop = product(df_train["store_nbr"].unique(), df_train["family"].unique())
    else:
        l = list(product(df_train["store_nbr"].unique(), df_train["family"].unique()))
        q = len(l) // max_split
        loop = l[q * n_split : q * (n_split + 1)] if n_split < max_split - 1 else l[q * n_split :]
    for store, family in loop:
        logger.info(f"store: {store} -- family: {family}")

        df_train_ = df_train[(df_train["store_nbr"] == store) & (df_train["family"] == family)]
        df_val_ = df_val[(df_val["store_nbr"] == store) & (df_val["family"] == family)]

        df_train_ = df_train_.sort_values(by="date").set_index(keys="date")

        series = df_train_["sales"].diff(1)[1:]

        # Find best pair (p, q)
        p, q, aic = _find_best_param(series)

        # Establish model and forecast
        model = SARIMAX(series, order=(p, 0, q), simple_differencing=False).fit(disp=False)

        pred_diff = model.forecast(15)
        reversed = np.concatenate(
            ([df_train_.iloc[0]["sales"]], df_train_["sales"].diff(1)[1:].to_numpy(), pred_diff.to_numpy()), axis=0
        ).cumsum()
        pred = reversed[-15:]

        tgt = df_val_["sales"].sort_index()

        result[f"{store}-{family}"] = {"tgt": tgt.tolist(), "pred": pred.tolist()}

    with open(path_log) as fp:
        json.dump(result, fp, indent=2)

    # # Calculate final MSLE
    # big_tgt, big_pred = [], []
    # for d in result.values():
    #     big_tgt.append(d["tgt"])
    #     big_pred.append(d["pred"])

    # big_tgt_np = np.concatenate(big_tgt, axis=0)
    # big_pred_np = np.concatenate(big_pred, axis=0)

    # final_msle = metrics.mean_squared_log_error(big_tgt_np, big_pred_np)

    # logger.info(f"MSLE: {final_msle:.4f}")
