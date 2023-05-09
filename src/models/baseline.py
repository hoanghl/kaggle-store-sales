import itertools

import numpy as np
import pandas as pd
from loguru import logger
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src import utils

date_split_train_val = "2017-07-31"


def run_baseline():
    conf = utils.load_conf()

    path_raw_train = conf["PATH"]["raw"]["train"]

    df = pd.read_csv(path_raw_train, parse_dates=["date"])

    # Simple preprocess
    encoder = LabelEncoder()
    df["family"] = encoder.fit_transform(df["family"])

    # Split train-val
    df_val = df[df["date"] >= date_split_train_val]
    df_train = df[df["date"] < date_split_train_val]

    # Start building baseline
    forecast = pd.DataFrame()
    couples = itertools.product(df["store_nbr"].unique(), df["family"].unique())
    for store, family in tqdm(list(couples)):
        df_ = df_train[(df_train["family"] == family) & (df_train["store_nbr"] == store)]
        sales = list(df_["sales"])
        for _ in range(16):
            sales.append(np.mean(sales[-7:]))

        target = df_val[(df_val["store_nbr"] == store) & (df_val["family"] == family)].copy()
        target["sales_prediction"] = sales[-16:]
        forecast = pd.concat([forecast, target])

    ## Calculate metric
    msle = metrics.mean_squared_log_error(forecast["sales"], forecast["sales_prediction"])

    logger.info(f"MSLE: {msle:.4f}")
