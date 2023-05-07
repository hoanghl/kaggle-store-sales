from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src import utils

SECONDS_IN_YEAR = 365 * 24 * 60 * 60
FIELDS = [
    "sales",
    "store_nbr",
    "family",
    "city",
    "state",
    "store_type",
    "cluster",
    "date_type_local",
    "date_name_local",
    "date_type_region",
    "date_name_region",
    "date_type_nation_1",
    "date_name_nation_1",
    "date_type_nation_2",
    "date_name_nation_2",
    "onpromotion",
    "dcoilwtico",
    "transactions",
    "date_sin",
    "date_cos",
]


tqdm.pandas()


def _encode_date_cyclical(date: str) -> Tuple[np.ndarray, np.ndarray]:
    ts = datetime.timestamp(pd.to_datetime(date))

    ts_sin = np.sin(ts * (2 * np.pi / SECONDS_IN_YEAR))
    ts_cos = np.cos(ts * (2 * np.pi / SECONDS_IN_YEAR))

    return ts_sin, ts_cos


def _build_feats(d: pd.DataFrame, conf: dict, tag: str):
    """Build features and save for each dataset

    Args:
        d (pd.DataFrame): Dataset
        conf (dict): configuration
        tag (str): "train", "test" or "val"
    """

    assert tag in ["train", "val", "test"], f"'tag' value = {tag}: differ from ['train', 'test', 'val]"

    frame_size = conf["GENERAL"]["Lh"] + conf["GENERAL"]["Lp"]

    # Extract to features from dataframe
    out = []

    d["date_sin"], d["date_cos"] = zip(*d["date"].map(_encode_date_cyclical))
    d["transactions"] = (d["transactions"] + 1e-4).map(np.log)

    for store in range(1, 55, 1):
        for family in range(1, 34, 1):
            logger.info(f"== {tag}: store = {store:3d} -- {family:3d}")

            df_ = d[(d["store_nbr"] == store) & (d["family"] == family)]

            for _, w in df_.groupby(np.arange(len(df_)) // frame_size):
                if len(w) < frame_size:
                    continue

                feats = w[FIELDS].to_numpy()

                out.append(feats[np.newaxis, :])

    feats = np.vstack(out)
    # shape: [DATASET_SIZE, Lh + Lp, NUM_FEATS = 18]

    # Save to npz file
    path_processed = conf["PATH"]["processed"][tag]

    np.save(path_processed, feats)


def build_feats():
    """Build features for training and testing data"""

    conf = utils.load_conf()

    # Load intermediate data
    path_inter_train = conf["PATH"]["inter"]["fact_train"]
    path_inter_test = conf["PATH"]["inter"]["fact_test"]

    fact_train = pd.read_csv(path_inter_train, parse_dates=["date"])
    fact_test = pd.read_csv(path_inter_test, parse_dates=["date"])

    # Split train to train and validate set
    date_split = "2015-12-24"
    fact_val = fact_train[fact_train["date"] > date_split]
    fact_train = fact_train[fact_train["date"] <= date_split]

    # Make up feature and save
    _build_feats(fact_train, conf, "train")
    _build_feats(fact_test, conf, "test")
    _build_feats(fact_val, conf, "val")
