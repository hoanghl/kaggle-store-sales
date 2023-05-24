# -*- coding: utf-8 -*-

import re
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src import utils

tqdm.pandas()


RE_DESCRIPTION = re.compile(r"((\w|\s|\:)+)(\+|\-)\d+")
NATION = "ecuador"
NUM_STORES = 54
IGNORED_VAL = 0
DATE_SPLIT_TRAIN_VAL = "2017-08-01"

conf = utils.load_conf()

if Path(conf["PATH"]["inter"]["holiday"]).exists():
    holiday = pd.read_csv(conf["PATH"]["inter"]["holiday"], parse_dates=["date"])
else:
    holiday = dict()


def _make_processed_holiday(r: pd.Series) -> dict:
    date_obj = r["date"]

    ret = {}
    if isinstance(r.date_type_national, str):
        ret = {"date_type_national": r.date_type_national, "date_name_national": r.date_name_national}

    else:
        if date_obj.weekday() >= 5:
            date_type_nation, date_name_nation = "weekend", "weekend"
        else:
            date_type_nation, date_name_nation = "work day", "work day"

        ret = {"date_type_national": date_type_nation, "date_name_national": date_name_nation}

    return ret


def _make_processed_dataframe(df: pd.DataFrame, paths: dict) -> pd.DataFrame:
    # Load things from CSV
    # pd.read_csv(paths["raw"]["transactions"], parse_dates=["date"])
    df_oil = pd.read_csv(paths["inter"]["oil"], parse_dates=["date"])
    df_stores = pd.read_csv(paths["raw"]["store"])

    df_holiday = pd.read_csv(paths["inter"]["holiday"], parse_dates=["date"])

    # Left join
    df_final = df.merge(df_oil, "left", on="date")
    # df_final = df_final.merge(df_trans, "left", on=["date", "store_nbr"])
    df_final = df_final.merge(df_stores, how="left", on="store_nbr")

    # Supplement holiday info
    df_holiday_loc = df_holiday[df_holiday["locale"] == "local"]
    df_holiday_reg = df_holiday[df_holiday["locale"] == "regional"]
    df_holiday_nat = df_holiday[df_holiday["locale"] == "national"]

    df_final = df_final.merge(
        df_holiday_loc, how="left", left_on=["city", "date"], right_on=["locale_name", "date"]
    ).rename(
        columns={
            "date_type": "date_type_local",
            "date_name": "date_name_local",
            "locale": "locale_local",
            "locale_name": "locale_name_local",
        }
    )

    df_final = df_final.merge(
        df_holiday_reg, how="left", left_on=["city", "date"], right_on=["locale_name", "date"]
    ).rename(
        columns={
            "date_type": "date_type_regional",
            "date_name": "date_name_regional",
            "locale": "locale_regional",
            "locale_name": "locale_name_regional",
        }
    )

    df_final = df_final.merge(df_holiday_nat, how="left", left_on=["date"], right_on=["date"])
    df_final = df_final.drop(columns=["locale", "locale_name"]).rename(
        columns={"date_type": "date_type_national", "date_name": "date_name_national"}
    )

    out = df_final.progress_apply(_make_processed_holiday, axis=1, result_type="expand")
    df_final[["date_type_national", "date_name_national"]] = out

    # Reallocation old indices
    df_final = df_final.fillna("ignored").set_index(df.index)

    return df_final


def make_processed():
    paths = conf["PATH"]

    df_train = pd.read_csv(paths["raw"]["train"], parse_dates=["date"], index_col="id")
    df_test = pd.read_csv(paths["raw"]["test"], parse_dates=["date"], index_col="id")

    df_train_processed = _make_processed_dataframe(df_train, paths)
    df_test_processed = _make_processed_dataframe(df_test, paths)

    # Split train-val
    df_val_processed = df_train_processed[df_train_processed["date"] >= DATE_SPLIT_TRAIN_VAL]
    df_train_processed = df_train_processed[df_train_processed["date"] < DATE_SPLIT_TRAIN_VAL]

    # Save processed
    df_train_processed.to_csv(paths["processed"]["train"], index=True)
    df_val_processed.to_csv(paths["processed"]["val"], index=True)
    df_test_processed.to_csv(paths["processed"]["test"], index=True)
