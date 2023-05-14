# -*- coding: utf-8 -*-

import re
from pathlib import Path
from typing import Literal

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
    def _set_default(default: Literal["work day", "weekend"] = "work day"):
        date_type = date_name = default

        return date_type, date_name

    date_type_local, date_name_local = "ignored", "ignored"
    date_type_region, date_name_region = "ignored", "ignored"
    date_type_nation_1, date_name_nation_1 = "ignored", "ignored"
    date_type_nation_2, date_name_nation_2 = "ignored", "ignored"

    df = holiday[holiday["date"] == r["date"]]
    date_obj = r["date"]

    if len(df) == 0:
        if date_obj.weekday() >= 5:
            date_type_nation_1, date_name_nation_1 = _set_default("weekend")
        else:
            date_type_nation_1, date_name_nation_1 = _set_default()
    else:
        # Get local holiday info
        d_local = df[df["locale_name"] == r["city"]]
        assert len(d_local) <= 1

        if len(d_local) == 1:
            row = d_local.iloc[0]

            date_type_local = row["date_type"]
            date_name_local = row["date_name"]
            # r["city"]

        # Get regional holiday info
        d_reg = df[df["locale_name"] == r["state"]]
        assert len(d_reg) <= 1

        if len(d_reg) == 1:
            row = d_reg.iloc[0]

            date_type_region = row["date_type"]
            date_name_region = row["date_name"]
            # r["state"]

        # Get national holiday info
        d_nat = df[df["locale_name"] == NATION]
        assert len(d_nat) <= 2

        if len(d_nat) >= 1:
            row = d_nat.iloc[0]

            date_type_nation_1 = row["date_type"]
            date_name_nation_1 = row["date_name"]

            if len(d_nat) == 2:
                row = d_nat.iloc[1]

                date_type_nation_2 = row["date_type"]
                date_name_nation_2 = row["date_name"]
        else:
            if date_obj.weekday() >= 5:
                date_type_nation_1, date_name_nation_1 = _set_default("weekend")
            else:
                date_type_nation_1, date_name_nation_1 = _set_default()

    out = {
        "date_type_local": date_type_local,
        "date_name_local": date_name_local,
        "date_type_region": date_type_region,
        "date_name_region": date_name_region,
        "date_type_nation_1": date_type_nation_1,
        "date_name_nation_1": date_name_nation_1,
        "date_type_nation_2": date_type_nation_2,
        "date_name_nation_2": date_name_nation_2,
    }

    return out


def _process(r: pd.Series):
    out = r.to_dict()

    holidays = _make_processed_holiday(r)
    out |= holidays

    return out


def _make_processed_dataframe(df: pd.DataFrame, paths: dict) -> pd.DataFrame:
    # Load things from CSV
    # pd.read_csv(paths["raw"]["transactions"], parse_dates=["date"])
    df_oil = pd.read_csv(paths["inter"]["oil"], parse_dates=["date"])
    df_stores = pd.read_csv(paths["raw"]["store"])

    df_holiday_inter = pd.read_csv(paths["inter"]["holiday"], parse_dates=["date"])

    # Left join
    df_final = df.merge(df_oil, "left", on="date")
    # df_final = df_final.merge(df_trans, "left", on=["date", "store_nbr"])
    df_final = df_final.merge(df_stores, how="left", on="store_nbr")
    df_final = df_final.merge(df_holiday_inter, how="left", on="date")

    # Supplement holiday info
    out = df_final.progress_apply(_process, axis=1, result_type="expand")
    df_final[out.columns] = out

    # Remove redundant columns

    df_final.drop(columns=["date_type", "date_name", "locale_name"], inplace=True)

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
    df_train_processed.to_csv(paths["processed"]["train"], index=False)
    df_val_processed.to_csv(paths["processed"]["val"], index=False)
    df_test_processed.to_csv(paths["processed"]["test"], index=False)
