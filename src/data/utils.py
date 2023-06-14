import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.stattools import adfuller
from tqdm.contrib.itertools import product

RE_DESCRIPTION = re.compile(r"((\w|\s|\:)+)(\+|\-)\d+")


def proc_des(des: str):
    tmp = re.findall(RE_DESCRIPTION, des)
    if len(tmp) > 0:
        des = tmp[0][0]

    des = des.removeprefix("traslado ")
    des = des.removeprefix("puente ")

    if des.startswith("cantonizacion de"):
        des = "cantonizacion de"
    elif des.startswith("dia de"):
        des = "dia de"
    elif des.startswith("fundacion"):
        des = "fundacion"
    elif des.count("independencia") != 0:
        des = "independencia"
    elif des.startswith("recupero"):
        des = "recupero"
    elif des.startswith("provincializacion"):
        des = "provincializacion"
    elif "de futbol" in des:
        des = "futbol"
    return des


def process_entry(entry, df: pd.DataFrame) -> Tuple:
    locale_name, locale = entry.locale_name, entry.locale
    date = entry.date
    raw_date_type = entry.type
    des = entry.description
    date_type = "work day"
    ignored = False

    # Process case type = 'Event'
    if raw_date_type == "event":
        date_type = "event"

    # Process case type = 'transfer'
    if raw_date_type == "holiday":
        if entry.transferred is False:
            date_type = "holiday"
        else:
            date_type = "work day"

    if raw_date_type == "bridge":
        date_type = "holiday"

    if raw_date_type == "transfer":
        date_type = "holiday"

    # Process case type = 'additional'
    if raw_date_type == "additional":
        # Check whether there is "bridge" occuring in the same date
        c1 = df["date"] == date
        c2 = df["type"].isin(["bridge", "event", "transfer", "holiday"])
        c3 = df["description"] == des
        c4 = df["locale_name"] == locale_name
        df_tmp = df[c1 & c2 & c3 & c4]
        assert len(df_tmp) <= 1
        if len(df_tmp) == 1:
            ignored = True
            return None, None, None, None, None, ignored

        date_type = "additional"

    # Some default set
    if date_type == "work day":
        des = "work day"
    if date_type in ["work day", "weekend"]:
        locale_name = "ecuador"

    return date, date_type, locale, locale_name, des, ignored


def get_prev_day(date_str: str, delta: int = 0) -> str:
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=delta)).strftime("%Y-%m-%d")


def get_delta(date_str1: Union[str, None], date_str2: Union[str, None]) -> int:
    if date_str1 is None or date_str2 is None:
        return 100
    return (datetime.strptime(date_str1, "%Y-%m-%d") - datetime.strptime(date_str2, "%Y-%m-%d")).days


def check_nan(a):
    return a != a


class Meta:
    def __init__(self, paths: dict) -> None:
        self.path_meta = Path(paths["meta"]["meta"])
        self._meta: Union[None, dict] = None

        if self.path_meta.exists():
            self._meta = json.loads(self.path_meta.read_text())

    def gen_json_file(self, df: pd.DataFrame):
        assert all(x in df for x in ["sales", "store_nbr", "family"])

        logger.info(f"Create file: {self.path_meta}")

        meta = {}
        stores, families = df["store_nbr"].unique(), df["family"].unique()

        for store, family in product(stores, families):
            df1 = df[(df["store_nbr"] == store) & (df["family"] == family)].set_index("date")
            df_nonzero = df1[df1["sales"] != 0]

            if len(df_nonzero) == 0:
                continue
            date_fist_nonzero = df_nonzero.iloc[0].name
            sales = df1.loc[date_fist_nonzero:]["sales"]

            # Use ADF test for stationary
            adf_p = adfuller(sales)[1]

            # Calculate max shift
            max_shift = -1
            max_pearson = -1
            for shift in [16, 21, 28, 30, 60, 90, 180, 365]:
                d_shift = sales.shift(shift)
                d_shift = d_shift[~d_shift.isna()]
                if len(d_shift) == 0:
                    continue
                d = sales.loc[d_shift.index]

                pearson = np.corrcoef(d, d_shift)[0, 1]
                if pearson > max_pearson:
                    max_pearson = pearson
                    max_shift = shift

            key = f"{store}-{family}"
            meta[key] = {
                "adf_p": adf_p,
                "pearson": max_pearson,
                "shift": max_shift,
            }

        with open(self.path_meta, "w+") as fp:
            json.dump(meta, fp, indent=2)

    def _get_key(self, store: Union[str, int], family: str) -> str:
        return f"{store}-{family}"

    def check_zero_pair(self, store: Union[str, int], family: str) -> bool:
        assert self._meta is not None

        k = self._get_key(store, family)

        return k not in self._meta

    def get_adf_test(self, store: Union[str, int], family: str) -> float:
        assert self._meta is not None

        k = f"{store}-{family}"

        return self._meta.get(k, {}).get("adf_p", None)

    def get_shift(self, store: Union[str, int], family: str):
        assert self._meta is not None

        k = f"{store}-{family}"

        return self._meta.get(k, {}).get("shift", None)
