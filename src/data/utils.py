
import re
from datetime import datetime, timedelta
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

RE_DESCRIPTION = re.compile(r"((\w|\s|\:)+)(\+|\-)\d+")
NATION = 'ecuador'
NUM_STORES = 54


def scale_minmax(df: pd.DataFrame, col: str):
    scaler = MinMaxScaler()
    df.loc[:, col] = scaler.fit_transform(
        np.expand_dims(df[col].to_numpy(), 1)
    ).squeeze()


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


def process_entry(entry, df: pd.DataFrame) -> Tuple[Union[str, None], Union[str, None],
                                                    Union[str, None], Union[str, None], bool]:
    locale_name, date = entry.locale_name, entry.date
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
        c1 = df['date'] == date
        c2 = df['type'].isin(['bridge', 'event', 'transfer', 'holiday'])
        c3 = df['description'] == des
        c4 = df['locale_name'] == locale_name
        df_tmp = df[c1 & c2 & c3 & c4]
        assert len(df_tmp) <= 1
        if len(df_tmp) == 1:
            ignored = True
            return None, None, None, None, ignored

        date_type = "additional"

    # Some default set
    if date_type == "work day":
        des = "work day"
    if date_type in ["work day", "weekend"]:
        locale_name = "ecuador"

    return date, date_type, locale_name, des, ignored


def get_prev_day(date_str: str, delta: int = 0) -> str:
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=delta)).strftime("%Y-%m-%d")


def get_delta(date_str1: Union[str, None], date_str2: Union[str, None]) -> int:
    if date_str1 is None or date_str2 is None:
        return 100
    return (datetime.strptime(date_str1, "%Y-%m-%d") - datetime.strptime(date_str2, "%Y-%m-%d")).days


def check_nan(a):
    return a != a
