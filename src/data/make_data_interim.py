# -*- coding: utf-8 -*-

from typing import List

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from src import utils

from . import utils as utils_data

tqdm.pandas()


def _make_interim_holiday(paths: dict):
    logger.info("= Make interim: holiday")

    path_raw = paths["raw"]["holiday"]
    path_inter = paths["inter"]["holiday"]

    holidayinfo: List[dict] = []
    df = pd.read_csv(path_raw)

    # Preprocess some columns
    df.loc[:, "type"] = df["type"].str.lower()
    df.loc[:, "locale_name"] = df["locale_name"].str.lower()
    df.loc[:, "locale"] = df["locale"].str.lower()
    df.loc[:, "description"] = df["description"].str.lower().apply(utils_data.proc_des)

    # Start looping
    for entry in tqdm(df.itertuples(), total=len(df)):
        date, date_type, locale_name, date_name, ignored = utils_data.process_entry(
            entry, df
        )

        if not ignored:
            holidayinfo.append(
                {
                    "date": date,
                    "date_type": date_type,
                    "date_name": date_name,
                    "locale_name": locale_name,
                }
            )

    # Write processed file
    header = list(holidayinfo[0].keys())
    rows = [list(x.values()) for x in holidayinfo]

    utils.write_csv(path_inter, header, rows)


def make_interim():
    paths = utils.load_conf()["PATH"]

    _make_interim_holiday(paths)
