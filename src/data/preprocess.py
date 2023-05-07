import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklego.preprocessing import RepeatingBasisFunction
from tqdm import tqdm

from . import utils

RE_DESCRIPTION = re.compile(r"((\w|\s|\:)+)(\+|\-)\d+")
NATION = "ecuador"
NUM_STORES = 54
IGNORED_VAL = 0
DATE_START, DATE_END = "2012-01-01", "2017-12-31"

tqdm.pandas()


class Processor:
    def _establish_db(self, *args, **kwargs):
        pass

    def encode(self, row):
        pass


class Mapping(Processor):
    def __init__(self, path_map) -> None:
        logger.info("Init Mapping")

        path_map = Path(path_map)

        assert path_map.exists(), f"Path to map not existed: {path_map}"

        self.mapping = self._establish_db(path_map)

    def _establish_db(self, path_map: Path):
        out = {}

        with open(path_map) as fp:
            d = json.load(fp)
        for k, v in d.items():
            out[k] = {x: i + 1 for i, x in enumerate(v)} | {"ignored": IGNORED_VAL}

        return out

    def __getitem__(self, k):
        return self.mapping[k]


class Holiday(Processor):
    def __init__(self, mapping: Mapping, path_raw_holiday: str, path_res_holiday: str) -> None:
        logger.info("Init Holiday")

        self.mapping = mapping

        if Path(path_res_holiday).exists():
            self.holiday = pd.read_csv(path_res_holiday)
        else:
            assert (
                path_raw_holiday is not None and Path(path_raw_holiday).exists()
            ), "As 'path_res_holiday' not existed, 'path_raw_holiday' must be specified"

            self.holiday = self._establish_db(path_raw_holiday)

            # Save to CSV
            self.holiday.to_csv(path_res_holiday, index=False)

    def preprocess(self, r: pd.Series) -> list:
        """Get info of given date

        Args:
            date (str): format "yyyy-mm-dd"
            city (str): city name (used to look for 'local')
            state (str): state name (used to look for 'regional')

        Returns:
            tuple: tuple of encoded holiday info
        """

        # TODO: HoangLe [Mar-06]: Reimplement this method

        def _set_default(default: Literal["work day", "weekend"] = "work day"):
            date_type = date_name = default

            return date_type, date_name

        date_type_local, date_name_local = "ignored", "ignored"
        date_type_region, date_name_region = "ignored", "ignored"
        date_type_nation_1, date_name_nation_1 = "ignored", "ignored"
        date_type_nation_2, date_name_nation_2 = "ignored", "ignored"

        df = self.holiday[self.holiday["date"] == r["date"]]
        date_obj = datetime.strptime(r["date"], "%Y-%m-%d")

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
                r["city"]

            # Get regional holiday info
            d_reg = df[df["locale_name"] == r["state"]]
            assert len(d_reg) <= 1

            if len(d_reg) == 1:
                row = d_reg.iloc[0]

                date_type_region = row["date_type"]
                date_name_region = row["date_name"]
                r["state"]

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

        out = [
            date_type_local,
            date_name_local,
            date_type_region,
            date_name_region,
            date_type_nation_1,
            date_name_nation_1,
            date_type_nation_2,
            date_name_nation_2,
        ]

        return out

    def _establish_db(self, path_raw: str):
        holidayinfo = []
        df = pd.read_csv(path_raw)

        # Preprocess some columns
        df.loc[:, "type"] = df["type"].str.lower()
        df.loc[:, "locale_name"] = df["locale_name"].str.lower()
        df.loc[:, "locale"] = df["locale"].str.lower()
        df.loc[:, "description"] = df["description"].str.lower().apply(utils.proc_des)

        # Start looping
        for entry in df.itertuples():
            date, date_type, locale_name, date_name, ignored = utils.process_entry(entry, df)

            if not ignored:
                holidayinfo.append(
                    {"date_type": date_type, "date": date, "date_name": date_name, "locale_name": locale_name}
                )

        out = pd.DataFrame.from_records(holidayinfo)

        return out

    def encode(self, row: pd.Series):
        return [
            self.mapping["date_type"][row["date_type_local"]],
            self.mapping["date_name"][row["date_name_local"]],
            self.mapping["date_type"][row["date_type_region"]],
            self.mapping["date_name"][row["date_name_region"]],
            self.mapping["date_type"][row["date_type_nation_1"]],
            self.mapping["date_name"][row["date_name_nation_1"]],
            self.mapping["date_type"][row["date_type_nation_2"]],
            self.mapping["date_name"][row["date_name_nation_2"]],
        ]


class OilPrice:
    def __init__(self, path_raw_oil: str, path_res_oil: str, read_idx: bool = False) -> None:
        logger.info("Init OilPrice")

        path_raw, path_res = Path(path_raw_oil), Path(path_res_oil)

        if not path_res.exists():
            # Create db
            assert path_raw.exists(), f"Path to raw oil not existed: {path_raw}"

            self.oil = self._establish_db(path_raw)
            self.oil.to_csv(path_res_oil, index=False)
        else:
            # Load processed
            self.oil = pd.read_csv(path_res, index_col="date" if read_idx is True else None)

    def _establish_db(self, path_raw):
        d = pd.read_csv(path_raw, index_col="date")

        # Some processes
        d.dropna(inplace=True)
        d.sort_values(["date"], inplace=True)

        # Fill missing
        oilprice = d.to_dict()["dcoilwtico"]
        final_oilprice = {}
        last_oil = IGNORED_VAL
        for d in pd.date_range(start=DATE_START, end=DATE_END):
            date = d.strftime("%Y-%m-%d")
            if date not in oilprice:
                final_oilprice[date] = last_oil
            else:
                final_oilprice[date] = oilprice[date]
                last_oil = oilprice[date]

        # Convert back to DataFrame
        data = {"date": list(final_oilprice.keys()), "dcoilwtico": list(final_oilprice.values())}
        d = pd.DataFrame.from_dict(data)

        # Scale: MinMaxScaler
        utils.scale_minmax(d, "dcoilwtico")

        return d

    def get_oilprice(self, date: str) -> float:
        oilprice = -1
        if date in self.oil.index:
            oilprice = self.oil.loc[date].dcoilwtico
        else:
            date_obj = datetime.strptime(date, "%Y-%m-%d")

            while date_obj > datetime(2011, 12, 31):
                date_obj = date_obj - timedelta(days=1)

                if date_obj.strftime("%Y-%m-%d") in self.oil.index:
                    oilprice = self.oil.loc[date_obj.strftime("%Y-%m-%d")].dcoilwtico
                    break

        return oilprice


class DateTime(Processor):
    def __init__(self) -> None:
        logger.info("Init DateTime")

        self.dt = self._establish_db()

    def _establish_db(self):
        range_of_dates = pd.date_range(start=DATE_START, end=DATE_END)

        X = pd.DataFrame(index=range_of_dates)
        X["day_nr"] = range(len(X))
        X["day_of_year"] = X.index.day_of_year

        rbf = RepeatingBasisFunction(n_periods=12, column="day_of_year", input_range=(1, 365), remainder="drop")
        rbf.fit(X)
        X = pd.DataFrame(index=X.index, data=rbf.transform(X))

        return X

    def encode(self, datetime: str) -> list:
        assert DATE_START <= datetime <= DATE_END, f"Given datetime invalid: {datetime}"

        return list(self.dt.loc[datetime])


class Store(Processor):
    def __init__(self, mapping: Mapping, path_raw_store: str, path_res_store: str, read_idx: bool = False) -> None:
        logger.info("Init Store")

        self.mapping = mapping

        path_raw, path_res = Path(path_raw_store), Path(path_res_store)

        if not path_res.exists():
            # Create db
            assert path_raw.exists(), f"Path to store's data not existed: {path_raw}"

            self.stores: pd.DataFrame = self._establish_db(path_raw)
            self.stores.to_csv(path_res)
        else:
            # Load processed
            self.stores = pd.read_csv(path_res, index_col="store_nbr" if read_idx is True else None)

    def _establish_db(self, path_raw):
        stores = pd.read_csv(path_raw, index_col="store_nbr")

        stores["city"] = stores["city"].str.lower()
        stores["state"] = stores["state"].str.lower()

        return stores

    def encode(self, row: pd.Series) -> list:
        return [
            self.mapping["city"][row["city"]],
            self.mapping["state"][row["state"]],
            self.mapping["store_type"][row["type"]],
            row["cluster"],
        ]


class Transaction(Processor):
    def __init__(self, path_raw) -> None:
        logger.info("Init Transaction")

        self.transaction = self._establish_db(path_raw)

    def _establish_db(self, path_raw: str):
        df = pd.read_csv(path_raw, header=0, index_col="id")

        df.loc[:, "family"] = df["family"].str.lower()

        return df


class TotalTrans(Processor):
    def __init__(self, path_raw) -> None:
        self.tot_trans = pd.read_csv(path_raw)


class Preprocess:
    def __init__(self, paths: dict[str, str], read_idx: bool = False) -> None:
        self.paths = paths

        self.mapping = Mapping(paths["mapping"])

        self.dt = DateTime()
        self.tot_trans = TotalTrans(paths["raw_tot_trans"])
        self.oilprice = OilPrice(paths["raw_oil"], paths["res_oil"], read_idx)
        self.holiday = Holiday(self.mapping, paths["raw_holiday"], paths["res_holiday"])
        self.store = Store(self.mapping, paths["raw_store"], paths["res_store"], read_idx)
        self.trans = Transaction(paths["raw_transaction"])

    def _encode_row(self, r: pd.Series):
        out = []

        out.append(r["sales"])
        out.append(r["store_nbr"])

        # Encode datetime
        # dt_enc = self.dt.encode(r['date'])
        # out.extend(dt_enc)
        out.append(r["date"])

        # Encode product family
        family_enc = self.mapping["family"][r["family"]]
        out.append(family_enc)

        # Encode store
        store_enc = self.store.encode(r)
        out.extend(store_enc)

        # Encode holiday
        holiday_enc = self.holiday.encode(r)
        out.extend(holiday_enc)

        # Append numeric values
        out.append(r["onpromotion"])
        out.append(r["dcoilwtico"])
        out.append(IGNORED_VAL if utils.check_nan(r["transactions"]) else r["transactions"])
        # out.append(r['sales_last1'])
        # out.append(r['sales_last2'])
        # out.append(r['sales_last3'])
        # out.append(r['sales_last4'])

        return out

    def encode(self, df: pd.DataFrame) -> list:
        out = df.progress_apply(self._encode_row, axis=1, result_type="expand")

        return out

    def split_traintest(self, df: pd.DataFrame, date_cutoff: str = "2016-05-21") -> tuple[pd.DataFrame, pd.DataFrame]:
        df_train, df_test = df[(df["date"] < date_cutoff) & (df["date"] >= "2012-01-05")], df[df["date"] >= date_cutoff]

        return df_train, df_test

    def save_split(self, split: Literal["train", "test"], version: str, data: list):
        # Redefine path
        path = Path(self.paths["train"] if split == "train" else self.paths["test"])
        parent = path.parent / version
        parent.mkdir(exist_ok=True, parents=True)
        path = parent / path.name

        # Save data
        data_np = np.array(data, dtype=np.float32)
        np.savez_compressed(path, data_np)

    def create_fact_table(self) -> pd.DataFrame:
        df = pd.read_csv(self.paths["raw_fact"])

        return df
