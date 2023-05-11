import csv
from pathlib import Path
from typing import Any, List

import yaml


def load_conf(path: str = "conf.yaml") -> dict:
    assert Path(path).exists()

    with open(path, encoding="utf-8") as fp:
        conf: dict = yaml.full_load(fp)

    return conf


def write_csv(path: str, header: List[str], rows: List[List[Any]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w+", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(header)

        writer.writerows(rows)
