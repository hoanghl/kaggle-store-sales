from pathlib import Path

import yaml


def load_conf(path: str = "conf.yaml") -> dict:
    assert Path(path).exists()

    with open(path, encoding="utf-8") as fp:
        conf: dict = yaml.full_load(fp)

    return conf
