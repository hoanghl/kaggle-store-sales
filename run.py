import argparse

import src


def parse_args():
    parser = argparse.ArgumentParser(description="Kaggle time-series competition")
    parser.add_argument("--task", "-t", type=str)
    parser.add_argument("--config", "-c", type=str, default="conf.yaml")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.task == "train":
        trainer = src.models.Trainer()

        trainer.train()
    elif args.task == "test":
        trainer = src.models.Trainer()

        trainer.test()

    elif args.task == "build_feats":
        src.features.build_feats()
    else:
        raise NotImplementedError()
