import argparse

import src


def parse_args():
    parser = argparse.ArgumentParser(description="Kaggle time-series competition")
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        choices=[
            "train",
            "test",
            "make_data_interim",
            "make_data_processed",
            "make_feat",
        ],
    )
    parser.add_argument("--config", "-c", type=str, default="conf.yaml")
    parser.add_argument("--model", "-m", type=str, default="baseline")
    parser.add_argument("--data", "-d", type=str, default="")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.task == "train":
        if args.model == "baseline":
            src.models.run_baseline()
        else:
            trainer = src.models.Trainer()

            trainer.train()
    elif args.task == "test":
        trainer = src.models.Trainer()

        trainer.test()

    elif args.task == "make_data_interim":
        src.data.make_interim()

    elif args.task == "make_data_processed":
        src.data.make_processed()

    elif args.task == "build_feats":
        src.features.build_feats()
    else:
        raise NotImplementedError()
