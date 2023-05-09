from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanSquaredLogError

from src import utils

from .dl import rnn_attention


class Data(Dataset):
    def __init__(self, path_processed: str) -> None:
        super().__init__()

        self.arr = np.load(path_processed)

    def __len__(self) -> int:
        return self.arr.shape[0]

    def __getitem__(self, index: int) -> Any:
        X = torch.tensor(self.arr[index, :, 1:], dtype=torch.float32)
        y = torch.tensor(self.arr[index, :, 0], dtype=torch.float32)

        return X, y


class LitData(pl.LightningDataModule):
    def __init__(self, conf) -> None:
        super().__init__()

        self._conf = conf

        self.data_train = Data(conf["PATH"]["processed"]["train"])
        self.data_val = Data(conf["PATH"]["processed"]["val"])
        self.data_test = Data(conf["PATH"]["processed"]["test"])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self._conf["TRAINING"]["BATCH_SIZE"], shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self._conf["TRAINING"]["BATCH_SIZE"], num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self._conf["TRAINING"]["BATCH_SIZE"], num_workers=4)


class LitModel(pl.LightningModule):
    def __init__(self, conf: dict) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._conf = conf

        self.net = rnn_attention.RNNAttention(
            conf["GENERAL"]["Lp"],
            conf["GENERAL"]["Lh"],
            d_inp=conf["TRAINING"]["D_FEAT"],
        ).to(dtype=torch.float32)
        self.criterion = nn.MSELoss()
        self.msle = MeanSquaredLogError()

        self._val_step_outputs = []

    def forward(self, batch, is_training: bool = True):
        outputs = self.net(*batch, is_training)

        return outputs

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        outputs = self.forward(batch)
        loss = self.criterion(outputs, torch.log(labels + 1e-6))

        self.log("loss_train", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _, labels = batch

        outputs = self.forward(batch, is_training=False)

        self._val_step_outputs.append((outputs, labels))

    def on_validation_epoch_end(self) -> None:
        pred = torch.cat([x[0] for x in self._val_step_outputs])
        gt = torch.cat([x[1] for x in self._val_step_outputs])

        metric = self.msle(torch.exp(pred), gt)

        self.log("msle_val", metric)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), float(self._conf["TRAINING"]["LR"]), weight_decay=5e-4)
        # optimizer = optim.SGD(self.parameters(), float(self._confg['LR']))

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1)

        return [optimizer], [lr_scheduler]


class Trainer:
    def __init__(self, model: str = "baseline") -> None:
        self._conf = utils.load_conf()

        self._litmodel = LitModel(self._conf)
        self._lidata = LitData(self._conf)

        self.trainer = self._init_trainer()

    def _init_trainer(self):
        # Define callbacks
        dt_str = datetime.now().strftime("%m%d_%H%M%S")

        path_ckpt = Path(self._conf["TRAINING"]["CKPT"]) / dt_str
        callback_modelckpt = ModelCheckpoint(
            str(path_ckpt), monitor="msle_val", save_top_k=1, mode="min", filename="{epoch}-{msle_val:.6f}"
        )
        callback_lr_monitor = LearningRateMonitor(logging_interval="step")
        logger_tboard = TensorBoardLogger(self._conf["TRAINING"]["LOGGER"], default_hp_metric=True, version=dt_str)

        trainer = pl.Trainer(
            devices=1,
            gradient_clip_val=1,
            accelerator=self._conf["TRAINING"]["DEVICE"],
            check_val_every_n_epoch=5,
            max_epochs=self._conf["TRAINING"]["N_EPOCHS"],
            log_every_n_steps=1,
            callbacks=[callback_modelckpt, callback_lr_monitor],
            logger=logger_tboard,
        )

        return trainer

    def train(self):
        self.trainer.fit(model=self._litmodel, datamodule=self._lidata)

    def test(self):
        self.trainer.test(model=self._litmodel, datamodule=self._lidata)
