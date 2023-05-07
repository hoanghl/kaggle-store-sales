from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAttention(nn.Module):
    def __init__(
        self,
        lp: int,
        lh: int,
        rnn: str = "lstm",
        d_inp: int = 128,
        d_hid: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        ratio_sched_sampling: float = 0.4,
    ) -> None:
        super().__init__()

        self._lp, self._lh = lp, lh
        self.window = lh + 1
        self.ratio_sched_sampling = ratio_sched_sampling

        if rnn == "lstm":
            self.rnn = nn.LSTM(
                input_size=d_inp,
                hidden_size=d_hid,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif rnn == "gru":
            # TODO: HoangLe [May-02]: Implement later
            # self.rnn = nn.GRU()
            pass
        else:
            raise NotImplementedError()

        self.lin1 = nn.Linear(1, d_hid)
        self.ff1 = nn.Sequential(nn.Linear(4 * d_hid, 4 * d_hid), nn.Tanh(), nn.Dropout(dropout))
        self.lin2 = nn.Linear(4 * d_hid, 1)

        self.ff2 = nn.Sequential(nn.Linear(d_hid, d_hid), nn.Tanh(), nn.Dropout(dropout), nn.Linear(d_hid, 1))

    def sched_sampling(self, ratio: float, tgt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Schedule sampling for training RNN

        Args:
            ratio (float): Schedule sampling ration (0 -> 1.0)
            tgt (torch.Tensor): target values
            pred (torch.Tensor): predicted values

        Returns:
            torch.Tensor: output tensor
        """
        # tgt, pred: [bz, w-1]

        sched = (torch.rand_like(tgt) > ratio).float()

        out = sched * pred + (1 - sched) * tgt
        # [bz, w-1]

        return out

    def attention(self, out: torch.Tensor, ht: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # out: [bz, l, d_hid]
        # ht: [2, bz, d_hid]
        # y: [bz, l]
        bz, l, _ = out.shape

        ht = ht.transpose(0, 1).reshape(bz, -1).unsqueeze(1).repeat(1, l, 1)
        # [bz, l, 2 * d_hid]

        yt = self.lin1(y.unsqueeze(-1))
        # [bz, l, d_hid]

        et = self.lin2(self.ff1(torch.cat((out, ht, yt), dim=-1))).squeeze(-1)
        # [bz, l]

        et = F.softmax(et, dim=-1)
        # [bz, l]

        st = et.unsqueeze(1) @ out
        # [bz, 1, d_hid]

        st = self.ff2(st.squeeze(1))
        # [bz, 1]

        return st

    def makup_y(self, i: int, tgt: torch.Tensor, pred: List[torch.Tensor], is_training: bool = True) -> torch.Tensor:
        # tgt: [bz, _lh + _lp]

        y_tgt = tgt[:, i : i + self.window - 1]
        # [b, w-1]

        if pred == []:
            y_out = y_tgt
        else:
            if i >= self._lh:
                j = i - self._lh
                y_out = torch.cat(pred[j : j + self.window - 1], dim=1)
            else:
                pred_ = torch.cat(pred, dim=1)
                # [bz, l1]
                y_out = torch.cat((tgt[:, i : self._lh], pred_), dim=1)
                # [bz, w-1]

        if is_training is True:
            y_out = self.sched_sampling(self.ratio_sched_sampling, y_tgt, y_out)

        return y_out

    def forward(self, X: torch.Tensor, tgt: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        # X: [bz, _lh + _lp, d]
        # tgt: [bz, _lh + _lp]

        tgt = torch.log(tgt + 1e-5)

        pred = []
        for i in range(self._lp):
            Xi = X[:, i : i + self.window]
            # [bz, w, d]
            yi = self.makup_y(i, tgt, pred, is_training)
            # [bz, w - 1]

            Xi, (h, _) = self.rnn(Xi)
            # Xi: [bz, w, d_hid]
            # h: [2, bz, d_hid]

            st = self.attention(Xi[:, :-1], h, yi)
            # [bz, 1]

            pred.append(st)

        # Make up output
        pred_tensor = torch.cat(pred, dim=-1)
        # [bz, _lp]
        out = torch.cat((tgt[:, : self._lh], pred_tensor), dim=-1)
        # [bz, _lh + _lp]

        return out


if __name__ == "__main__":
    bz, lp, lh = 10, 15, 7
    rnn = RNNAttention(15, 7).to("mps")

    X = torch.rand(bz, lh + lp, 128, device="mps")
    y = torch.rand(bz, lh + lp, device="mps")

    out = rnn(X, y)

    print(out.shape)
