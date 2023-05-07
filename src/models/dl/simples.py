import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, d_feat: int = 32, d_hid: int = 128) -> None:
        super().__init__()

        self.lin1 = nn.Linear(d_feat, d_hid)
        self.fc1 = nn.Sequential(
            nn.Linear(d_hid, d_hid),
            nn.LeakyReLU(),
            nn.LayerNorm(d_hid),
            nn.Linear(d_hid, d_hid),
            nn.LeakyReLU(),
            nn.LayerNorm(d_hid),
        )
        self.lin2 = nn.Sequential(nn.Linear(d_hid, 1), nn.LeakyReLU())

    def forward(self, x):
        x = self.lin1(x)
        x = self.fc1(x)
        x = self.lin2(x)

        x = x.squeeze(-1)

        return x


class SimpleLSTM(nn.Module):
    def __init__(self, d_feat: int = 32, d_hid: int = 128) -> None:
        super().__init__()

        self.lin1 = nn.Linear(1, d_hid)
        self.lstm = nn.LSTM(d_hid, d_hid, 4, batch_first=True, dropout=0.1)
        self.lin2 = nn.Sequential(nn.Linear(d_hid, 1), nn.LeakyReLU())

    def forward(self, x):
        # x: [bz, L]

        x = x.unsqueeze(-1)
        # [bz, L, 1]

        x = self.lin1(x)
        # [bz, L, d_hid]

        x, (_, _) = self.lstm(x)
        # h: [bz, L, d_hid]

        x = x[:, -1]
        # [bz, d_hid]

        x = self.lin2(x)

        x = x.squeeze(-1)

        return x
