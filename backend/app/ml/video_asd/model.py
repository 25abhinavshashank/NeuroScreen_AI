from __future__ import annotations

"""LSTM classifier head, ported 1:1 from the reference repo's
vgg16LSTMVideoClassifier (recurrent_networks.py):
  LSTM(512, dropout=0.5) -> Dense(512, relu) -> Dropout(0.5) -> Dense(2)
Softmax is handled by CrossEntropyLoss during training rather than as
an explicit final layer (the standard PyTorch pattern)."""

import torch
from torch import nn

HIDDEN_UNITS = 512


class VideoASDClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2, hidden_units: int = HIDDEN_UNITS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_units,
            batch_first=True,
            dropout=0.0,  # single-layer LSTM; reference's 0.5 dropout applies to inputs, handled below
        )
        self.input_dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(hidden_units, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_dropout(x)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(x)

        last_hidden = hidden[-1]  # (batch, hidden_units)
        out = self.relu(self.dense1(last_hidden))
        out = self.dropout(out)
        return self.dense2(out)  # logits; use CrossEntropyLoss
