import torch
import torch.nn as nn

class LSTMVideoSummarizer(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_dim)
        """
        out, _ = self.lstm(x)
        scores = self.fc(out)
        return scores.squeeze(-1)
