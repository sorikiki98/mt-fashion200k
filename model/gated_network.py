import torch
import torch.nn as nn


class GatedNetwork(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, m_i):
        """
               m_i: Tensor of shape (batch_size, T, input_dim)
                    - T: number of turns per sample
                    - input_dim: dimensionality of each turn embedding
               Returns:
                   P_i: Tensor of shape (batch_size, T, d), with sigmoid scores
               """
        lstm_out, _ = self.lstm(m_i)
        mlp_out = self.mlp(lstm_out)
        p_i = torch.sigmoid(mlp_out)
        return p_i
