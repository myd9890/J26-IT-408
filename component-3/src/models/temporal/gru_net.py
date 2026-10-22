"""
PyTorch SeverityGRU Recurrent Neural Network Architecture (FR10).

Predicts 7-day severity progression from sequential multimodal observation vectors:
s_t = [severity_score, urgency_score, scope_pct, impact_score, disease_id] normalized to [0, 1].
"""

from typing import Optional
import torch
import torch.nn as nn


class SeverityGRU(nn.Module):
    """
    Recurrent neural network with GRU cell for 7-day severity progression forecasting.
    Includes sequence-length masking to prevent trailing zero-padding pollution.
    """

    def __init__(self, input_size: int = 5, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()  # Guarantees normalized output in [0.0, 1.0]
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, seq_len, input_size) with values in [0.0, 1.0].
            lengths: Optional 1D LongTensor of valid sequence lengths in the batch.
                     If provided, extracts the hidden state at the true final observation,
                     avoiding trailing zero-padding pollution.

        Returns:
            Normalized predictions of shape (B, 1) in [0.0, 1.0].
        """
        out, _ = self.gru(x)
        if lengths is not None:
            batch_idx = torch.arange(x.size(0), device=x.device)
            valid_idx = torch.clamp(lengths.to(x.device) - 1, min=0)
            last_hidden = out[batch_idx, valid_idx, :]
        else:
            last_hidden = out[:, -1, :]  # (B, hidden_size)
        pred_norm = self.head(last_hidden)  # (B, 1)
        return pred_norm
