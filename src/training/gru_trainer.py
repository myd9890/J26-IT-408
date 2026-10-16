"""
PyTorch GRU Longitudinal Forecaster Training Routine (FR10).

Executes:
1. Plant-level partitioned longitudinal dataset loading.
2. SequenceDataset creation and batch collation with length tracking.
3. Model training with AdamW and CosineAnnealingLR scheduling.
4. Model checkpointing to results/gru_forecaster.pt.
5. Evaluation on test partition (RMSE and MAE).
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.data.dataset import plant_level_split
from src.features.vector_builder import build_sequences_from_df
from src.models.temporal.gru_net import SeverityGRU
from src.utils.config import get_config


class SequenceDataset(Dataset):
    """Dataset yielding (input_sequence_tensor, target_scalar_tensor)."""

    def __init__(self, sequences: List[np.ndarray], targets: List[float]):
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1) / 100.0

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def pad_collate_fn(batch):
    """Pads sequences of variable length in the batch and tracks true valid lengths."""
    sequences, targets = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    batch_size = len(sequences)
    input_size = sequences[0].shape[1]

    padded_seqs = torch.zeros(batch_size, max_len, input_size, dtype=torch.float32)
    for i, seq in enumerate(sequences):
        padded_seqs[i, :len(seq), :] = seq

    targets_tensor = torch.stack(targets, dim=0)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    return padded_seqs, targets_tensor, lengths_tensor


def train_gru(
    metadata_path: str = "data/synthetic/metadata.csv",
    output_weights_path: str = "results/gru_forecaster.pt",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    hidden_size: Optional[int] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Trains the SeverityGRU on plant-level partitioned longitudinal sequences.
    """
    epochs = epochs or int(get_config("gru.epochs", 50))
    batch_size = batch_size or int(get_config("gru.batch_size", 16))
    lr = lr or float(get_config("gru.learning_rate", 0.002))
    hidden_size = hidden_size or int(get_config("gru.hidden_size", 32))

    torch.manual_seed(seed)
    np.random.seed(seed)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path)
    train_df, val_df, test_df = plant_level_split(metadata_df, 0.70, 0.15, 0.15, seed=seed)

    X_train_seq, y_train = build_sequences_from_df(train_df)
    X_val_seq, y_val = build_sequences_from_df(val_df)
    X_test_seq, y_test = build_sequences_from_df(test_df)

    train_ds = SequenceDataset(X_train_seq, y_train)
    val_ds = SequenceDataset(X_val_seq, y_val)
    test_ds = SequenceDataset(X_test_seq, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    model = SeverityGRU(input_size=5, hidden_size=hidden_size, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(output_weights_path), exist_ok=True)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for seqs, targets, lengths in train_loader:
            optimizer.zero_grad()
            preds = model(seqs, lengths=lengths)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        scheduler.step()
        train_loss_history.append(float(np.mean(batch_losses)))

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for seqs, targets, lengths in val_loader:
                preds = model(seqs, lengths=lengths)
                loss = criterion(preds, targets)
                val_losses.append(loss.item())

        mean_val_loss = float(np.mean(val_losses))
        val_loss_history.append(mean_val_loss)

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mse": best_val_loss,
                "hidden_size": hidden_size,
                "input_size": 5
            }, output_weights_path)

    # Save training history JSON
    history_path = os.path.join(os.path.dirname(output_weights_path), "gru_training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "epochs": list(range(1, epochs + 1)),
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "best_epoch": int(torch.load(output_weights_path, map_location="cpu", weights_only=False)["epoch"])
        }, f, indent=2)

    # Evaluate on test set
    checkpoint = torch.load(output_weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds = []
    test_actuals = []
    with torch.no_grad():
        for seqs, targets, lengths in test_loader:
            preds = model(seqs, lengths=lengths)
            test_preds.extend((preds.squeeze(1) * 100.0).tolist())
            test_actuals.extend((targets.squeeze(1) * 100.0).tolist())

    test_preds_np = np.array(test_preds)
    test_actuals_np = np.array(test_actuals)

    rmse = float(np.sqrt(np.mean((test_actuals_np - test_preds_np) ** 2)))
    mae = float(np.mean(np.abs(test_actuals_np - test_preds_np)))

    results = {
        "gru_test_rmse": round(rmse, 4),
        "gru_test_mae": round(mae, 4),
        "best_epoch": checkpoint["epoch"],
        "num_test_sequences": len(test_preds)
    }

    print("[GRU Train] Test Results:", results)
    return results
