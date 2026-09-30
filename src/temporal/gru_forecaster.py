"""
PyTorch GRU Longitudinal Forecaster for 7-Day Banana Disease Progression (FR10).

Architecture:
- Input Features (5): [severity_score, urgency_score, scope_pct, impact_score, disease_id] normalized to [0, 1].
- GRU Layer: 32 hidden units, 1 recurrent layer, batch_first=True.
- Prediction Head: Linear(32, 16) -> ReLU -> Linear(16, 1) -> Sigmoid, scaled to [0.0, 100.0].
- Trained on plant-level partitioned longitudinal cohorts to prevent data leakage.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.schemas.contracts import DiseaseType, SeverityCategory
from src.data.dataset import plant_level_split
from src.severity.impact_calculators import calculate_disease_impact
from src.severity.fusion_strategies import map_score_to_category, fusion_damped_hybrid
from src.severity.xgboost_engine import DISEASE_TO_ID


class SeverityGRU(nn.Module):
    """
    Recurrent neural network with GRU cell for 7-day severity progression forecasting.
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
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Guarantees normalized output in [0.0, 1.0]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, seq_len, input_size) with values in [0.0, 1.0].

        Returns:
            Normalized predictions of shape (B, 1) in [0.0, 1.0].
        """
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]  # (B, hidden_size)
        pred_norm = self.head(last_hidden)  # (B, 1)
        return pred_norm


class SequenceDataset(Dataset):
    """Dataset yielding (input_sequence_tensor, target_scalar_tensor)."""

    def __init__(self, sequences: List[np.ndarray], targets: List[float]):
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1) / 100.0

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def build_sequences_from_df(
    df: pd.DataFrame,
    min_history: int = 2
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Extracts sliding temporal sequences from plant inspection rows.
    """
    sequences = []
    targets = []

    # Ensure chronological order by available sorting column
    sort_cols = ["plant_id"]
    if "day_index" in df.columns:
        sort_cols.append("day_index")
    elif "timepoint_idx" in df.columns:
        sort_cols.append("timepoint_idx")
    elif "observation_date" in df.columns:
        sort_cols.append("observation_date")

    df_sorted = df.sort_values(by=sort_cols).reset_index(drop=True)

    for plant_id, group in df_sorted.groupby("plant_id"):
        rows = group.to_dict(orient="records")
        n_obs = len(rows)
        if n_obs < min_history + 1:
            continue

        feature_list = []
        severity_list = []

        for r in rows:
            dis_str = str(r["disease_label"]).lower()
            dis_id = float(DISEASE_TO_ID.get(dis_str, 4)) / 4.0
            urg = float(r.get("urgency_score", 0.0))
            scope = float(r.get("ground_truth_scope_pct", r.get("scope_pct", 0.0)))
            aspect = float(r.get("ground_truth_aspect_ratio", r.get("aspect_ratio", 3.5)))
            num_les = int(r.get("num_lesions", 0))
            coal_ratio = float(r.get("coalescence_ratio_pct", r.get("coalescence_ratio", 0.0)))
            d_morph = float(np.clip((3.5 - aspect) / (3.5 - 1.5) * 100.0, 0.0, 100.0))

            dis_enum = DiseaseType(dis_str) if dis_str in [e.value for e in DiseaseType] else DiseaseType.HEALTHY
            impact, _ = calculate_disease_impact(dis_enum, scope, num_les, coal_ratio, d_morph)

            if "severity_score" in r and not pd.isna(r["severity_score"]) and float(r["severity_score"]) > 0.0:
                sev = float(r["severity_score"])
            else:
                sev = fusion_damped_hybrid(impact, scope, urg)

            severity_list.append(sev)
            vec = [sev / 100.0, urg / 100.0, scope / 100.0, impact / 100.0, dis_id]
            feature_list.append(vec)

        # Sliding window: sequences of length 2..n-1 predicting the next observation
        for k in range(min_history, n_obs):
            seq = np.array(feature_list[:k], dtype=np.float32)
            target = severity_list[k]
            sequences.append(seq)
            targets.append(target)

    return sequences, targets


def pad_collate_fn(batch):
    """Pads sequences of variable length in the batch."""
    sequences, targets = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    batch_size = len(sequences)
    input_size = sequences[0].shape[1]

    padded_seqs = torch.zeros(batch_size, max_len, input_size, dtype=torch.float32)
    for i, seq in enumerate(sequences):
        padded_seqs[i, :len(seq), :] = seq

    targets_tensor = torch.stack(targets, dim=0)
    return padded_seqs, targets_tensor


def train_gru(
    metadata_path: str = "data/synthetic/metadata.csv",
    output_weights_path: str = "results/gru_forecaster.pt",
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 0.002,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Trains the SeverityGRU on plant-level partitioned longitudinal sequences.
    """
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

    model = SeverityGRU(input_size=5, hidden_size=32, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(output_weights_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for seqs, targets in train_loader:
            optimizer.zero_grad()
            preds = model(seqs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for seqs, targets in val_loader:
                preds = model(seqs)
                loss = criterion(preds, targets)
                val_losses.append(loss.item())

        mean_val_loss = float(np.mean(val_losses))
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mse": best_val_loss
            }, output_weights_path)

    # Evaluate on test set
    checkpoint = torch.load(output_weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds = []
    test_actuals = []
    with torch.no_grad():
        for seqs, targets in test_loader:
            preds = model(seqs)
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


class GRUProgressionForecaster:
    """
    Operational inference engine for GRU-based 7-day progression forecasting.
    """

    def __init__(self, model_path: Optional[str] = "results/gru_forecaster.pt"):
        self.model = SeverityGRU(input_size=5, hidden_size=32, num_layers=1)
        self.is_loaded = False
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.is_loaded = True

    def predict_7d(
        self,
        historical_records: List[Dict[str, Any]]
    ) -> Tuple[float, SeverityCategory]:
        """
        Forecasts severity score in 7 days based on plant history.

        Args:
            historical_records: Chronologically sorted list of dicts with keys:
              severity_score, urgency_score, scope_pct, impact_score, disease_label.

        Returns:
            (forecast_7d_score, forecast_7d_category)
        """
        if not historical_records:
            return 0.0, SeverityCategory.LOW

        if not self.is_loaded:
            # Fallback to last known severity score if weights not yet loaded
            last_score = float(historical_records[-1].get("severity_score", 0.0))
            return round(last_score, 2), map_score_to_category(last_score)

        features = []
        for r in historical_records:
            dis_str = str(r.get("disease_label", "healthy")).lower()
            dis_id = float(DISEASE_TO_ID.get(dis_str, 4)) / 4.0
            urg = float(r.get("urgency_score", 0.0))
            scope = float(r.get("ground_truth_scope_pct", r.get("scope_pct", 0.0)))
            aspect = float(r.get("ground_truth_aspect_ratio", r.get("aspect_ratio", 3.5)))
            num_les = int(r.get("num_lesions", 0))
            coal_ratio = float(r.get("coalescence_ratio_pct", r.get("coalescence_ratio", 0.0)))
            d_morph = float(np.clip((3.5 - aspect) / (3.5 - 1.5) * 100.0, 0.0, 100.0))

            dis_enum = DiseaseType(dis_str) if dis_str in [e.value for e in DiseaseType] else DiseaseType.HEALTHY
            impact, _ = calculate_disease_impact(dis_enum, scope, num_les, coal_ratio, d_morph)

            if "severity_score" in r and not pd.isna(r["severity_score"]) and float(r["severity_score"]) > 0.0:
                sev = float(r["severity_score"])
            else:
                sev = fusion_damped_hybrid(impact, scope, urg)

            features.append([sev / 100.0, urg / 100.0, scope / 100.0, impact / 100.0, dis_id])

        seq_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 5)

        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(seq_tensor).item()  # [0.0, 1.0]

        forecast_score = round(float(np.clip(pred_norm * 100.0, 0.0, 100.0)), 2)
        forecast_cat = map_score_to_category(forecast_score)
        return forecast_score, forecast_cat
