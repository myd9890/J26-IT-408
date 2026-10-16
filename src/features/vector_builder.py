"""
Feature Vector & Sliding Temporal Sequence Builder (FR6, FR10).

Constructs:
1. 8-dimensional tabular feature vector for XGBoost fusion:
   x = [impact_score, scope_pct, urgency_score, disease_id,
        num_lesions, coalescence_ratio_pct, morph_distortion_idx, leaf_aspect_ratio]
2. 5-dimensional sliding temporal sequences for PyTorch GRU progression forecasting:
   s_t = [severity_score, urgency_score, scope_pct, impact_score, disease_id]
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.schemas.contracts import DiseaseType
from src.features.impact_scales import calculate_disease_impact
from src.models.severity.fusion_rules import fusion_damped_hybrid


FEATURE_NAMES = [
    "impact_score",
    "scope_pct",
    "urgency_score",
    "disease_id",
    "num_lesions",
    "coalescence_ratio_pct",
    "morph_distortion_idx",
    "leaf_aspect_ratio"
]

DISEASE_TO_ID: Dict[str, int] = {
    DiseaseType.BLACK_SIGATOKA.value: 0,
    DiseaseType.YELLOW_SIGATOKA.value: 1,
    DiseaseType.CORDANA.value: 2,
    DiseaseType.BBTV.value: 3,
    DiseaseType.HEALTHY.value: 4,
}


def assemble_feature_vector(
    impact_score: float,
    scope_pct: float,
    urgency_score: float,
    disease: Union[DiseaseType, str],
    num_lesions: int = 0,
    coalescence_ratio_pct: float = 0.0,
    morph_distortion_idx: float = 0.0,
    leaf_aspect_ratio: float = 3.5
) -> np.ndarray:
    """
    Constructs the 8-dimensional feature vector for XGBoost fusion.

    Args:
        impact_score: Biological impact score in [0.0, 100.0].
        scope_pct: Foliar scope coverage in [0.0, 100.0].
        urgency_score: Environmental transmission urgency in [0.0, 100.0].
        disease: DiseaseType enum or string representation.
        num_lesions: Count of lesion components.
        coalescence_ratio_pct: Area percentage of largest single lesion.
        morph_distortion_idx: Lamina bunching distortion index.
        leaf_aspect_ratio: Minimum-area bounding box length-to-width ratio.

    Returns:
        1D numpy array of shape (8,) with float32 values.
    """
    dis_str = disease.value if isinstance(disease, DiseaseType) else str(disease).lower()
    disease_id = float(DISEASE_TO_ID.get(dis_str, 4))

    vec = np.array([
        float(np.clip(impact_score, 0.0, 100.0)),
        float(np.clip(scope_pct, 0.0, 100.0)),
        float(np.clip(urgency_score, 0.0, 100.0)),
        disease_id,
        float(max(0, num_lesions)),
        float(np.clip(coalescence_ratio_pct, 0.0, 100.0)),
        float(np.clip(morph_distortion_idx, 0.0, 100.0)),
        float(max(1.0, leaf_aspect_ratio))
    ], dtype=np.float32)

    return vec


def extract_features_from_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 8-D feature matrix X and target severity array y from dataset DataFrame.
    """
    X_rows = []
    y_vals = []

    for row_idx, row in df.iterrows():
        disease_str = str(row["disease_label"]).lower()
        try:
            disease_enum = DiseaseType(disease_str)
        except ValueError:
            disease_enum = DiseaseType.HEALTHY

        scope_pct = float(row.get("ground_truth_scope_pct", row.get("scope_pct", 0.0)))
        urgency_score = float(row.get("urgency_score", 0.0))
        num_lesions = int(row.get("num_lesions", 0))
        coalescence_ratio = float(row.get("coalescence_ratio_pct", row.get("coalescence_ratio", 0.0)))
        aspect_ratio = float(row.get("ground_truth_aspect_ratio", row.get("aspect_ratio", 3.5)))

        # Compute morph_distortion_idx
        d_morph = float(np.clip((3.5 - aspect_ratio) / (3.5 - 1.5) * 100.0, 0.0, 100.0))

        # Biological impact
        impact, _ = calculate_disease_impact(
            disease=disease_enum,
            scope_pct=scope_pct,
            num_lesions=num_lesions,
            coalescence_ratio_pct=coalescence_ratio,
            morph_distortion_idx=d_morph
        )

        vec = assemble_feature_vector(
            impact_score=impact,
            scope_pct=scope_pct,
            urgency_score=urgency_score,
            disease=disease_enum,
            num_lesions=num_lesions,
            coalescence_ratio_pct=coalescence_ratio,
            morph_distortion_idx=d_morph,
            leaf_aspect_ratio=aspect_ratio
        )
        X_rows.append(vec)

        # Target severity: use ground truth if present; otherwise simulate expert inspection rating
        if "severity_score" in row and not pd.isna(row["severity_score"]) and float(row["severity_score"]) > 0.0:
            target_sev = float(row["severity_score"])
        elif "expert_severity" in row and not pd.isna(row["expert_severity"]) and float(row["expert_severity"]) > 0.0:
            target_sev = float(row["expert_severity"])
        else:
            from src.evaluation.expert_simulator import AgroExpertSimulator
            base_sev = fusion_damped_hybrid(impact, scope_pct, urgency_score)
            s_idx = int(row_idx) if isinstance(row_idx, (int, np.integer)) else 42
            sim = AgroExpertSimulator(sigma_noise=4.2, seed=42 + (s_idx % 1000))
            target_sev = sim.simulate_expert_rating(base_sev)
        y_vals.append(target_sev)

    X = np.vstack(X_rows).astype(np.float32)
    y = np.array(y_vals, dtype=np.float32)
    return X, y


def build_sequences_from_df(
    df: pd.DataFrame,
    min_history: int = 2
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Extracts sliding temporal sequences from plant inspection rows for GRU training.
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
            elif "expert_severity" in r and not pd.isna(r["expert_severity"]) and float(r["expert_severity"]) > 0.0:
                sev = float(r["expert_severity"])
            else:
                from src.evaluation.expert_simulator import AgroExpertSimulator
                base_sev = fusion_damped_hybrid(impact, scope, urg)
                sim = AgroExpertSimulator(sigma_noise=4.2, seed=int(r.get("timepoint_idx", 0)) + 42)
                sev = sim.simulate_expert_rating(base_sev)

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
