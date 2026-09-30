"""
OLS Progression Trend Classifier & Cold-Start Estimator (FR9).

Implements:
1. Ordinary Least Squares (OLS) linear regression estimating weekly rate of change:
   β_7d = 7.0 * (sum((t_i - t_bar)(Y_i - Y_bar)) / sum((t_i - t_bar)^2))
2. Three-state progression trajectory classification with ±3.0 points/7d band:
   - Worsening: β_7d > +3.0
   - Stable:    -3.0 <= β_7d <= +3.0
   - Improving: β_7d < -3.0
3. Cold-Start Reliability Protocol:
   - n = 1: ProgressionTrend.INITIAL_OBSERVATION, confidence="LOW_COLD_START", forecast_7d=None
   - n = 2: Two-point linear extrapolation Y_hat = clip(Y_2 + β_7d, 0, 100), confidence="MEDIUM"
   - n >= 3: Full OLS + GRU forecast pipeline, confidence="HIGH"
"""

from typing import List, Tuple, Optional, Dict, Any, Union
from datetime import datetime, date
import numpy as np

from src.schemas.contracts import ProgressionTrend, SeverityCategory
from src.severity.fusion_strategies import map_score_to_category


def parse_date(date_val: Union[str, datetime, date]) -> datetime:
    """Parses date string (YYYY-MM-DD) or datetime into a standard datetime object."""
    if isinstance(date_val, datetime):
        return date_val
    if isinstance(date_val, date):
        return datetime.combine(date_val, datetime.min.time())
    if isinstance(date_val, str):
        # Handle ISO strings
        clean_str = date_val.split("T")[0]
        return datetime.strptime(clean_str, "%Y-%m-%d")
    raise ValueError(f"Cannot parse date: {date_val}")


def calculate_ols_trend(
    dates: List[Union[str, datetime]],
    severity_scores: List[float],
    tolerance_band_7d: float = 3.0
) -> Dict[str, Any]:
    """
    Computes OLS slope over historical inspection dates and returns trend,
    weekly rate of change, and cold-start confidence status.

    Args:
        dates: List of observation dates in chronological order.
        severity_scores: Corresponding sequential severity scores in [0.0, 100.0].
        tolerance_band_7d: Weekly slope threshold epsilon defining Stable zone (default 3.0).

    Returns:
        Dictionary with keys:
          - progression_trend: ProgressionTrend enum member
          - weekly_rate_of_change: float (points/7d) or None
          - confidence_flag: "LOW_COLD_START", "MEDIUM", or "HIGH"
          - forecast_7d_score: Optional float (linear extrapolation for n=2)
          - forecast_7d_category: Optional SeverityCategory
    """
    n = len(severity_scores)
    if len(dates) != n:
        raise ValueError(f"Length mismatch: {len(dates)} dates vs {n} scores")

    if n == 0:
        return {
            "progression_trend": ProgressionTrend.INITIAL_OBSERVATION,
            "weekly_rate_of_change": None,
            "confidence_flag": "LOW_COLD_START",
            "forecast_7d_score": None,
            "forecast_7d_category": None
        }

    # -------------------------------------------------------------------------
    # n = 1: Initial Observation (Cold-Start)
    # -------------------------------------------------------------------------
    if n == 1:
        return {
            "progression_trend": ProgressionTrend.INITIAL_OBSERVATION,
            "weekly_rate_of_change": None,
            "confidence_flag": "LOW_COLD_START",
            "forecast_7d_score": None,
            "forecast_7d_category": None
        }

    # Convert dates to relative day offsets from the first date
    d0 = parse_date(dates[0])
    t_days = np.array([(parse_date(d) - d0).total_seconds() / 86400.0 for d in dates], dtype=np.float64)
    y_scores = np.array(severity_scores, dtype=np.float64)

    # -------------------------------------------------------------------------
    # n = 2: Two-Point Extrapolation
    # -------------------------------------------------------------------------
    if n == 2:
        dt = t_days[1] - t_days[0]
        if dt <= 0:
            dt = 7.0  # Fallback to weekly assumption if same-day

        daily_slope = (y_scores[1] - y_scores[0]) / dt
        beta_7d = round(float(daily_slope * 7.0), 2)

        if beta_7d > tolerance_band_7d:
            trend = ProgressionTrend.WORSENING
        elif beta_7d < -tolerance_band_7d:
            trend = ProgressionTrend.IMPROVING
        else:
            trend = ProgressionTrend.STABLE

        extrapolated = float(np.clip(y_scores[1] + beta_7d, 0.0, 100.0))
        cat = map_score_to_category(extrapolated)

        return {
            "progression_trend": trend,
            "weekly_rate_of_change": beta_7d,
            "confidence_flag": "MEDIUM",
            "forecast_7d_score": round(extrapolated, 2),
            "forecast_7d_category": cat
        }

    # -------------------------------------------------------------------------
    # n >= 3: Full Ordinary Least Squares (OLS)
    # -------------------------------------------------------------------------
    t_mean = np.mean(t_days)
    y_mean = np.mean(y_scores)

    numerator = np.sum((t_days - t_mean) * (y_scores - y_mean))
    denominator = np.sum((t_days - t_mean) ** 2)

    if denominator <= 1e-7:
        daily_slope = 0.0
    else:
        daily_slope = float(numerator / denominator)

    beta_7d = round(float(daily_slope * 7.0), 2)

    if beta_7d > tolerance_band_7d:
        trend = ProgressionTrend.WORSENING
    elif beta_7d < -tolerance_band_7d:
        trend = ProgressionTrend.IMPROVING
    else:
        trend = ProgressionTrend.STABLE

    return {
        "progression_trend": trend,
        "weekly_rate_of_change": beta_7d,
        "confidence_flag": "HIGH",
        "forecast_7d_score": None,     # Handled by GRU forecaster for n >= 3
        "forecast_7d_category": None
    }
