"""
OLS Progression Trend Classifier & Cold-Start Estimator (FR9).

(Compatibility Facade - logic moved to src.models.temporal.trend_ols)
"""

from src.models.temporal.trend_ols import calculate_ols_trend, parse_date

__all__ = ["calculate_ols_trend", "parse_date"]
