"""
Centralized Configuration Loader for Component 3.

Loads parameters from configs/config.yaml, providing typed lookups with
resilient fallbacks for standalone script execution and production deployment.
"""

import os
from typing import Any, Dict, Optional
import yaml

_CACHED_CONFIG: Optional[Dict[str, Any]] = None
_DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config.yaml")
)


def load_yaml_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads YAML configuration from disk with caching.

    Args:
        config_path: Path to YAML config file. Defaults to configs/config.yaml.

    Returns:
        Dictionary of loaded configuration parameters.
    """
    global _CACHED_CONFIG
    path = config_path or _DEFAULT_CONFIG_PATH

    if not os.path.exists(path):
        # Return fallback configuration dictionary
        return _get_fallback_defaults()

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            _CACHED_CONFIG = cfg
            return cfg
    except Exception as exc:
        print(f"[Config] Warning: Failed to parse {path}: {exc}. Using fallback defaults.")
        return _get_fallback_defaults()


def get_config(key_path: Optional[str] = None, default: Any = None) -> Any:
    """
    Retrieves a nested configuration value using dot notation (e.g. 'fusion.damped_hybrid.w_impact').

    Args:
        key_path: Dot-delimited path or None to return full configuration dictionary.
        default: Default fallback value if the key is not found.

    Returns:
        The configuration value or default.
    """
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None:
        _CACHED_CONFIG = load_yaml_config()

    if key_path is None:
        return _CACHED_CONFIG

    keys = key_path.split(".")
    curr: Any = _CACHED_CONFIG
    for k in keys:
        if isinstance(curr, dict) and k in curr:
            curr = curr[k]
        else:
            return default
    return curr


def _get_fallback_defaults() -> Dict[str, Any]:
    """Provides complete fallback parameters if config.yaml is absent."""
    return {
        "seed": 42,
        "image_size": [256, 256],
        "severity_thresholds": {
            "low_max": 20.0,
            "moderate_max": 50.0,
            "high_max": 75.0,
        },
        "fusion": {
            "weighted_sum": {
                "w_impact": 0.45,
                "w_scope": 0.35,
                "w_urgency": 0.20,
            },
            "damped_hybrid": {
                "w_impact": 0.55,
                "w_scope": 0.45,
                "env_floor": 0.70,
                "env_scale": 0.30,
            },
        },
        "trend_tolerance_band_7d": 3.0,
        "gru": {
            "input_size": 5,
            "hidden_size": 48,
            "num_layers": 1,
            "learning_rate": 0.002,
            "epochs": 50,
            "batch_size": 16,
        },
        "xgboost": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
        "synthetic": {
            "expert_noise_sigma": 4.2,
        },
    }
