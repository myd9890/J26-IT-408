"""
Agro-Expert Ground-Truth Visual Assessment Simulator & Correlation Evaluator.

Simulates human agronomic expert visual severity estimation error:
- Gaussian visual noise with standard deviation sigma = 4.2% (matching ~71% inter-rater agreement in literature).
- Multi-expert consensus scoring across independent raters.
- Computes Spearman rank correlation coefficient, Pearson correlation, and Lin's Concordance Correlation Coefficient (CCC).
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy import stats


class AgroExpertSimulator:
    """
    Simulates visual estimation variance of field agronomists evaluating banana foliar severity.
    """

    def __init__(self, sigma_noise: float = 4.2, seed: int = 42):
        """
        Args:
            sigma_noise: Standard deviation of human visual estimation error in percentage points (default 4.2%).
            seed: Random seed for reproducibility.
        """
        self.sigma_noise = sigma_noise
        self.rng = np.random.default_rng(seed)

    def simulate_expert_rating(self, true_severity: float) -> float:
        """
        Generates a single human visual inspection estimate.

        Args:
            true_severity: True physical foliar severity in [0.0, 100.0].

        Returns:
            Simulated visual rating bounded in [0.0, 100.0].
        """
        noise = self.rng.normal(0.0, self.sigma_noise)
        rating = float(np.clip(true_severity + noise, 0.0, 100.0))
        return round(rating, 2)

    def simulate_panel_ratings(
        self,
        true_severities: List[float],
        num_experts: int = 3
    ) -> Dict[str, Any]:
        """
        Simulates an expert evaluation panel rating a list of banana leaf observations.

        Returns:
            Dictionary containing individual expert ratings, panel consensus (mean), and standard deviation.
        """
        ratings = []
        for _ in range(num_experts):
            r_expert = [self.simulate_expert_rating(y) for y in true_severities]
            ratings.append(r_expert)

        ratings_matrix = np.array(ratings, dtype=np.float64)  # Shape: (num_experts, N)
        consensus_mean = np.mean(ratings_matrix, axis=0)
        consensus_std = np.std(ratings_matrix, axis=0)

        return {
            "individual_expert_ratings": ratings,
            "consensus_mean": np.round(consensus_mean, 2).tolist(),
            "consensus_std": np.round(consensus_std, 2).tolist(),
        }

    @staticmethod
    def evaluate_correlation(
        predictions: List[float],
        ground_truth: List[float]
    ) -> Dict[str, float]:
        """
        Computes Spearman rank correlation, Pearson correlation, RMSE, and MAE.
        """
        preds = np.array(predictions, dtype=np.float64)
        targets = np.array(ground_truth, dtype=np.float64)

        if len(preds) < 2:
            return {"spearman_rho": 1.0, "pearson_r": 1.0, "rmse": 0.0, "mae": 0.0}

        spearman_rho, spearman_p = stats.spearmanr(preds, targets)
        pearson_r, pearson_p = stats.pearsonr(preds, targets)

        rmse = float(np.sqrt(np.mean((targets - preds) ** 2)))
        mae = float(np.mean(np.abs(targets - preds)))

        # Lin's Concordance Correlation Coefficient (CCC)
        mean_p, mean_t = np.mean(preds), np.mean(targets)
        var_p, var_t = np.var(preds), np.var(targets)
        cov = np.mean((preds - mean_p) * (targets - mean_t))
        ccc = (2.0 * cov) / (var_p + var_t + (mean_p - mean_t) ** 2 + 1e-8)

        return {
            "spearman_rho": round(float(spearman_rho), 4),
            "pearson_r": round(float(pearson_r), 4),
            "concordance_cc": round(float(ccc), 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4)
        }
