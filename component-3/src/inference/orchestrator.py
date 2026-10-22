"""
Master Component 3 Orchestrator (FR1–FR11).

Integrates:
- Component 1 Vision Diagnosis input & Component 2 Microclimate Urgency input
- Semantic segmentation (U-Net 3-class / Classical baseline fallback)
- Foliar morphometrics & scope calculation (FR3)
- Disease-specific impact scoring across standard scales (FR4)
- Multimodal fusion (F_hybrid, F_WS, F_mult, F_XGB) (FR6)
- Severity categorization into validated bands (FR7)
- SQLite longitudinal persistence (FR8)
- OLS progression trend classification with cold-start handling (FR9)
- PyTorch GRU 7-day progression forecasting (FR10)
- Decision-ready payload generation for Component 4 (FR11)
"""

import os
from typing import Optional, Dict, Any, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from src.schemas.contracts import (
    Component1Input,
    Component2Input,
    Component4Output,
    DiseaseType,
    SeverityCategory,
    ProgressionTrend,
)
from src.models.segmentation.unet import UNet
from src.models.segmentation.classical import ClassicalSegmenter
from src.features.morphometrics import extract_leaf_and_lesion_features, MorphometricFeatures
from src.features.impact_scales import calculate_disease_impact
from src.features.vector_builder import assemble_feature_vector
from src.models.severity.fusion_rules import (
    fusion_multiplicative,
    fusion_weighted_sum,
    fusion_damped_hybrid,
    map_score_to_category,
    get_recommended_action,
)
from src.models.severity.xgboost_regressor import XGBoostSeverityEngine
from src.inference.database import SeverityDatabase, DEFAULT_DB_PATH
from src.models.temporal.trend_ols import calculate_ols_trend
from src.inference.predictors import GRUProgressionForecaster


class Component3Orchestrator:
    """
    End-to-end multimodal disease severity assessment and forecasting engine.
    """

    def __init__(
        self,
        unet_weights_path: Optional[str] = None,
        xgboost_model_path: Optional[str] = "results/xgboost_model.json",
        gru_weights_path: Optional[str] = "results/gru_forecaster.pt",
        db_path: str = DEFAULT_DB_PATH,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Segmentation model: prefer real weights if available, otherwise synthetic
        if unet_weights_path is None:
            if os.path.exists("results/unet_real_weights.pt"):
                unet_weights_path = "results/unet_real_weights.pt"
            elif os.path.exists("results/unet_weights.pt"):
                unet_weights_path = "results/unet_weights.pt"

        self.unet: Optional[UNet] = None
        self.classical_baseline = ClassicalSegmenter()

        if unet_weights_path and os.path.exists(unet_weights_path):
            try:
                ckpt = torch.load(unet_weights_path, map_location=self.device, weights_only=False)
                base_filters = ckpt.get("base_filters", 16)
                self.unet = UNet(n_channels=3, n_classes=3, base_filters=base_filters).to(self.device)
                self.unet.load_state_dict(ckpt["model_state_dict"])
                self.unet.eval()
            except Exception as e:
                print(f"[Warning] Failed to load U-Net checkpoint: {e}. Fallback to classical baseline.")
                self.unet = None

        # 2. XGBoost fusion engine
        self.xgb_engine = XGBoostSeverityEngine(model_path=xgboost_model_path)

        # 3. SQLite temporal database
        self.db = SeverityDatabase(db_path=db_path)

        # 4. GRU longitudinal forecaster
        self.gru_forecaster = GRUProgressionForecaster(model_path=gru_weights_path)

    def _load_image(self, image_path: str) -> np.ndarray:
        """Loads and resolves image to 256x256 RGB numpy array."""
        candidates = [
            image_path,
            os.path.join("data/real_seg", image_path),
            os.path.join("data/synthetic", image_path),
            os.path.join("data/processed", image_path),
        ]
        resolved_path = None
        for cand in candidates:
            if os.path.exists(cand):
                resolved_path = cand
                break

        if resolved_path is None:
            raise FileNotFoundError(f"Image could not be resolved from path: {image_path}")

        with Image.open(resolved_path) as img:
            rgb_img = img.convert("RGB")
            if rgb_img.size != (256, 256):
                rgb_img = rgb_img.resize((256, 256), Image.Resampling.LANCZOS)
            return np.array(rgb_img, dtype=np.uint8)

    def _predict_mask(self, rgb_image: np.ndarray) -> np.ndarray:
        """Predicts 3-class segmentation mask using U-Net or classical fallback."""
        if self.unet is not None:
            img_tensor = TF.to_tensor(rgb_image).unsqueeze(0).to(self.device)  # (1, 3, 256, 256)
            with torch.no_grad():
                pred_tensor = self.unet.predict_mask(img_tensor).squeeze(0).cpu()
                return pred_tensor.numpy().astype(np.uint8)
        else:
            return self.classical_baseline.predict(rgb_image)

    def assess_and_forecast(
        self,
        c1_input: Component1Input,
        c2_input: Component2Input,
        fusion_method: str = "F_hybrid"
    ) -> Component4Output:
        """
        Executes complete multimodal pipeline from C1 and C2 inputs to C4 output.

        Args:
            c1_input: Validated vision diagnosis payload.
            c2_input: Validated microclimate urgency payload.
            fusion_method: 'F_hybrid', 'F_WS', 'F_mult', or 'F_XGB'.

        Returns:
            Validated Component4Output payload with 17 decision fields.
        """
        # Step 1: Load image and predict 3-class segmentation mask (FR2)
        rgb_image = self._load_image(c1_input.image_path)
        pred_mask = self._predict_mask(rgb_image)

        # Step 2: Extract morphometric features and foliar scope (FR3)
        morph_feats = extract_leaf_and_lesion_features(pred_mask)

        # Step 3: Disease-specific impact scoring (FR4)
        impact_score, stover_grade = calculate_disease_impact(
            disease=c1_input.disease_label,
            scope_pct=morph_feats.scope_pct,
            num_lesions=morph_feats.num_lesions,
            coalescence_ratio_pct=morph_feats.coalescence_ratio_pct,
            morph_distortion_idx=morph_feats.morph_distortion_idx
        )

        # Step 4: Multimodal fusion (FR6)
        urgency = c2_input.urgency_score

        if fusion_method == "F_mult":
            severity_score = fusion_multiplicative(impact_score, urgency, morph_feats.scope_pct)
        elif fusion_method == "F_WS":
            severity_score = fusion_weighted_sum(impact_score, morph_feats.scope_pct, urgency)
        elif fusion_method == "F_XGB" and self.xgb_engine.model is not None:
            feature_vec = assemble_feature_vector(
                impact_score=impact_score,
                scope_pct=morph_feats.scope_pct,
                urgency_score=urgency,
                disease=c1_input.disease_label,
                num_lesions=morph_feats.num_lesions,
                coalescence_ratio_pct=morph_feats.coalescence_ratio_pct,
                morph_distortion_idx=morph_feats.morph_distortion_idx,
                leaf_aspect_ratio=morph_feats.leaf_aspect_ratio
            )
            severity_score = self.xgb_engine.predict(feature_vec)
        else:
            # Default operational formula: Damped Multiplicative Hybrid
            severity_score = fusion_damped_hybrid(impact_score, morph_feats.scope_pct, urgency)
            fusion_method = "F_hybrid"

        # Step 5: Severity categorization & Agronomic prescription (FR7)
        severity_category = map_score_to_category(severity_score)
        recommended_action = get_recommended_action(severity_category, c1_input.disease_label)

        # Step 6: Log observation to SQLite persistence store (FR8)
        self.db.insert_record(
            plant_id=c1_input.plant_id,
            zone_id=c1_input.zone_id,
            leaf_id=c1_input.leaf_id,
            observation_date=c1_input.observation_date,
            disease_label=c1_input.disease_label.value,
            scope_pct=morph_feats.scope_pct,
            impact_score=impact_score,
            stover_grade=stover_grade,
            urgency_score=urgency,
            severity_score=severity_score,
            severity_category=severity_category.value,
            fusion_method=fusion_method
        )

        # Step 7: Temporal trend analysis & 7-day forecasting (FR9, FR10)
        history = self.db.get_plant_history(c1_input.plant_id)
        dates = [r["observation_date"] for r in history]
        severities = [float(r["severity_score"]) for r in history]

        trend_results = calculate_ols_trend(dates, severities)
        progression_trend = trend_results["progression_trend"]
        weekly_rate = trend_results["weekly_rate_of_change"]
        confidence_flag = trend_results["confidence_flag"]

        # 7-day forecast determination
        if len(history) == 1:
            forecast_7d_score = None
            forecast_7d_category = None
        elif len(history) == 2:
            forecast_7d_score = trend_results["forecast_7d_score"]
            forecast_7d_category = trend_results["forecast_7d_category"]
        else:
            # n >= 3: PyTorch GRU progression forecaster
            forecast_7d_score, forecast_7d_category = self.gru_forecaster.predict_7d(history)

        # Step 8: Return validated Component4Output payload (FR11)
        output = Component4Output(
            plant_id=c1_input.plant_id,
            zone_id=c1_input.zone_id,
            observation_date=c1_input.observation_date,
            disease_label=c1_input.disease_label,
            scope_pct=morph_feats.scope_pct,
            impact_score=impact_score,
            stover_grade=stover_grade,
            urgency_score=urgency,
            severity_score=severity_score,
            severity_category=severity_category,
            fusion_method_used=fusion_method,
            progression_trend=progression_trend,
            weekly_rate_of_change=weekly_rate,
            forecast_7d_score=forecast_7d_score,
            forecast_7d_category=forecast_7d_category,
            confidence_flag=confidence_flag,
            recommended_action=recommended_action
        )

        return output

    def assess_interactive(
        self,
        c1_input: Component1Input,
        c2_input: Component2Input,
        fusion_method: str = "F_hybrid"
    ) -> Dict[str, Any]:
        """
        Executes complete multimodal pipeline and returns rich visual assets (base64 image & mask),
        morphometrics, multi-fusion strategy comparison, and longitudinal trajectory for UI display.
        """
        import io
        import base64

        # Step 1: Load image and predict mask
        rgb_image = self._load_image(c1_input.image_path)
        pred_mask = self._predict_mask(rgb_image)

        # Step 2: Extract morphometric features
        morph_feats = extract_leaf_and_lesion_features(pred_mask)

        # Step 3: Disease-specific impact scoring
        impact_score, stover_grade = calculate_disease_impact(
            disease=c1_input.disease_label,
            scope_pct=morph_feats.scope_pct,
            num_lesions=morph_feats.num_lesions,
            coalescence_ratio_pct=morph_feats.coalescence_ratio_pct,
            morph_distortion_idx=morph_feats.morph_distortion_idx
        )

        # Step 4: Multimodal fusion and comparative evaluation across all 4 formulas
        urgency = c2_input.urgency_score
        f_hybrid = fusion_damped_hybrid(impact_score, morph_feats.scope_pct, urgency)
        f_ws = fusion_weighted_sum(impact_score, morph_feats.scope_pct, urgency)
        f_mult = fusion_multiplicative(impact_score, urgency, morph_feats.scope_pct)

        feature_vec = assemble_feature_vector(
            impact_score=impact_score,
            scope_pct=morph_feats.scope_pct,
            urgency_score=urgency,
            disease=c1_input.disease_label,
            num_lesions=morph_feats.num_lesions,
            coalescence_ratio_pct=morph_feats.coalescence_ratio_pct,
            morph_distortion_idx=morph_feats.morph_distortion_idx,
            leaf_aspect_ratio=morph_feats.leaf_aspect_ratio
        )
        if self.xgb_engine.model is not None:
            f_xgb = self.xgb_engine.predict(feature_vec)
        else:
            f_xgb = f_hybrid

        if fusion_method == "F_mult":
            severity_score = f_mult
        elif fusion_method == "F_WS":
            severity_score = f_ws
        elif fusion_method == "F_XGB":
            severity_score = f_xgb
        else:
            severity_score = f_hybrid
            fusion_method = "F_hybrid"

        # Step 5: Severity category & prescription
        severity_category = map_score_to_category(severity_score)
        recommended_action = get_recommended_action(severity_category, c1_input.disease_label)

        # Step 6: Log observation
        self.db.insert_record(
            plant_id=c1_input.plant_id,
            zone_id=c1_input.zone_id,
            leaf_id=c1_input.leaf_id,
            observation_date=c1_input.observation_date,
            disease_label=c1_input.disease_label.value,
            scope_pct=morph_feats.scope_pct,
            impact_score=impact_score,
            stover_grade=stover_grade,
            urgency_score=urgency,
            severity_score=severity_score,
            severity_category=severity_category.value,
            fusion_method=fusion_method
        )

        # Step 7: Temporal trajectory
        history = self.db.get_plant_history(c1_input.plant_id)
        dates = [r["observation_date"] for r in history]
        severities = [float(r["severity_score"]) for r in history]

        trend_results = calculate_ols_trend(dates, severities)
        progression_trend = trend_results["progression_trend"]
        weekly_rate = trend_results["weekly_rate_of_change"]
        confidence_flag = trend_results["confidence_flag"]

        if len(history) == 1:
            forecast_7d_score = None
            forecast_7d_category = None
        elif len(history) == 2:
            forecast_7d_score = trend_results["forecast_7d_score"]
            forecast_7d_category = trend_results["forecast_7d_category"]
        else:
            forecast_7d_score, forecast_7d_category = self.gru_forecaster.predict_7d(history)

        output = Component4Output(
            plant_id=c1_input.plant_id,
            zone_id=c1_input.zone_id,
            observation_date=c1_input.observation_date,
            disease_label=c1_input.disease_label,
            scope_pct=morph_feats.scope_pct,
            impact_score=impact_score,
            stover_grade=stover_grade,
            urgency_score=urgency,
            severity_score=severity_score,
            severity_category=severity_category,
            fusion_method_used=fusion_method,
            progression_trend=progression_trend,
            weekly_rate_of_change=weekly_rate,
            forecast_7d_score=forecast_7d_score,
            forecast_7d_category=forecast_7d_category,
            confidence_flag=confidence_flag,
            recommended_action=recommended_action
        )

        # Step 8: Encode base64 image and colorized mask
        pil_img = Image.fromarray(rgb_image)
        buf_img = io.BytesIO()
        pil_img.save(buf_img, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buf_img.getvalue()).decode("utf-8")

        # RGBA Colorized Mask
        mask_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        mask_rgba[pred_mask == 1] = [16, 185, 129, 170]  # Emerald green (healthy lamina)
        mask_rgba[pred_mask == 2] = [239, 68, 68, 220]   # Crimson/amber (lesions)
        pil_mask = Image.fromarray(mask_rgba, mode="RGBA")
        buf_mask = io.BytesIO()
        pil_mask.save(buf_mask, format="PNG")
        mask_b64 = base64.b64encode(buf_mask.getvalue()).decode("utf-8")

        return {
            "assessment": output.model_dump(),
            "image_base64": f"data:image/jpeg;base64,{img_b64}",
            "mask_base64": f"data:image/png;base64,{mask_b64}",
            "morphometrics": {
                "scope_pct": round(morph_feats.scope_pct, 2),
                "num_lesions": int(morph_feats.num_lesions),
                "coalescence_ratio_pct": round(morph_feats.coalescence_ratio_pct, 2),
                "leaf_aspect_ratio": round(morph_feats.leaf_aspect_ratio, 2),
                "morph_distortion_idx": round(morph_feats.morph_distortion_idx, 2)
            },
            "fusion_comparison": {
                "F_hybrid": round(float(f_hybrid), 2),
                "F_XGB": round(float(f_xgb), 2),
                "F_WS": round(float(f_ws), 2),
                "F_mult": round(float(f_mult), 2)
            },
            "history": history
        }
