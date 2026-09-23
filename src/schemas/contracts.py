"""
Pydantic Data Contracts for Component 3.

Defines schemas and validated models for:
- Component 1 input (FR1)
- Component 2 input (FR5)
- Component 4 output (FR11)
- Core enumeration types
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import re


class DiseaseType(str, Enum):
    """Supported banana foliar disease types."""
    BLACK_SIGATOKA = "black_sigatoka"
    YELLOW_SIGATOKA = "yellow_sigatoka"
    CORDANA = "cordana"
    BBTV = "bbtv"
    HEALTHY = "healthy"


class SeverityCategory(str, Enum):
    """Categorical foliar severity classes."""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


class ProgressionTrend(str, Enum):
    """Temporal progression trajectory classes."""
    WORSENING = "Worsening"
    STABLE = "Stable"
    IMPROVING = "Improving"
    INITIAL_OBSERVATION = "INITIAL_OBSERVATION"


DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class Component1Input(BaseModel):
    """
    Contract for vision diagnosis payload from Component 1 (FR1).
    """
    plant_id: str = Field(..., description="Unique plant identifier, e.g., PLANT_001")
    zone_id: str = Field(..., description="Plantation management zone, e.g., ZONE_A")
    leaf_id: str = Field(..., description="Leaf specimen identifier, e.g., LEAF_01")
    observation_date: str = Field(..., description="Date of observation in ISO format (YYYY-MM-DD)")
    disease_label: DiseaseType = Field(..., description="Diagnosed disease classification")
    disease_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from vision model")
    image_path: str = Field(..., description="Path or URI to raw leaf image")

    @field_validator("observation_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if not DATE_REGEX.match(v):
            raise ValueError(f"observation_date must be in ISO format YYYY-MM-DD, got {v}")
        return v


class Component2Input(BaseModel):
    """
    Contract for environmental transmission risk payload from Component 2 (FR5).
    """
    zone_id: str = Field(..., description="Plantation management zone matching Component 1")
    observation_date: str = Field(..., description="Date of microclimate observation (YYYY-MM-DD)")
    urgency_score: float = Field(..., ge=0.0, le=100.0, description="Environmental disease urgency in [0.0, 100.0]")

    @field_validator("observation_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if not DATE_REGEX.match(v):
            raise ValueError(f"observation_date must be in ISO format YYYY-MM-DD, got {v}")
        return v


class Component4Output(BaseModel):
    """
    Contract for decision support and precision intervention payload to Component 4 (FR11).
    Contains full multimodal severity breakdown, longitudinal trend, and 7-day forecast.
    """
    plant_id: str = Field(..., description="Unique plant identifier")
    zone_id: str = Field(..., description="Plantation zone identifier")
    observation_date: str = Field(..., description="Date of observation (YYYY-MM-DD)")
    disease_label: DiseaseType = Field(..., description="Diagnosed disease")
    scope_pct: float = Field(..., ge=0.0, le=100.0, description="Foliar lesion coverage percentage")
    impact_score: float = Field(..., ge=0.0, le=100.0, description="Disease-specific impact score")
    stover_grade: Optional[int] = Field(None, ge=0, le=6, description="Discrete Stover-Gauhl grade (0-6) for Sigatoka")
    urgency_score: float = Field(..., ge=0.0, le=100.0, description="Component 2 environmental urgency score")
    severity_score: float = Field(..., ge=0.0, le=100.0, description="Fused composite severity score [0.0, 100.0]")
    severity_category: SeverityCategory = Field(..., description="Severity category (Low, Moderate, High, Critical)")
    fusion_method_used: str = Field(..., description="Fusion strategy identifier (e.g., F_hybrid, F_XGB)")
    progression_trend: ProgressionTrend = Field(..., description="Temporal progression trend")
    weekly_rate_of_change: Optional[float] = Field(None, description="Weekly OLS severity slope (points/7d)")
    forecast_7d_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Predicted severity in 7 days")
    forecast_7d_category: Optional[SeverityCategory] = Field(None, description="Predicted severity category in 7 days")
    confidence_flag: str = Field(..., description="Confidence rating (LOW_COLD_START, MEDIUM, HIGH)")
    recommended_action: str = Field(..., description="Agronomic action prescription for Component 4")
