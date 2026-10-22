"""
API Request and Response Data Transfer Objects (DTOs).
"""

from typing import Optional
from pydantic import BaseModel, Field
from src.schemas.contracts import Component1Input, Component2Input


class AssessmentRequest(BaseModel):
    """Request payload combining Component 1 vision diagnosis and Component 2 microclimate."""
    vision_input: Component1Input = Field(..., description="Component 1 Vision Diagnosis payload")
    environmental_input: Component2Input = Field(..., description="Component 2 Microclimate Urgency payload")
    fusion_method: Optional[str] = Field("F_hybrid", description="Fusion strategy: F_hybrid, F_WS, F_mult, or F_XGB")


class InteractiveAssessmentRequest(BaseModel):
    """Request payload for rich interactive demonstration with optional client-uploaded image."""
    vision_input: Component1Input = Field(..., description="Component 1 Vision Diagnosis payload")
    environmental_input: Component2Input = Field(..., description="Component 2 Microclimate Urgency payload")
    fusion_method: Optional[str] = Field("F_hybrid", description="Fusion strategy: F_hybrid, F_WS, F_mult, or F_XGB")
    image_base64: Optional[str] = Field(None, description="Optional raw base64 data URL for custom leaf image")


class ImageUploadPayload(BaseModel):
    """Payload for browser image upload encoded as base64."""
    image_base64: str = Field(..., description="Base64 encoded image data URL")
    filename: Optional[str] = Field("custom_leaf.jpg", description="Original filename")
