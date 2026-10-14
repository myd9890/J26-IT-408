
"""Validated inputs accepted by the optimization system."""

from pydantic import BaseModel, Field


class ZoneInput(BaseModel):
    """Disease prediction and treatment requirements for one zone."""

    zone_id: str
    disease: str

    severity: float = Field(ge=0, le=100)
    risk: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    urgency: float = Field(ge=0, le=100)

    chemical: float = Field(ge=0)
    labour: float = Field(ge=0)
    cost: float = Field(ge=0)


class ResourceInput(BaseModel):
    """Resources available for one optimization run."""

    chemical: float = Field(ge=0)
    labour: float = Field(ge=0)
    budget: float = Field(ge=0)
    max_visits: int = Field(ge=0)


class OptimizationRequest(BaseModel):
    """Complete input for generating a treatment allocation plan."""

    zones: list[ZoneInput] = Field(min_length=1)
    resources: ResourceInput
