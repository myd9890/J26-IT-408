"""
FastAPI Microservice for Multimodal Disease Severity & Progression Forecasting (FR11).

Exposes RESTful endpoints:
- POST /api/v1/assess-and-forecast: End-to-end diagnosis & forecast producing Component4Output.
- GET  /api/v1/plant-history/{plant_id}: Chronological inspection timeline for a given plant.
- GET  /health: System operational health and model readiness check.
"""

from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.schemas.contracts import (
    Component1Input,
    Component2Input,
    Component4Output,
)
from src.pipeline.c3_orchestrator import Component3Orchestrator


app = FastAPI(
    title="Component 3: Multimodal Severity Assessment & Temporal Forecasting API",
    description="Multimodal microservice assessing foliar severity, tracking longitudinal history, and forecasting 7-day progression for Cavendish banana plantations.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator singleton
_orchestrator: Optional[Component3Orchestrator] = None


def get_orchestrator() -> Component3Orchestrator:
    """Returns singleton orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Component3Orchestrator()
    return _orchestrator


class AssessmentRequest(BaseModel):
    """Request payload combining Component 1 vision diagnosis and Component 2 microclimate."""
    vision_input: Component1Input = Field(..., description="Component 1 Vision Diagnosis payload")
    environmental_input: Component2Input = Field(..., description="Component 2 Microclimate Urgency payload")
    fusion_method: Optional[str] = Field("F_hybrid", description="Fusion strategy: F_hybrid, F_WS, F_mult, or F_XGB")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    """Healthcheck endpoint verifying service status and model readiness."""
    orch = get_orchestrator()
    return {
        "status": "ok",
        "service": "Component 3 Multimodal Severity Engine",
        "version": "1.0.0",
        "models_loaded": {
            "unet": orch.unet is not None,
            "xgboost": orch.xgb_engine.model is not None,
            "gru": orch.gru_forecaster.is_loaded
        }
    }


@app.post(
    "/api/v1/assess-and-forecast",
    response_model=Component4Output,
    status_code=status.HTTP_200_OK
)
def assess_and_forecast_endpoint(payload: AssessmentRequest) -> Component4Output:
    """
    Assesses foliar severity from multimodal inputs and forecasts 7-day disease progression.
    Returns validated Component4Output payload for precision intervention.
    """
    orch = get_orchestrator()
    try:
        output = orch.assess_and_forecast(
            c1_input=payload.vision_input,
            c2_input=payload.environmental_input,
            fusion_method=payload.fusion_method
        )
        return output
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}"
        )


@app.get("/api/v1/plant-history/{plant_id}", status_code=status.HTTP_200_OK)
def get_plant_history_endpoint(plant_id: str) -> Dict[str, Any]:
    """Retrieves full longitudinal inspection history and current status for a plant."""
    orch = get_orchestrator()
    history = orch.db.get_plant_history(plant_id)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No inspection records found for plant_id: {plant_id}"
        )

    return {
        "plant_id": plant_id,
        "total_inspections": len(history),
        "history": history
    }
