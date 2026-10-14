import os
import io
import uuid
import base64
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from PIL import Image

from src.schemas.contracts import (
    Component1Input,
    Component2Input,
    Component4Output,
    DiseaseType,
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


class InteractiveAssessmentRequest(BaseModel):
    """Request payload for rich interactive demonstration with optional client-uploaded image."""
    vision_input: Component1Input = Field(..., description="Component 1 Vision Diagnosis payload")
    environmental_input: Component2Input = Field(..., description="Component 2 Microclimate Urgency payload")
    fusion_method: Optional[str] = Field("F_hybrid", description="Fusion strategy: F_hybrid, F_WS, F_mult, or F_XGB")
    image_base64: Optional[str] = Field(None, description="Optional raw base64 data URL for custom leaf image")


DEMO_PRESETS = [
    {
        "id": "bs_moderate",
        "name": "Black Sigatoka (Field Sample)",
        "badge": "Field Real",
        "disease_label": "black_sigatoka",
        "disease_confidence": 0.96,
        "plant_id": "REAL_PLANT_TR_001",
        "zone_id": "ZONE_A",
        "leaf_id": "LEAF_01",
        "observation_date": "2026-10-20",
        "image_path": "images/real_train_0001_172_jpeg.rf.94528c81ceec44f2e54c6a574fa3416a.jpg",
        "urgency_score": 75.0,
        "temperature_c": 28.5,
        "relative_humidity_pct": 88.0,
        "leaf_wetness_hours": 12.0,
        "consecutive_wet_days": 4,
        "description": "Active Black Sigatoka necrosis under humid tropical microclimate (75% urgency)."
    },
    {
        "id": "bs_severe",
        "name": "Severe Sigatoka (Grade 5)",
        "badge": "Extensive Necrosis",
        "disease_label": "black_sigatoka",
        "disease_confidence": 0.98,
        "plant_id": "REAL_PLANT_TR_003",
        "zone_id": "ZONE_B",
        "leaf_id": "LEAF_02",
        "observation_date": "2026-10-20",
        "image_path": "images/real_train_0003_436_jpeg.rf.94c742f939c7bea4faeedcd7b1b2ed7d.jpg",
        "urgency_score": 85.0,
        "temperature_c": 29.0,
        "relative_humidity_pct": 92.0,
        "leaf_wetness_hours": 14.0,
        "consecutive_wet_days": 5,
        "description": "Advanced necrotic coalesced lesions covering ~42% of leaf area under high infection pressure."
    },
    {
        "id": "cordana_sample",
        "name": "Cordana Leaf Spot",
        "badge": "Margin Necrosis",
        "disease_label": "cordana",
        "disease_confidence": 0.94,
        "plant_id": "REAL_PLANT_TR_006",
        "zone_id": "ZONE_A",
        "leaf_id": "LEAF_03",
        "observation_date": "2026-10-20",
        "image_path": "images/real_train_0006_103_jpeg.rf.94c2385f49d096edf7bbd687b485e1a8.jpg",
        "urgency_score": 60.0,
        "temperature_c": 26.5,
        "relative_humidity_pct": 82.0,
        "leaf_wetness_hours": 8.0,
        "consecutive_wet_days": 2,
        "description": "Distinct oval necrotic lesions with chlorotic halos along leaf margins."
    },
    {
        "id": "bbtv_sample",
        "name": "BBTV (Bunchy Top Stunting)",
        "badge": "Viral Stunting",
        "disease_label": "bbtv",
        "disease_confidence": 0.92,
        "plant_id": "PLANT_004",
        "zone_id": "ZONE_D",
        "leaf_id": "LEAF_04",
        "observation_date": "2026-10-20",
        "image_path": "images/PLANT_004_T0_D00.jpg",
        "urgency_score": 35.0,
        "temperature_c": 27.0,
        "relative_humidity_pct": 65.0,
        "leaf_wetness_hours": 4.0,
        "consecutive_wet_days": 1,
        "description": "Systemic viral stunting with characteristic narrow leaf aspect ratio (1.97) and chlorosis."
    },
    {
        "id": "healthy_control",
        "name": "Healthy Control Lamina",
        "badge": "Baseline Control",
        "disease_label": "healthy",
        "disease_confidence": 0.99,
        "plant_id": "PLANT_005",
        "zone_id": "ZONE_C",
        "leaf_id": "LEAF_01",
        "observation_date": "2026-10-20",
        "image_path": "images/real_train_0000_91_jpeg.rf.9355cb01c710fd703ccb802eb457668e.jpg",
        "urgency_score": 40.0,
        "temperature_c": 25.0,
        "relative_humidity_pct": 70.0,
        "leaf_wetness_hours": 2.0,
        "consecutive_wet_days": 0,
        "description": "Vibrant, asymptomatic Cavendish banana lamina exhibiting 0% foliar lesion coverage."
    }
]


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


@app.get("/api/v1/demo-presets", status_code=status.HTTP_200_OK)
def get_demo_presets() -> Dict[str, Any]:
    """Returns curated demo presets for one-click viva evaluation."""
    return {
        "count": len(DEMO_PRESETS),
        "presets": DEMO_PRESETS
    }


@app.get("/api/v1/image-preview")
def get_image_preview(path: str = Query(..., description="Relative or absolute path to image")):
    """Streams preview image as JPEG bytes for browser display."""
    orch = get_orchestrator()
    try:
        rgb_img = orch._load_image(path)
        pil_img = Image.fromarray(rgb_img)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unable to load image from path '{path}': {str(e)}"
        )


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


@app.post("/api/v1/assess-interactive", status_code=status.HTTP_200_OK)
def assess_interactive_endpoint(payload: InteractiveAssessmentRequest) -> Dict[str, Any]:
    """
    Rich interactive assessment endpoint returning Component4Output, base64 mask/image,
    morphometrics, 4-strategy fusion comparison, and plant history for web frontend.
    """
    orch = get_orchestrator()
    try:
        c1_input = payload.vision_input
        # Handle optional client uploaded base64 image
        if payload.image_base64:
            b64_data = payload.image_base64
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_data)
            upload_dir = "data/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            upload_filename = f"upload_{uuid.uuid4().hex[:8]}.jpg"
            upload_path = os.path.join(upload_dir, upload_filename)
            with open(upload_path, "wb") as f:
                f.write(img_bytes)
            c1_input.image_path = upload_path

        result = orch.assess_interactive(
            c1_input=c1_input,
            c2_input=payload.environmental_input,
            fusion_method=payload.fusion_method or "F_hybrid"
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Interactive pipeline error: {str(e)}"
        )


class ImageUploadPayload(BaseModel):
    """Payload for browser image upload encoded as base64."""
    image_base64: str = Field(..., description="Base64 encoded image data URL")
    filename: Optional[str] = Field("custom_leaf.jpg", description="Original filename")


@app.post("/api/v1/upload-image", status_code=status.HTTP_200_OK)
def upload_image_endpoint(payload: ImageUploadPayload) -> Dict[str, Any]:
    """Accepts base64 image upload and returns local path for instant UI testing."""
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(payload.filename)[1] or ".jpg"
    safe_name = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    dest_path = os.path.join(upload_dir, safe_name)

    b64_data = payload.image_base64
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]

    try:
        content = base64.b64decode(b64_data)
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        pil_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)
        pil_img.save(dest_path, format="JPEG", quality=90)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded data is not a valid image: {str(e)}"
        )

    return {
        "status": "success",
        "image_path": dest_path,
        "filename": payload.filename
    }


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


# Static Frontend Mount & Index Route
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend_index():
    """Serves the interactive demo frontend dashboard."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend dashboard initializing..."}

