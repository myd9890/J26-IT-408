import os
import io
import uuid
import base64
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from PIL import Image

from src.schemas.contracts import Component4Output
from src.inference.orchestrator import Component3Orchestrator
from src.api.schemas import (
    AssessmentRequest,
    InteractiveAssessmentRequest,
    ImageUploadPayload,
)
from src.api.presets import DEMO_PRESETS


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
