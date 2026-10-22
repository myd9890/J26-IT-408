"""Small HTTP adapter around the existing Component 1 predictor.

Run from the component-1 directory with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import base64
import io
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.inference.predictor import BananaDiseasePredictor


ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = ROOT / "results" / "api_history"
HISTORY_INDEX = HISTORY_DIR / "history.json"

app = FastAPI(title="AgriLeaf Component 1 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor = None
_predictor_lock = threading.Lock()


def get_predictor() -> BananaDiseasePredictor:
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = BananaDiseasePredictor(
                    prototype_path=str(ROOT / "models" / "resnet18_prototypes_20shot_seed42.pth"),
                    calibration_path=str(ROOT / "models" / "resnet18_confidence_calibration.json"),
                    enable_gatekeeper=True,
                )
    return _predictor


def png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def result_images(result: dict) -> tuple[str | None, str | None]:
    if "display_image" not in result or "gradcam" not in result:
        return None, None
    original = Image.fromarray(np.asarray(result["display_image"]).astype(np.uint8), "RGB")
    heatmap = np.asarray(result["gradcam"], dtype=np.float32).clip(0, 1)
    colored = (matplotlib.colormaps["jet"](heatmap)[..., :3] * 255).astype(np.uint8)
    overlay = Image.blend(original, Image.fromarray(colored, "RGB"), 0.45)
    return png_base64(original), png_base64(overlay)


def public_result(result: dict, scan_id: str) -> dict:
    original, overlay = result_images(result)
    payload = {
        "id": scan_id,
        "image_path": result.get("image_path"),
        "predicted_class": result.get("predicted_class"),
        "status": result.get("status"),
        "message": result.get("message"),
        "model_confidence_score": result.get("model_confidence_score"),
        "model_confidence_percentage": result.get("model_confidence_percentage"),
        "class_scores": result.get("class_scores", {}),
        "distances": result.get("distances", {}),
        "nearest_distance_margin": result.get("nearest_distance_margin"),
        "explanation_target": result.get("explanation_target"),
        "gatekeeper": result.get("gatekeeper"),
        "original_image_base64": original,
        "gradcam_overlay_base64": overlay,
    }
    return payload


def save_history(payload: dict) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    (HISTORY_DIR / f"{payload['id']}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    history = []
    if HISTORY_INDEX.exists():
        try:
            history = json.loads(HISTORY_INDEX.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            history = []
    history.insert(0, {k: v for k, v in payload.items() if not k.endswith("_base64")})
    HISTORY_INDEX.write_text(json.dumps(history[:100], indent=2), encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "component": "banana_disease_detection"}


@app.post("/api/v1/scans")
async def create_scan(image: UploadFile = File(...)) -> dict:
    if not image.filename:
        raise HTTPException(status_code=400, detail="An image file is required")
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")

    scan_id = uuid.uuid4().hex
    upload_dir = ROOT / "results" / "api_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename).suffix.lower() or ".jpg"
    image_path = upload_dir / f"{scan_id}{suffix}"
    image_path.write_bytes(content)

    try:
        result = get_predictor().predict(image_path, explain=True)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    payload = public_result(result, scan_id)
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    save_history(payload)
    return payload


@app.get("/api/v1/scans")
def list_scans() -> list[dict]:
    if not HISTORY_INDEX.exists():
        return []
    try:
        return json.loads(HISTORY_INDEX.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


@app.get("/api/v1/scans/{scan_id}")
def get_scan(scan_id: str) -> dict:
    detail_path = HISTORY_DIR / f"{scan_id}.json"
    if detail_path.exists():
        try:
            return json.loads(detail_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    for scan in list_scans():
        if scan.get("id") == scan_id:
            return scan
    raise HTTPException(status_code=404, detail="Scan not found")
