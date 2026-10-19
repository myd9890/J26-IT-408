
"""FastAPI endpoints for the resource optimization system."""

from fastapi import FastAPI, HTTPException

from src.schemas.optimization import OptimizationRequest

from src.services.optimization_service import run_optimization
from src.services.scenario_service import compare_scenarios
from src.services.baseline_service import compare_baselines
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Severity Resource Optimization API",
    version="1.0.0",
    description=(
        "Decision-support API that prioritizes disease-affected zones and "
        "allocates limited treatment resources using MILP."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)


def validate_zone_ids(request: OptimizationRequest) -> None:
    """Reject requests containing duplicate zone identifiers."""
    ids = [zone.zone_id for zone in request.zones]

    if len(ids) != len(set(ids)):
        raise HTTPException(
            status_code=422,
            detail="Duplicate zone IDs are not allowed"
        )


@app.get("/", tags=["System"])
def home():
    return {
        "message": "Severity Resource Optimization API running",
        "version": "1.0.0"
    }


@app.post("/api/v1/optimize", tags=["Optimization"])
def optimize(request: OptimizationRequest):
    validate_zone_ids(request)

    try:
        return run_optimization(request)

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post("/api/v1/scenarios", tags=["Analysis"])
def scenarios_endpoint(request: OptimizationRequest):
    validate_zone_ids(request)

    try:
        return compare_scenarios(request)

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post("/api/v1/baselines", tags=["Analysis"])
def baselines_endpoint(request: OptimizationRequest):
    validate_zone_ids(request)

    try:
        return compare_baselines(request)

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
