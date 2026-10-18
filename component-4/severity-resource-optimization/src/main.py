"""Run the optimizer with the sample data and save comparison results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.schemas.optimization import OptimizationRequest
from src.services.baseline_service import compare_baselines
from src.services.optimization_service import run_optimization
from src.services.scenario_service import compare_scenarios


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"


def load_sample_request() -> OptimizationRequest:
    """Load zones and available resources from the data directory."""
    with (DATA_DIRECTORY / "zones.csv").open(encoding="utf-8", newline="") as file:
        zones = list(csv.DictReader(file))

    number_fields = {
        "confidence",
        "risk",
        "severity",
        "urgency",
        "chemical",
        "labour",
        "cost",
    }
    for zone in zones:
        for field in number_fields:
            zone[field] = float(zone[field])

    with (DATA_DIRECTORY / "resources.json").open(encoding="utf-8") as file:
        resources = json.load(file)

    return OptimizationRequest(zones=zones, resources=resources)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write result dictionaries to a CSV file."""
    if not rows:
        return

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_results(request: OptimizationRequest) -> None:
    """Generate scenario and baseline comparison CSV files."""
    RESULTS_DIRECTORY.mkdir(exist_ok=True)

    scenarios = compare_scenarios(request)["scenarios"]
    scenario_rows = [
        {
            "Scenario": item["scenario"],
            "Selected Zones": ", ".join(item["selected_zones"]),
            "Total Priority": item["total_priority"],
            "Chemical Used": item["chemical_used"],
            "Labour Used": item["labour_used"],
            "Cost": item["total_cost"],
        }
        for item in scenarios
    ]
    write_csv(RESULTS_DIRECTORY / "scenario_comparison.csv", scenario_rows)

    baselines = compare_baselines(request)["comparison"]
    baseline_rows = [
        {
            "Method": item["method"],
            "Selected Zones": ", ".join(item["selected_zones"]),
            "Zones Treated": item["zones_treated"],
            "Total Priority": item["total_priority"],
            "Chemical Used": item["chemical_used"],
            "Labour Used": item["labour_used"],
            "Cost": item["total_cost"],
        }
        for item in baselines
    ]
    write_csv(RESULTS_DIRECTORY / "baseline_comparison.csv", baseline_rows)


def main() -> None:
    """Print the sample allocation and update saved experiment results."""
    request = load_sample_request()
    result = run_optimization(request)
    save_results(request)

    print(json.dumps(result, indent=2))
    print(f"\nResults saved in: {RESULTS_DIRECTORY}")


if __name__ == "__main__":
    main()
