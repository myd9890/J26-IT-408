
"""Evaluate how resource shortages change the treatment plan."""

from src.services.optimization_service import run_optimization


def compare_scenarios(request):
    """Run the optimizer under normal and reduced-resource scenarios."""
    base = request.resources

    ResourceModel = type(base)
    RequestModel = type(request)

    scenarios = {
        "Normal": base,

        "Chemical Shortage": ResourceModel(
            chemical=base.chemical * 0.6,
            labour=base.labour,
            budget=base.budget,
            max_visits=base.max_visits
        ),

        "Labour Shortage": ResourceModel(
            chemical=base.chemical,
            labour=base.labour * 0.6,
            budget=base.budget,
            max_visits=base.max_visits
        ),

        "Budget Shortage": ResourceModel(
            chemical=base.chemical,
            labour=base.labour,
            budget=base.budget * 0.5,
            max_visits=base.max_visits
        )
    }

    results = []

    for name, resources in scenarios.items():
        scenario_request = RequestModel(
            zones=request.zones,
            resources=resources
        )

        response = run_optimization(scenario_request)
        summary = response["summary"]

        results.append({
            "scenario": name,
            "selected_zones": [
                zone["zone_id"]
                for zone in response["allocation"]
                if zone["selected"] == 1
            ],
            "total_priority": summary["total_priority"],
            "chemical_used": summary["chemical_used"],
            "labour_used": summary["labour_used"],
            "total_cost": summary["total_cost"]
        })

    return {
        "status": "Success",
        "scenarios": results
    }
