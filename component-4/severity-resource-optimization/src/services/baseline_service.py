
"""Compare optimized allocation against a simple severity-first method."""

from src.services.priority_service import calculate_priority
from src.services.optimization_service import run_optimization


def compare_baselines(request):
    """Return severity-first and MILP allocation summaries."""
    limits = request.resources

    # Method 1: Severity Ranking
    ranked = sorted(
        request.zones,
        key=lambda zone: zone.severity,
        reverse=True
    )

    selected = []
    chemical_used = 0
    labour_used = 0
    total_cost = 0

    for zone in ranked:
        if (
            chemical_used + zone.chemical <= limits.chemical
            and labour_used + zone.labour <= limits.labour
            and total_cost + zone.cost <= limits.budget
            and len(selected) < limits.max_visits
        ):
            selected.append(zone)

            chemical_used += zone.chemical
            labour_used += zone.labour
            total_cost += zone.cost

    baseline_priority = round(
        sum(calculate_priority(z) for z in selected),
        2
    )

    # Method 2: MILP Optimization
    milp_result = run_optimization(request)
    milp_summary = milp_result["summary"]

    milp_zones = [
        zone["zone_id"]
        for zone in milp_result["allocation"]
        if zone["selected"] == 1
    ]

    improvement = (
        (milp_summary["total_priority"] - baseline_priority)
        / baseline_priority * 100
        if baseline_priority > 0
        else None
    )

    return {
        "status": "Success",
        "comparison": [
            {
                "method": "Severity Ranking",
                "selected_zones": [
                    zone.zone_id for zone in selected
                ],
                "zones_treated": len(selected),
                "total_priority": baseline_priority,
                "chemical_used": chemical_used,
                "labour_used": labour_used,
                "total_cost": total_cost
            },
            {
                "method": "MILP Optimization",
                "selected_zones": milp_zones,
                "zones_treated": milp_summary["zones_treated"],
                "total_priority": milp_summary["total_priority"],
                "chemical_used": milp_summary["chemical_used"],
                "labour_used": milp_summary["labour_used"],
                "total_cost": milp_summary["total_cost"]
            }
        ],
        "priority_improvement_percent": (
            round(improvement, 2)
            if improvement is not None
            else None
        )
    }
