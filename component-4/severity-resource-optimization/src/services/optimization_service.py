
"""Mixed-integer programming model for treatment allocation."""

import pulp

from src.services.priority_service import calculate_priority


def run_optimization(request):
    """Select zones that maximize priority within the resource limits."""
    zones = request.zones
    resources = request.resources

    model = pulp.LpProblem(
        "Resource_Optimization",
        pulp.LpMaximize
    )

    x = {
        i: pulp.LpVariable(
            f"treat_{i}",
            cat=pulp.LpBinary
        )
        for i in range(len(zones))
    }

    scores = [
        calculate_priority(zone)
        for zone in zones
    ]

    # Objective function
    model += pulp.lpSum(
        scores[i] * x[i]
        for i in x
    )

    # Resource constraints
    model += pulp.lpSum(
        zones[i].chemical * x[i]
        for i in x
    ) <= resources.chemical

    model += pulp.lpSum(
        zones[i].labour * x[i]
        for i in x
    ) <= resources.labour

    model += pulp.lpSum(
        zones[i].cost * x[i]
        for i in x
    ) <= resources.budget

    model += pulp.lpSum(
        x.values()
    ) <= resources.max_visits

    # Solve optimization model
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        raise RuntimeError(
            f"Optimization failed: {status}"
        )

    allocation = []

    total_chemical = 0
    total_labour = 0
    total_cost = 0
    total_priority = 0

    for i, zone in enumerate(zones):
        selected = int(pulp.value(x[i]) > 0.5)

        if selected:
            total_chemical += zone.chemical
            total_labour += zone.labour
            total_cost += zone.cost
            total_priority += scores[i]

        allocation.append({
            "zone_id": zone.zone_id,
            "disease": zone.disease,
            "priority": scores[i],
            "selected": selected,
            "chemical_allocated": (
                zone.chemical if selected else 0
            ),
            "labour_allocated": (
                zone.labour if selected else 0
            ),
            "estimated_cost": (
                zone.cost if selected else 0
            )
        })

    allocation.sort(
        key=lambda zone: zone["priority"],
        reverse=True
    )

    return {
        "status": status,
        "allocation": allocation,
        "summary": {
            "zones_treated": sum(
                zone["selected"]
                for zone in allocation
            ),
            "total_priority": round(
                total_priority, 2
            ),
            "chemical_used": total_chemical,
            "labour_used": total_labour,
            "total_cost": total_cost,
            "chemical_remaining": (
                resources.chemical - total_chemical
            ),
            "labour_remaining": (
                resources.labour - total_labour
            ),
            "budget_remaining": (
                resources.budget - total_cost
            )
        }
    }
