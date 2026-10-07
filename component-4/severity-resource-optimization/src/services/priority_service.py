
"""Priority scoring for disease-affected plantation zones."""


SEVERITY_WEIGHT = 0.40
RISK_WEIGHT = 0.30
CONFIDENCE_WEIGHT = 0.10
URGENCY_WEIGHT = 0.20


def calculate_priority(zone) -> float:
    """Return a weighted intervention score from 0 to 100."""

    score = (
        zone.severity * SEVERITY_WEIGHT
        + zone.risk * RISK_WEIGHT
        + zone.confidence * 100 * CONFIDENCE_WEIGHT
        + zone.urgency * URGENCY_WEIGHT
    )

    return round(score, 2)
