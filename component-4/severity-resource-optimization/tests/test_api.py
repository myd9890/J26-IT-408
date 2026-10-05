
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

sample_request = {
    "zones": [
        {
            "zone_id": "Z01",
            "disease": "Black Sigatoka",
            "severity": 78,
            "risk": 82,
            "confidence": 0.92,
            "urgency": 90,
            "chemical": 3,
            "labour": 2,
            "cost": 3000
        },
        {
            "zone_id": "Z02",
            "disease": "Yellow Sigatoka",
            "severity": 45,
            "risk": 60,
            "confidence": 0.85,
            "urgency": 55,
            "chemical": 2,
            "labour": 1,
            "cost": 1800
        },
        {
            "zone_id": "Z03",
            "disease": "Cordana Leaf Spot",
            "severity": 30,
            "risk": 55,
            "confidence": 0.76,
            "urgency": 35,
            "chemical": 1,
            "labour": 1,
            "cost": 1200
        },
        {
            "zone_id": "Z04",
            "disease": "Black Sigatoka",
            "severity": 92,
            "risk": 90,
            "confidence": 0.95,
            "urgency": 95,
            "chemical": 4,
            "labour": 3,
            "cost": 4500
        },
        {
            "zone_id": "Z05",
            "disease": "Yellow Sigatoka",
            "severity": 60,
            "risk": 70,
            "confidence": 0.81,
            "urgency": 65,
            "chemical": 2,
            "labour": 2,
            "cost": 2300
        }
    ],
    "resources": {
        "chemical": 7,
        "labour": 5,
        "budget": 8000,
        "max_visits": 3
    }
}


def test_optimizer_returns_optimal():
    response = client.post(
        "/api/v1/optimize",
        json=sample_request
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "Optimal"
    assert data["summary"]["zones_treated"] == 3
    assert data["summary"]["total_priority"] == 204.6


def test_resource_constraints():
    response = client.post(
        "/api/v1/optimize",
        json=sample_request
    )

    assert response.status_code == 200
    summary = response.json()["summary"]
    limits = sample_request["resources"]

    assert summary["chemical_used"] <= limits["chemical"]
    assert summary["labour_used"] <= limits["labour"]
    assert summary["total_cost"] <= limits["budget"]
    assert summary["zones_treated"] <= limits["max_visits"]


def test_scenario_api():
    response = client.post(
        "/api/v1/scenarios",
        json=sample_request
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "Success"
    assert len(data["scenarios"]) == 4


def test_baseline_comparison():
    response = client.post(
        "/api/v1/baselines",
        json=sample_request
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data["comparison"]) == 2
    assert data["priority_improvement_percent"] == 16.71


def test_invalid_severity():
    invalid_request = {
        **sample_request,
        "zones": [
            {**sample_request["zones"][0], "severity": 150}
        ]
    }

    response = client.post(
        "/api/v1/optimize",
        json=invalid_request
    )

    assert response.status_code == 422


def test_duplicate_zone_ids():
    invalid_request = {
        **sample_request,
        "zones": [
            sample_request["zones"][0],
            sample_request["zones"][0]
        ]
    }

    response = client.post(
        "/api/v1/optimize",
        json=invalid_request
    )

    assert response.status_code == 422
