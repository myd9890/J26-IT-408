"""
Curated Demonstration Presets for viva and interactive evaluation.
"""

from typing import List, Dict, Any

DEMO_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "bs_moderate",
        "name": "Black Sigatoka (Field Sample)",
        "badge": "Field Real",
        "disease_label": "black_sigatoka",
        "disease_confidence": 0.96,
        "plant_id": "REAL_PLANT_TR_001",
        "zone_id": "ZONE_A",
        "leaf_id": "LEAF_01",
        "observation_date": "2026-09-29",
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
        "observation_date": "2026-09-29",
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
        "observation_date": "2026-09-29",
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
        "observation_date": "2026-09-12",
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
        "observation_date": "2026-09-12",
        "image_path": "images/real_train_0000_91_jpeg.rf.9355cb01c710fd703ccb802eb457668e.jpg",
        "urgency_score": 40.0,
        "temperature_c": 25.0,
        "relative_humidity_pct": 70.0,
        "leaf_wetness_hours": 2.0,
        "consecutive_wet_days": 0,
        "description": "Vibrant, asymptomatic Cavendish banana lamina exhibiting 0% foliar lesion coverage."
    }
]
