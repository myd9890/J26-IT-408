"""Local API for the saved Black and Yellow Sigatoka models."""

import csv
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "Models"
DEMO_ROOT = PROJECT_ROOT / "forecast_demo"
BLACK_FEATURES = ["air_temp_c", "estimated_leaf_wet", "consecutive_wetness_hours"]
YELLOW_FEATURES = [
    "mean_temp_previous_14d_c",
    "wet_hours_previous_14d",
    "consecutive_wetness_hours",
]
ALERT_RULES = {
    "No Risk": (
        "none",
        False,
        "No elevated environmental risk indicated. Continue routine monitoring.",
    ),
    "Low": ("info", False, "Slightly favourable conditions. Continue monitoring."),
    "Moderate": (
        "warning",
        True,
        "Moderately favourable conditions. Increase monitoring and consider field inspection.",
    ),
    "High": (
        "critical",
        True,
        "Highly favourable conditions. Prioritize field inspection and notify responsible staff.",
    ),
}

black_model = joblib.load(
    MODEL_ROOT / "black_sigatoka/random_forest/random_forest_model.joblib"
)
yellow_model = joblib.load(
    MODEL_ROOT / "yellow_sigatoka/xgboost/xgboost_model.joblib"
)
black_metadata = json.loads(
    (MODEL_ROOT / "black_sigatoka/random_forest/model_metadata.json").read_text(
        encoding="utf-8"
    )
)
yellow_metadata = json.loads(
    (MODEL_ROOT / "yellow_sigatoka/xgboost/xgboost_metadata.json").read_text(
        encoding="utf-8"
    )
)


def category(score, disease):
    if disease == "black":
        if score == 0:
            return "No Risk"
        if score < 25:
            return "Low"
        if score < 75:
            return "Moderate"
        return "High"
    if score < 5:
        return "No Risk"
    if score < 30:
        return "Low"
    if score < 65:
        return "Moderate"
    return "High"


def numeric_values(values, names):
    result = {}
    for name in names:
        value = values.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number.")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        result[name] = float(value)
    return result


def alert_details(level, future=False):
    severity, required, message = ALERT_RULES[level]
    if future:
        message = "Forecast environmental conditions: " + message
    return {
        "alert_severity": severity,
        "alert_required": required,
        "alert_message": message,
    }


def predict(features, model, columns, disease, future=False):
    frame = pd.DataFrame([features], columns=columns)
    raw_score = float(model.predict(frame)[0])
    if not math.isfinite(raw_score):
        raise ValueError("The model returned an invalid score.")
    score = min(100.0, max(0.0, raw_score))
    level = category(score, disease)
    return {
        "score": round(score, 2),
        "level": level,
        **alert_details(level, future=future),
    }


def predict_current(payload):
    black = numeric_values(payload.get("black", {}), BLACK_FEATURES)
    yellow = numeric_values(payload.get("yellow", {}), YELLOW_FEATURES)
    if black["estimated_leaf_wet"] not in (0, 1):
        raise ValueError("estimated_leaf_wet must be 0 or 1.")
    if not 0 <= black["consecutive_wetness_hours"] <= 120:
        raise ValueError("Black wetness duration must be between 0 and 120 hours.")
    if (black["estimated_leaf_wet"] == 0) != (
        black["consecutive_wetness_hours"] == 0
    ):
        raise ValueError("Dry hours need zero duration; wet hours need a positive duration.")
    if black["consecutive_wetness_hours"] != yellow["consecutive_wetness_hours"]:
        raise ValueError("Black and Yellow consecutive wetness durations must match.")
    if not 17.011 <= black["air_temp_c"] <= 33.866:
        raise ValueError("Black air temperature must be within the saved model range 17.011 to 33.866 C.")
    if not 8 <= yellow["mean_temp_previous_14d_c"] <= 40:
        raise ValueError("Yellow 14-day mean temperature must be between 8 and 40 C.")
    if not 0 <= yellow["wet_hours_previous_14d"] <= 336:
        raise ValueError("Yellow wet hours must be between 0 and 336.")
    if not 0 <= yellow["consecutive_wetness_hours"] <= min(
        336, yellow["wet_hours_previous_14d"]
    ):
        raise ValueError("Yellow streak must be within the 14-day wet-hour count.")

    return {
        "black": predict(black, black_model, BLACK_FEATURES, "black"),
        "yellow": predict(yellow, yellow_model, YELLOW_FEATURES, "yellow"),
        "note": "Scores reproduce synthetic environmental-risk labels; they are not disease diagnoses.",
    }


def _validated_forecast_inputs(payload):
    """Validate the submitted Live Risk features before making a history."""
    predict_current(payload)  # Shares feature and physical-consistency validation.
    black = numeric_values(payload.get("black", {}), BLACK_FEATURES)
    yellow = numeric_values(payload.get("yellow", {}), YELLOW_FEATURES)
    for name, value in (
        ("Black consecutive wetness hours", black["consecutive_wetness_hours"]),
        ("Yellow wet hours", yellow["wet_hours_previous_14d"]),
        ("Yellow consecutive wetness hours", yellow["consecutive_wetness_hours"]),
    ):
        if not value.is_integer():
            raise ValueError(f"{name} must be a whole number of hours.")
    return black, yellow


def reconstruct_history(payload):
    """Fit a synthetic 336-hour history to submitted summary features.

    Temperatures retain the reference pattern plus one uniform offset over the
    first 335 hours; the final reading is exactly the submitted current value.
    Wet flags retain the reference pattern where possible, with the requested
    ending streak and wet-hour count enforced exactly. This is a construction,
    not recovery of the user's actual hourly history.
    """
    black, yellow = _validated_forecast_inputs(payload)
    history = read_weather_file(DEMO_ROOT / "history_336h.csv", 336)
    temperature = history["air_temp_c"].to_numpy(dtype=float).copy()
    target_mean = yellow["mean_temp_previous_14d_c"]
    current_temperature = black["air_temp_c"]
    prefix_mean = (336 * target_mean - current_temperature) / 335
    temperature_offset = prefix_mean - float(temperature[:335].mean())
    temperature[:335] += temperature_offset
    temperature[335] = current_temperature
    if not np.isfinite(temperature).all() or (temperature < 8).any() or (temperature > 40).any():
        raise ValueError(
            "The requested 14-day mean and current temperature cannot form a plausible 336-hour history within 8 to 40 C."
        )

    wet_hours = int(yellow["wet_hours_previous_14d"])
    streak = int(yellow["consecutive_wetness_hours"])
    wet = history["estimated_leaf_wet"].to_numpy(dtype=int).copy()
    wet[336 - streak :] = 1 if streak else 0
    if streak < 336:
        wet[335 - streak] = 0
    fixed_prefix_end = 335 - streak if streak < 336 else 0
    prefix = wet[:fixed_prefix_end]
    needed = wet_hours - streak
    reference_count = int(prefix.sum())
    if reference_count > needed:
        remove = np.flatnonzero(prefix == 1)[::-1][: reference_count - needed]
        prefix[remove] = 0
    elif reference_count < needed:
        add = np.flatnonzero(prefix == 0)[: needed - reference_count]
        prefix[add] = 1
    if int(wet.sum()) != wet_hours or wet_streak(wet) != streak:
        raise ValueError("The requested wet-hour total and ending wet streak are inconsistent.")

    history["air_temp_c"] = temperature
    history["estimated_leaf_wet"] = wet
    if abs(float(temperature.mean()) - target_mean) > 1e-9:
        raise ValueError("Could not reproduce the requested 14-day mean temperature.")
    return history


def forecast_from_inputs(payload):
    history = reconstruct_history(payload)
    future = read_weather_file(DEMO_ROOT / "future_168h.csv", 168)
    feature_inputs = forecast_feature_inputs(history, future)
    black, yellow = _validated_forecast_inputs(payload)
    current = feature_inputs[0]
    if (
        abs(current["yellow"]["mean_temp_previous_14d_c"]
            - yellow["mean_temp_previous_14d_c"]) > 1e-9
        or current["yellow"]["wet_hours_previous_14d"]
            != yellow["wet_hours_previous_14d"]
        or current["yellow"]["consecutive_wetness_hours"]
            != yellow["consecutive_wetness_hours"]
    ):
        raise ValueError("Constructed history does not match the submitted Yellow summary.")
    # Use the exact submitted values for Current so its model predictions and
    # alert levels are identical to /api/predict-current, even near tree splits.
    current["black"] = black
    current["yellow"] = yellow
    if any(item["black"]["consecutive_wetness_hours"] > 120 for item in feature_inputs):
        raise ValueError(
            "The simulated future extends the Black wet streak beyond the saved model's 120-hour training range. Enter a shorter current wet streak."
        )
    results = score_forecast(feature_inputs)
    expected = predict_current(payload)
    if results[0]["black"] != expected["black"] or results[0]["yellow"] != expected["yellow"]:
        raise ValueError("Constructed Current features do not match the submitted Live Risk inputs.")
    actions = {
        "black": "Black Sigatoka",
        "yellow": "Yellow Sigatoka",
    }
    for row in results:
        for key, disease in actions.items():
            row[key]["recommended_action"] = recommendation(disease, row[key]["level"])
    return {
        "results": results,
        "feature_inputs": feature_inputs,
        "note": (
            "Seven-day simulated risk outlook based on your latest environmental readings. "
            "The 336-hour history is synthetically reconstructed to match the submitted "
            "summary, using the project demo pattern as a reference; the 168-hour future "
            "uses the fixed simulated future pattern. Neither is measured weather or a live forecast."
        ),
        "history_reconstruction": {
            "method": "Reference history pattern; uniform temperature offset over hours 1-335; wet flags adjusted to match total and ending streak.",
            "mean_temperature_tolerance_c": 1e-9,
            "history_is_observed": False,
            "future_is_observed": False,
        },
    }


def recommendation(disease, level):
    if disease == "Black Sigatoka":
        return {
            "No Risk": "Continue routine scouting and keep records of field conditions.",
            "Low": "Inspect older leaves for early dark streaks. Remove affected leaves using clean tools.",
            "Moderate": "Increase scouting, remove symptomatic leaves, and ask a local crop adviser to confirm symptoms.",
            "High": "Arrange prompt field confirmation and follow local integrated Sigatoka management guidance.",
        }[level]
    return {
        "No Risk": "Continue routine scouting and maintain clean, well-drained blocks.",
        "Low": "Check leaves for yellow-bordered streaks or spots and record affected plants.",
        "Moderate": "Inspect affected blocks more often, remove symptomatic leaves safely, and improve drainage and airflow.",
        "High": "Arrange prompt field confirmation and follow local integrated Sigatoka management guidance.",
    }[level]


def read_weather_file(path, expected_rows):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(line for line in file if not line.startswith("#")))
    if len(rows) != expected_rows:
        raise ValueError(f"{path.name} must contain {expected_rows} hourly rows.")
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame["air_temp_c"] = pd.to_numeric(frame["air_temp_c"], errors="raise")
    frame["estimated_leaf_wet"] = pd.to_numeric(
        frame["estimated_leaf_wet"], errors="raise"
    ).astype(int)
    if frame["timestamp"].duplicated().any() or not frame["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{path.name} timestamps must be unique and ordered.")
    if not frame["estimated_leaf_wet"].isin([0, 1]).all():
        raise ValueError(f"{path.name} wetness must be binary.")
    return frame


def wet_streak(values):
    duration = 0
    for value in reversed(values):
        if value == 0:
            break
        duration += 1
    return duration


def forecast_feature_inputs(history, future):
    """Notebook semantics: include endpoint, never include hours after prediction."""
    weather = pd.concat([history, future], ignore_index=True)
    if len(history) != 336 or len(future) != 168:
        raise ValueError("Forecast requires 336 historical and 168 future hours.")
    if not weather["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Forecast weather must contain continuous hourly timestamps.")
    inputs = []

    for day in range(8):
        end = 335 + day * 24
        observed = weather.iloc[: end + 1]
        window = observed.tail(336)
        if len(window) != 336:
            raise ValueError("Yellow Sigatoka requires exactly 336 hourly readings.")
        black_features = {
            "air_temp_c": float(observed["air_temp_c"].iloc[-1]),
            "estimated_leaf_wet": int(observed["estimated_leaf_wet"].iloc[-1]),
            "consecutive_wetness_hours": wet_streak(
                observed["estimated_leaf_wet"].tolist()
            ),
        }
        yellow_features = {
            "mean_temp_previous_14d_c": float(window["air_temp_c"].mean()),
            "wet_hours_previous_14d": int(window["estimated_leaf_wet"].sum()),
            "consecutive_wetness_hours": wet_streak(
                window["estimated_leaf_wet"].tolist()
            ),
        }
        inputs.append({
            "day": "Current" if day == 0 else f"Day {day}",
            "as_of": observed["timestamp"].iloc[-1].isoformat(),
            "yellow_window_hours": len(window),
            "history_hours": 336 - day * 24,
            "future_hours": day * 24,
            "black": black_features,
            "yellow": yellow_features,
        })
    return inputs


def score_forecast(inputs):
    results = []
    for day, item in enumerate(inputs):
        results.append(
            {
                "day": item["day"],
                "black": predict(
                    item["black"],
                    black_model,
                    BLACK_FEATURES,
                    "black",
                    future=day > 0,
                ),
                "yellow": predict(
                    item["yellow"],
                    yellow_model,
                    YELLOW_FEATURES,
                    "yellow",
                    future=day > 0,
                ),
            }
        )
    return results


def forecast_demo():
    history = read_weather_file(DEMO_ROOT / "history_336h.csv", 336)
    future = read_weather_file(DEMO_ROOT / "future_168h.csv", 168)
    return {
        "results": score_forecast(forecast_feature_inputs(history, future)),
        "note": "Uses the saved simulated weather example, not live sensor readings or a provider forecast.",
    }


SCENARIO_LIMITS = {
    "temperature_adjustment_c": (-5, 5, 0),
    "humidity_adjustment_pp": (-30, 30, 0),
    "rainfall_multiplier": (0, 5, 1),
}
WETNESS_METHOD = "RH>=95 OR (T-DEW<=1.5 AND SOLAR<0.15) OR RAIN>=0.5; UNCALIBRATED"


def scenario_settings(payload):
    if not isinstance(payload, dict):
        raise ValueError("Scenario settings must be a JSON object.")
    unknown = set(payload) - SCENARIO_LIMITS.keys()
    if unknown:
        raise ValueError(f"Unknown scenario settings: {', '.join(sorted(unknown))}.")
    settings = {}
    for name, (minimum, maximum, default) in SCENARIO_LIMITS.items():
        value = numeric_values({name: payload.get(name, default)}, [name])[name]
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}.")
        settings[name] = value
    return settings


def scenario_future(future, settings):
    """Construct missing weather assumptions in memory; never infer observations.

    Alternating wet hours are humidity-driven (96% RH, 0.1 mm/h) or
    rain-driven (90% RH, 0.6 mm/h); dry hours use 85% RH, 0.1 mm/h.
    Solar is a fixed assumed 0.2 in the proxy's original NASA units.
    These assumptions reproduce every baseline wet flag, not missing weather.
    """
    adjusted = future.copy(deep=True)
    wet = future["estimated_leaf_wet"].to_numpy() == 1
    humidity_driven = wet & (np.arange(len(future)) % 2 == 0)
    temperature = future["air_temp_c"].to_numpy() + settings["temperature_adjustment_c"]
    humidity = np.clip(
        np.where(wet, np.where(humidity_driven, 96.0, 90.0), 85.0)
        + settings["humidity_adjustment_pp"], 1, 100,
    )
    rain = np.where(wet & ~humidity_driven, 0.6, 0.1) * settings["rainfall_multiplier"]
    # Magnus relation used by the project's Yellow simulation generator.
    alpha = np.log(humidity / 100) + 17.625 * temperature / (243.04 + temperature)
    dew = 243.04 * alpha / (17.625 - alpha)
    solar = np.full(len(future), 0.2)
    adjusted["air_temp_c"] = temperature
    adjusted["relative_humidity_pct"] = humidity
    adjusted["rainfall_mm_hr"] = rain
    adjusted["dew_point_c"] = dew
    adjusted["solar_radiation_original_nasa_units"] = solar
    adjusted["estimated_leaf_wet"] = (
        (humidity >= 95) | ((temperature - dew <= 1.5) & (solar < 0.15)) | (rain >= 0.5)
    ).astype(int)
    return adjusted


def forecast_scenario(payload):
    settings = scenario_settings(payload)
    history = read_weather_file(DEMO_ROOT / "history_336h.csv", 336)
    future = scenario_future(read_weather_file(DEMO_ROOT / "future_168h.csv", 168), settings)
    inputs = forecast_feature_inputs(history, future)
    return {
        "results": score_forecast(inputs),
        "settings": settings,
        "feature_inputs": inputs,
        "wetness_estimation_method": WETNESS_METHOD,
        "note": (
            "Simulated scenario, not a live weather forecast. Current uses the unchanged "
            "336-hour simulated history, not recent sensors. Adjustments apply to all 168 "
            "future hours. Missing RH/rain/solar are constructed assumptions: wet hours "
            "alternate 96% RH/0.1 mm/h and 90% RH/0.6 mm/h; dry hours use 85% RH/0.1 mm/h; "
            "solar=0.2 in original NASA units. RH is bounded to 1-100%; dew point is "
            "recomputed from adjusted temperature/RH. Wetness uses the uncalibrated "
            "project Yellow proxy, shared by both models as in the demo. Model features "
            "may exceed training ranges under extreme settings. Scores reproduce synthetic "
            "environmental-risk labels, not disease diagnoses or validated probabilities."
        ),
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_json(200, {})

    def do_GET(self):
        if self.path == "/api/health":
            self.send_json(200, {"status": "ok"})
        elif self.path == "/api/forecast-demo":
            try:
                self.send_json(200, forecast_demo())
            except Exception as error:
                self.send_json(500, {"error": str(error)})
        else:
            self.send_json(404, {"error": "Route not found."})

    def do_POST(self):
        routes = {
            "/api/predict-current": predict_current,
            "/api/forecast-scenario": forecast_scenario,
            "/api/forecast-from-inputs": forecast_from_inputs,
        }
        if self.path not in routes:
            self.send_json(404, {"error": "Route not found."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            self.send_json(200, routes[self.path](payload))
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.send_json(500, {"error": str(error)})

    def log_message(self, format_string, *args):
        print(f"{self.address_string()} - {format_string % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Risk model API listening at http://127.0.0.1:8765")
    server.serve_forever()
