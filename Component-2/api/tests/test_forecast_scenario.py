"""Run: python -m unittest discover -s api/tests -v (uses saved models)."""
import json
from pathlib import Path
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from api import model_api as api


class ForecastScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = api.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.history = api.read_weather_file(api.DEMO_ROOT / "history_336h.csv", 336)
        cls.future = api.read_weather_file(api.DEMO_ROOT / "future_168h.csv", 168)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(self.base + path, data=body,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    def test_original_demo_response_unchanged_and_default_matches(self):
        # Captured from the original running endpoint before implementation.
        original = json.loads(Path(__file__).with_name("demo_expected.json").read_text())
        demo = self.request("/api/forecast-demo")
        self.assertEqual(demo, original)
        default = self.request("/api/forecast-scenario", {})
        self.assertEqual(default["results"], original["results"])
        self.assertEqual(default["settings"], {
            "temperature_adjustment_c": 0, "humidity_adjustment_pp": 0,
            "rainfall_multiplier": 1,
        })

    def test_adjustments_recompute_weather_and_actual_model_predictions(self):
        settings = api.scenario_settings({"temperature_adjustment_c": -2,
                                          "humidity_adjustment_pp": 15,
                                          "rainfall_multiplier": 2})
        original_future = self.future.copy(deep=True)
        adjusted = api.scenario_future(self.future, settings)
        pd.testing.assert_frame_equal(self.future, original_future)
        np.testing.assert_allclose(adjusted.air_temp_c, self.future.air_temp_c - 2)
        self.assertTrue(adjusted.estimated_leaf_wet.eq(1).all())
        response = self.request("/api/forecast-scenario", settings)
        baseline = api.forecast_demo()["results"]
        self.assertEqual(response["results"][0], baseline[0])
        for key in ("black", "yellow"):
            self.assertNotEqual([r[key]["score"] for r in response["results"]],
                                [r[key]["score"] for r in baseline])
        # Independently reconstruct endpoint features, then call each saved model.
        series = pd.concat([self.history, adjusted], ignore_index=True)
        for day, row in enumerate(response["results"]):
            end = 336 + day * 24
            observed, window = series.iloc[:end], series.iloc[end - 336:end]
            black = {"air_temp_c": float(observed.air_temp_c.iloc[-1]),
                     "estimated_leaf_wet": int(observed.estimated_leaf_wet.iloc[-1]),
                     "consecutive_wetness_hours": api.wet_streak(observed.estimated_leaf_wet.tolist())}
            yellow = {"mean_temp_previous_14d_c": float(window.air_temp_c.mean()),
                      "wet_hours_previous_14d": int(window.estimated_leaf_wet.sum()),
                      "consecutive_wetness_hours": api.wet_streak(window.estimated_leaf_wet.tolist())}
            self.assertEqual(row["black"], api.predict(black, api.black_model,
                             api.BLACK_FEATURES, "black", future=day > 0))
            self.assertEqual(row["yellow"], api.predict(yellow, api.yellow_model,
                             api.YELLOW_FEATURES, "yellow", future=day > 0))

    def test_each_adjustment_applies_to_all_future_hours(self):
        baseline = api.scenario_future(self.future, api.scenario_settings({}))
        for setting, value in [("temperature_adjustment_c", 2),
                               ("humidity_adjustment_pp", -10),
                               ("rainfall_multiplier", 5)]:
            adjusted = api.scenario_future(self.future, api.scenario_settings({setting: value}))
            self.assertEqual(len(adjusted), 168)
            if setting.startswith("temperature"):
                np.testing.assert_allclose(adjusted.air_temp_c, baseline.air_temp_c + 2)
            elif setting.startswith("humidity"):
                np.testing.assert_allclose(adjusted.relative_humidity_pct,
                                           baseline.relative_humidity_pct - 10)
            else:
                np.testing.assert_allclose(adjusted.rainfall_mm_hr, baseline.rainfall_mm_hr * 5)
            self.assertNotEqual(api.score_forecast(api.forecast_feature_inputs(self.history, adjusted)),
                                api.forecast_demo()["results"])

    def test_wetness_recomputed_with_documented_proxy_and_dewpoint(self):
        adjusted = api.scenario_future(self.future, api.scenario_settings({
            "temperature_adjustment_c": 3, "humidity_adjustment_pp": 20,
            "rainfall_multiplier": 0}))
        self.assertTrue(adjusted.relative_humidity_pct.between(1, 100).all())
        self.assertTrue((adjusted.dew_point_c <= adjusted.air_temp_c + 1e-10).all())
        expected = ((adjusted.relative_humidity_pct >= 95)
                    | ((adjusted.air_temp_c - adjusted.dew_point_c <= 1.5)
                       & (adjusted.solar_radiation_original_nasa_units < 0.15))
                    | (adjusted.rainfall_mm_hr >= 0.5)).astype(int)
        np.testing.assert_array_equal(adjusted.estimated_leaf_wet, expected)

    def test_yellow_windows_exactly_336_and_end_at_prediction(self):
        response = api.forecast_scenario({"temperature_adjustment_c": 1})
        future = api.scenario_future(self.future, response["settings"])
        series = pd.concat([self.history, future], ignore_index=True)
        for day, item in enumerate(response["feature_inputs"]):
            window = series.iloc[day * 24:336 + day * 24]
            self.assertEqual(len(window), 336)
            self.assertEqual(item["yellow_window_hours"], 336)
            self.assertEqual(item["history_hours"], 336 - day * 24)
            self.assertEqual(item["future_hours"], day * 24)
            self.assertEqual(item["as_of"], window.timestamp.iloc[-1].isoformat())
            self.assertAlmostEqual(item["yellow"]["mean_temp_previous_14d_c"], window.air_temp_c.mean())
            self.assertEqual(item["yellow"]["wet_hours_previous_14d"], int(window.estimated_leaf_wet.sum()))

    def test_streaks_reset_and_continue_across_boundary(self):
        self.assertEqual(api.wet_streak([1, 1, 0]), 0)
        self.assertEqual(api.wet_streak([1, 0, 1, 1]), 2)
        history, future = self.history.copy(), self.future.copy()
        history["estimated_leaf_wet"] = 0
        history.loc[333:, "estimated_leaf_wet"] = 1
        future["estimated_leaf_wet"] = 1
        inputs = api.forecast_feature_inputs(history, future)
        for key in ("black", "yellow"):
            self.assertEqual(inputs[0][key]["consecutive_wetness_hours"], 3)
            self.assertEqual(inputs[1][key]["consecutive_wetness_hours"], 27)
        future.loc[22, "estimated_leaf_wet"] = 0
        inputs = api.forecast_feature_inputs(history, future)
        for key in ("black", "yellow"):
            self.assertEqual(inputs[1][key]["consecutive_wetness_hours"], 1)
        history["estimated_leaf_wet"] = 1
        future["estimated_leaf_wet"] = 1
        inputs = api.forecast_feature_inputs(history, future)
        self.assertEqual(inputs[7]["black"]["consecutive_wetness_hours"], 504)
        self.assertEqual(inputs[7]["yellow"]["consecutive_wetness_hours"], 336)

    def test_forecast_shape_bounds_levels_and_alerts(self):
        for settings in ({}, {"temperature_adjustment_c": -5, "humidity_adjustment_pp": 30,
                              "rainfall_multiplier": 5}, {"humidity_adjustment_pp": -30,
                                                         "rainfall_multiplier": 0}):
            response = api.forecast_scenario(settings)
            self.assertEqual([r["day"] for r in response["results"]],
                             ["Current"] + [f"Day {day}" for day in range(1, 8)])
            self.assertIn("Simulated scenario, not a live weather forecast", response["note"])
            for day, row in enumerate(response["results"]):
                for key in ("black", "yellow"):
                    prediction = row[key]
                    self.assertTrue(0 <= prediction["score"] <= 100)
                    self.assertIn(prediction["level"], api.ALERT_RULES)
                    for field, value in api.alert_details(prediction["level"], future=day > 0).items():
                        self.assertEqual(prediction[field], value)

    def test_invalid_adjustments_return_clear_http_validation_errors(self):
        for payload in ([1], {"unknown": 2}, {"temperature_adjustment_c": 6},
                        {"humidity_adjustment_pp": -31}, {"rainfall_multiplier": -1},
                        {"rainfall_multiplier": 6}, {"humidity_adjustment_pp": "wet"},
                        {"temperature_adjustment_c": True}, {"temperature_adjustment_c": None},
                        {"temperature_adjustment_c": float("nan")},
                        {"rainfall_multiplier": float("inf")}):
            with self.subTest(payload=payload), self.assertRaises(HTTPError) as caught:
                self.request("/api/forecast-scenario", payload)
            self.assertEqual(caught.exception.code, 400)
            self.assertTrue(json.load(caught.exception)["error"])

    def test_forecast_from_dry_live_inputs_matches_current_model_predictions(self):
        payload = {
            "black": {"air_temp_c": 29, "estimated_leaf_wet": 0,
                      "consecutive_wetness_hours": 0},
            "yellow": {"mean_temp_previous_14d_c": 29,
                       "wet_hours_previous_14d": 0,
                       "consecutive_wetness_hours": 0},
        }
        response = self.request("/api/forecast-from-inputs", payload)
        current = api.predict_current(payload)
        self.assertEqual([row["day"] for row in response["results"]],
                         ["Current"] + [f"Day {i}" for i in range(1, 8)])
        self.assertEqual(response["results"][0]["black"]["score"], current["black"]["score"])
        self.assertEqual(response["results"][0]["yellow"]["score"], current["yellow"]["score"])
        self.assertEqual(response["feature_inputs"][0]["black"], payload["black"])
        self.assertEqual(response["feature_inputs"][0]["yellow"]["wet_hours_previous_14d"], 0)
        self.assertTrue(all(item["yellow_window_hours"] == 336
                            for item in response["feature_inputs"]))
        self.assertTrue(all("recommended_action" in row[key]
                            for row in response["results"] for key in ("black", "yellow")))

    def test_forecast_from_wet_live_inputs_reconstructs_exact_summary(self):
        payload = {
            "black": {"air_temp_c": 27, "estimated_leaf_wet": 1,
                      "consecutive_wetness_hours": 72},
            "yellow": {"mean_temp_previous_14d_c": 27,
                       "wet_hours_previous_14d": 96,
                       "consecutive_wetness_hours": 72},
        }
        response = self.request("/api/forecast-from-inputs", payload)
        history = api.reconstruct_history(payload)
        self.assertAlmostEqual(float(history.air_temp_c.mean()), 27, places=9)
        self.assertEqual(float(history.air_temp_c.iloc[-1]), 27)
        self.assertEqual(int(history.estimated_leaf_wet.sum()), 96)
        self.assertEqual(api.wet_streak(history.estimated_leaf_wet), 72)
        self.assertTrue(history.air_temp_c.between(8, 40).all())
        self.assertEqual(response["results"][0]["black"]["level"],
                         api.predict_current(payload)["black"]["level"])
        self.assertIn("simulated risk outlook", response["note"])

    def test_changed_inputs_are_not_cached_and_current_matches_predict_endpoint(self):
        dry = {"black": {"air_temp_c": 29, "estimated_leaf_wet": 0,
                          "consecutive_wetness_hours": 0},
               "yellow": {"mean_temp_previous_14d_c": 29, "wet_hours_previous_14d": 0,
                          "consecutive_wetness_hours": 0}}
        wet = {"black": {"air_temp_c": 27, "estimated_leaf_wet": 1,
                          "consecutive_wetness_hours": 72},
               "yellow": {"mean_temp_previous_14d_c": 27, "wet_hours_previous_14d": 96,
                          "consecutive_wetness_hours": 72}}
        first = self.request("/api/forecast-from-inputs", dry)
        second = self.request("/api/forecast-from-inputs", wet)
        self.assertNotEqual(first["feature_inputs"], second["feature_inputs"])
        for payload, response in ((dry, first), (wet, second)):
            current = api.predict_current(payload)
            for disease in ("black", "yellow"):
                self.assertEqual(response["results"][0][disease]["score"], current[disease]["score"])
                self.assertEqual(response["results"][0][disease]["level"], current[disease]["level"])

    def test_impossible_forecast_inputs_are_rejected_clearly(self):
        valid = {"black": {"air_temp_c": 27, "estimated_leaf_wet": 1,
                           "consecutive_wetness_hours": 72},
                 "yellow": {"mean_temp_previous_14d_c": 27,
                            "wet_hours_previous_14d": 96,
                            "consecutive_wetness_hours": 72}}
        invalid = []
        mismatch = json.loads(json.dumps(valid))
        mismatch["yellow"]["consecutive_wetness_hours"] = 71
        invalid.append(mismatch)
        low_total = json.loads(json.dumps(valid))
        low_total["yellow"]["wet_hours_previous_14d"] = 71
        invalid.append(low_total)
        impossible_mean = json.loads(json.dumps(valid))
        impossible_mean["yellow"]["mean_temp_previous_14d_c"] = 40
        invalid.append(impossible_mean)
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(HTTPError) as caught:
                self.request("/api/forecast-from-inputs", payload)
            self.assertEqual(caught.exception.code, 400)
            self.assertTrue(json.load(caught.exception)["error"])


if __name__ == "__main__":
    unittest.main()
