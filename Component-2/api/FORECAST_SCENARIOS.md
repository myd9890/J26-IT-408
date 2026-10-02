# PP1 interactive forecast

The existing service is `api/model_api.py` (Python ThreadingHTTPServer, not FastAPI).
Run it with `.venv\Scripts\python.exe -u api\model_api.py` from the project root.

`GET /api/forecast-demo` retains its original response and fixed weather data.
The Flutter forecast page now uses `POST /api/forecast-from-inputs` with the
latest readings successfully submitted on Live Risk. The old scenario endpoint
below remains available to API clients, but its controls are no longer shown in
Flutter.

## Forecast from Live Risk inputs

The request uses the same nested feature names as `/api/predict-current`:

```json
{
  "black": {
    "air_temp_c": 27,
    "estimated_leaf_wet": 1,
    "consecutive_wetness_hours": 72
  },
  "yellow": {
    "mean_temp_previous_14d_c": 27,
    "wet_hours_previous_14d": 96,
    "consecutive_wetness_hours": 72
  }
}
```

The service checks that the Black and Yellow ending streaks match, that dry/wet
status agrees with the streak, that hourly counts are whole numbers, and that
the requested feature values fit the saved model ranges. It builds an in-memory
336-hour history from the fixed demo history pattern. It offsets the first 335
temperatures uniformly to match the requested 14-day mean, then sets the last
temperature to the entered current value. It adjusts wet flags to match the
requested 14-day wet-hour total and ending streak while preserving the reference
pattern where possible. The ending wet spell is contiguous and follows a dry
hour (unless it fills the entire history). Impossible temperatures or wetness
totals return HTTP 400.

This is a synthetic reconstruction, not recovery of measured hourly history.
Its mean matches within 1e-9 C. The original 168-hour future CSV is used as a
fixed simulated future assumption. Current and daily endpoints are scored with
the saved Black Random Forest and Yellow XGBoost. Yellow windows contain exactly
336 readings; streaks are recalculated from the resulting sequence and cross
the history/future boundary. The Current forecasts must equal
`/api/predict-current` for the submitted feature rows. Each prediction includes
`recommended_action`, derived from its existing model-specific risk level.

`POST /api/forecast-scenario` accepts a JSON object with optional settings:

```json
{
  "temperature_adjustment_c": 0,
  "humidity_adjustment_pp": 0,
  "rainfall_multiplier": 1
}
```

Temperature offset: -5 to +5 °C; humidity offset: -30 to +30 percentage points;
rain multiplier: 0 to 5. Defaults match the original demo exactly. Invalid types,
nonfinite values, unknown settings, and out-of-range adjustments return HTTP 400
with an `error` message. These limits are demonstration controls, not validated
model applicability limits. Extreme settings can exceed model training ranges.

## Baseline assumptions and wetness

The original notebook creates seeded temperature and binary wet/dry series;
neither CSV contains RH, rainfall, dew point, or solar measurements. Recovering
these missing quantities is impossible. For scenarios only, construct explicit
future weather assumptions in memory from the saved wet flags:

| Baseline future hour | RH | Rain mm/h | Solar, original NASA units |
|---|---:|---:|---:|
| Wet, even zero-based future index | 96% | 0.1 | 0.2 |
| Wet, odd zero-based future index | 90% | 0.6 | 0.2 |
| Dry | 85% | 0.1 | 0.2 |

These are assumed simulation inputs, not reconstructed or observed weather.
Apply temperature/RH offsets and rain multiplier to every future hour; bound RH
to 1–100%, keep solar fixed, and recompute dew point using the project's Magnus
relation. Recalculate wetness using the existing Yellow project proxy documented
in `datasets/yellow_sigatoka/YS_METHODS_README.md` and its generator:

`RH >= 95 OR (T - dewpoint <= 1.5 AND solar < 0.15) OR rain >= 0.5`.

This uncalibrated proxy is shared by both models in this demonstration to retain
the notebook's common wetness series. Black's training documentation also has a
different RH>=98/rain>=1 historical proxy; this scenario does not claim calibrated
equivalence to that method. Every baseline wet flag is reproduced by the selected
proxy. Rain- and humidity-driven assumptions allow both controls to affect wetness.
Small adjustments can leave thresholded wetness and model scores unchanged.

## Features and response

Keep the original 336 simulated history hours unchanged. Join adjusted future
hours in chronological order and assess Current, then endpoints 24, 48, …, 168
hours into the future. Black uses endpoint temperature/wetness and the trailing
wet streak across all available hours. Yellow uses exactly the last 336 hourly
readings, including the endpoint, for mean temperature, wet total, and trailing
wet streak (capped by the window). Streaks reset on dry hours and cross the
history/future boundary. Day 1 uses 312 historical/24 future readings; Day 7 uses
168 historical/168 future readings. No future information enters Current.

Saved Random Forest and XGBoost models score the recalculated features directly.
The original clipping to 0–100, full-precision category thresholds, rounding, and
alert rules are unchanged. No adjustments are applied directly to risk scores.

The scenario response contains `results`, normalized `settings`, `feature_inputs`
with timestamps/window composition, `wetness_estimation_method`, and a
simulation/provenance `note`. The Flutter page currently uses the separate
input-based endpoint described above.

Recommendations must use `day['black']['level']` / `day['yellow']['level']`, not
display names as response keys. No new risk category or threshold is introduced.

No CSV, notebook, training dataset, saved model, or research output is written.

## Verification

From the project root:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
& .\.venv\Scripts\python.exe -m unittest discover -s api/tests -v
```

From `Research-Project-main`, run `flutter test` and `flutter build web --no-pub`.
`api/tests/demo_expected.json` is the original API response captured before this
change; it checks that refactoring did not alter the fixed demonstration.
