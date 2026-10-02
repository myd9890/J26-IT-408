# Black Sigatoka — EDA Issue Resolution and Dataset Review

## What was actually fixed
- Restored the original dataset README and generator script. They explain the synthetic scenario construction and original category thresholds.
- Added source-**donor-time chronological** train/validation/test files, using historical donor times, not random scenario IDs. 2022–2023 is training, first half of 2024 validation, second half 2024 test. A **120-hour purge on both sides of Jan 1 and Jul 1, 2024** excludes boundary donors. Every synthetic row associated with an excluded donor is excluded from the proposed modeling splits. Counts below are observed.
- Defined a baseline feature policy. The target is `risk_score`; predictors are `air_temp_c`, `estimated_leaf_wet`, `consecutive_wetness_hours`. Exclude `risk_level`, intermediate risk outputs, source and ID fields, and all-empty synthetic columns.
- Created a historical QC-flagged copy; retained and flagged the disputed dew-point record rather than rewriting a NASA-derived value without authoritative source data.
- Preserved all original CSVs byte-for-byte in `original_unchanged/`.

## Exact revised split counts
|Split|Rows|No Risk|Low|Moderate|High|Earliest donor UTC|Latest donor UTC|
|---|---:|---:|---:|---:|---:|---|---|
|train|4501|1037|1300|1144|1020|2022-01-04 04:00:00+00:00|2023-12-26 00:00:00+00:00|
|validation|937|323|136|235|243|2024-01-06 01:00:00+00:00|2024-06-25 22:00:00+00:00|
|test|524|132|57|112|223|2024-07-07 01:00:00+00:00|2024-12-31 18:00:00+00:00|

Excluded due to 120-hour boundary purging: **38 synthetic rows**. All source donor timestamps are distinct across revised splits, with chronological gaps. Class balance is no longer forced and is an expected trade-off of source-aware splitting.

## EDA issues that cannot responsibly be "fixed" by rewriting numbers
- The physical/biological gap between simulated and historical data is real: training synthetic wet-row fraction was 75%, whereas NASA-derived historical wetness was 5.09%. Preserve this difference and report a separate temporal historical proxy-risk evaluation.
- Historical NASA-derived risks are 24,990 No Risk, 1,288 Low, 26 Moderate, 0 High. Lack of historical High cases cannot be corrected without new independent evidence.
- The original README gives model parameters (Tmin 16.6°C; Topt 27.2°C; Tmax 30.3°C; Weibull scale 32.6h, shape 1.76) but the included Python generator **copies** scores from the absent upstream `Black_Sigatoka_Simulated_Training_Only.csv`, rather than containing the full generating score function. Do not claim full formula reconstruction.
- Synthetic humidity, rain, dewpoint and solar were conditionally generated from wet/dry scenarios, and wind/solar sourced from historical temperature-matched donors. They are not independent measured inputs; avoid using them to inflate baseline accuracy.
- Canopy temperature and historical sensor outcomes are unavailable; no imputation can create true sensor measurements.
- The wetness estimator (`RH >= 98% OR hourly rainfall >= 1 mm`) is an uncalibrated proxy and is not independently validated against plantation leaf-wetness sensors.
- One historical dew-point inconsistency is flagged in `quality_review/historical_dewpoint_review.csv` and `quality_review/BS_HISTORICAL_WITH_QC_FLAGS.csv`. Retained original values and recommend excluding the single flagged observation from dew-point-specific analyses pending source recheck, without affecting the chosen three baseline features.

## Appropriate next actions
1. Use the revised `BS_SOURCE_TIME_*.csv` splits for an initial leakage-conscious baseline, but recognize their source time does not turn synthetic scenarios into an actual chronological continuous wetness series.
2. Check label determinism and repeated 3-feature combinations. The earlier EDA found six combinations associated with differing scores; this needs further audit against missing original label generator. Avoid claiming a perfectly reproducible equation.
3. Keep training/validation/test separation fixed; do not inspect the held-out test for tuning. Use historical NASA periods only for clearly labeled proxy-risk behaviour checks, not independently verified infection accuracy.
4. Recover upstream original label generator if available; then design a fully reproducible synthetic scoring pipeline and document source-aware temporal grouping.
5. Gather genuine plantation microclimate, measured leaf wetness and confirmed disease observations for external disease-risk validation.

## Integrity and known limitations
- These files are derived, newly proposed development splits and are NOT an authoritative biological dataset.
- Purged temporal partitions reduce known donor-source overlap but cannot rule out synthetic formula leakage, shared model assumptions, or environmental autocorrelation beyond 120 hours.
- Source data remains untouched; no environment values have been fabricated or overwritten.
