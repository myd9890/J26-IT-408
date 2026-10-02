# Black Sigatoka Post-Fix Validation Report

Validated with the project `.venv` against the three files in `datasets/black_sigatoka/revised_splits/`. The source CSVs were read only. This report evaluates data integrity and leakage controls; it does not validate biological disease prediction.

## Decision

**Conditionally suitable for preliminary ML:** use the revised splits only for an initial synthetic-risk baseline with `air_temp_c`, `estimated_leaf_wet`, and `consecutive_wetness_hours` as predictors and `risk_score` as the target. Keep the revised split assignment fixed and do not tune on the test set.

**Not suitable for claiming field disease performance:** the target is a simulated mathematical risk response, the full generating function is unavailable, and there are no confirmed disease observations. Performance measures equation imitation under the constructed scenarios.

## Validation status

| Check | Status | Result |
|---|---|---|
| Revised dimensions and schema | PASS | Train 4,501 × 25; validation 937 × 25; test 524 × 25. All have the same expected columns. |
| Missingness | PASS | Only the five documented all-missing synthetic fields are missing. |
| Score range and four classes | PASS | Every score is within 0–100 and all four classes occur in every split. |
| Documented risk thresholds | PASS | Zero mismatches across 5,962 rows. |
| Exact row/ID/environment duplicates across splits | PASS | None found. |
| Cross-split analogue timestamp overlap | PASS | No exact timestamp overlap. |
| 120-hour temporal exclusion | PASS | No cross-split donor times within 120 hours; observed boundary gaps exceed 120 hours. |
| Stored rolling-window separation | PASS | Known windows are 24/72 hours; maximum wet-spell duration is 120 hours. Observed cross-split gaps exceed these. |
| Longer dependence control | UNCERTAIN | Environmental autocorrelation beyond 120 hours was not measured or ruled out. |
| Original-to-revised record integrity | PASS | 5,962 retained records match original values exactly outside the intentionally changed `split`; 38 records were excluded. |
| Leakage safety of the complete 25-column files | FAIL | `scenario_id` encodes class, `risk_level` is target-derived, and multiple metadata/constructed fields must be excluded. |
| Three-input target determinism | FAIL | Six identical baseline-input combinations retain slightly different scores. |
| Historical dew-point handling | PASS | The one inconsistent NASA row remains unchanged and is explicitly flagged in the QC copy. |
| Biological/field validation | UNCERTAIN | No plantation sensors, measured leaf wetness, or confirmed disease outcomes are available. |

## 1. Revised files

All three files contain these 25 columns:

`scenario_id`, `record_type`, `timestamp_utc`, `air_temp_c`, `relative_humidity_pct`, `rainfall_mm_hr`, `dew_point_c`, `dew_point_depression_c`, `wind_speed_m_s`, `solar_radiation_mj_hr`, `estimated_leaf_wet`, `consecutive_wetness_hours`, `wet_hours_previous_24h`, `wet_hours_previous_72h`, `rain_previous_24h_mm`, `mean_temp_previous_24h_c`, `mean_rh_previous_24h_pct`, `canopy_temperature_c`, `temperature_input_type`, `wetness_estimation_method`, `risk_score`, `risk_level`, `label_source`, `split`, `weather_analogue_timestamp_utc`.

| Split | Rows | Score range | No Risk | Low | Moderate | High |
|---|---:|---:|---:|---:|---:|---:|
| Train | 4,501 | 0.0000–99.9950 | 1,037 | 1,300 | 1,144 | 1,020 |
| Validation | 937 | 0.0000–99.9939 | 323 | 136 | 235 | 243 |
| Test | 524 | 0.0000–99.9946 | 132 | 57 | 112 | 223 |

In every split, `timestamp_utc`, `rain_previous_24h_mm`, `mean_temp_previous_24h_c`, `mean_rh_previous_24h_pct`, and `canopy_temperature_c` are 100% missing. No other column has missing values. This matches `verification_summary.json` and the documented point-scenario design.

## 2. Risk labels

The README defines:

- `risk_score == 0`: No Risk
- `0 < risk_score < 25`: Low
- `25 <= risk_score < 75`: Moderate
- `75 <= risk_score <= 100`: High

All 5,962 revised records follow these thresholds, with no missing, unknown, or out-of-range target values. This verifies label-to-bin consistency, not the underlying continuous risk formula.

## 3. Duplicates and temporal separation

There are no duplicate complete rows, duplicate `scenario_id` values, exact cross-split records after ignoring `split`, or exact cross-split environmental records. The three-input vectors still repeat across splits: train–validation share 12 unique combinations, train–test 13, and validation–test 4. Such matches are expected for a small constructed input space but weaken strict sample independence.

Donor timestamps are reused within splits: train contains 2,583 unique timestamps for 4,501 rows, validation 699 for 937 rows, and test 484 for 524 rows. No donor timestamp occurs in more than one revised split.

| Split pair | Exact shared donor times | Minimum distance | Times within 120 h |
|---|---:|---:|---:|
| Train–validation | 0 | 265 h | 0 |
| Train–test | 0 | 4,657 h | 0 |
| Validation–test | 0 | 267 h | 0 |

Relative to the documented boundaries, the closest retained donors are 144 hours before and 121 hours after 1 January 2024, and 122 hours before and 145 hours after 1 July 2024. The intended 120-hour exclusion on both sides therefore passed.

`wet_hours_previous_24h` and `wet_hours_previous_72h` exactly equal consecutive duration clipped to 24 and 72 hours. The longest `consecutive_wetness_hours` value is 120. The observed 265/267-hour cross-split gaps prevent overlap for these declared lookbacks. A longer purge is not required by any stored rolling feature, but independence beyond 120 hours remains uncertain because weather autocorrelation was not estimated. Also, these wet-window fields are stylized from one scenario duration, not reconstructed historical sequences.

## 4. Original-to-revised integrity

The original three files contain 6,000 unique scenarios. Revised files retain 5,962 and exclude the documented 38 boundary scenarios. Every revised ID exists in the originals. A raw CSV-string comparison by `scenario_id` found zero changed cells across all columns except `split`, whose change is intentional. Environmental measurements, constructed environmental values, risk scores, labels, IDs, and analogue timestamps were preserved.

The revised `split` value agrees with its filename for every row. Checksums of all original, revised, historical, and QC files were unchanged during validation.

## 5. Leakage review

The complete revised tables are unsafe as direct model matrices:

- `scenario_id` reveals the risk class for all 5,962 rows through prefixes such as `SIM_LOW` and `SIM_HIGH`.
- `risk_level` is derived directly from `risk_score` and must never be a predictor.
- `risk_score` is the regression target and must be removed from `X`.
- `split` and `weather_analogue_timestamp_utc` reveal partition/time information.
- `record_type`, `temperature_input_type`, `wetness_estimation_method`, and `label_source` are provenance metadata rather than predictors.
- `wet_hours_previous_24h` and `wet_hours_previous_72h` are exact clipped transformations of `consecutive_wetness_hours` and add deterministic redundancy.
- Humidity, rain, dew point, and solar values were constructed conditionally on scenario wetness; using them in the risk regressor can create synthetic shortcuts.
- The five all-missing columns contain no learnable information.

The documented three-feature baseline avoids the direct target and identifier leakage above. It cannot avoid learning the synthetic scoring relationship, which is the limited purpose of this preliminary experiment.

## 6. Six repeated input combinations

All six conflicts identified earlier remain in the revised records. Each pair has the same `air_temp_c`, `estimated_leaf_wet`, and `consecutive_wetness_hours`, but two slightly different `risk_score` values:

| Temperature °C | Wet | Hours | Scores | Spread | Class | Revised placement |
|---:|---:|---:|---:|---:|---|---|
| 20.624 | 1 | 108 | 60.8005, 60.8106 | 0.0101 | Moderate | Train / validation |
| 22.185 | 1 | 20 | 11.6491, 11.6515 | 0.0024 | Low | Train / train |
| 22.604 | 1 | 11 | 5.1169, 5.1175 | 0.0006 | Low | Train / validation |
| 23.297 | 1 | 37 | 44.4125, 44.4167 | 0.0042 | Moderate | Train / validation |
| 28.929 | 1 | 47 | 75.6192, 75.6225 | 0.0033 | High | Train / train |
| 28.962 | 1 | 65 | 91.4572, 91.4577 | 0.0005 | High | Validation / test |

The differences are small and never change class, but they prove that `risk_score` is not perfectly deterministic from the three stored baseline columns. The restored generator rounds `air_temp_c` to three decimals while copying pre-existing scores, so lost upstream precision is a plausible explanation. The absent upstream score generator prevents confirmation. This should be documented as small irreducible label ambiguity; do not delete or rewrite these records without the authoritative source.

## 7. Historical dew point

One historical row remains inconsistent: `NASA_18119`, 2024-01-25 23:00 UTC, has air temperature 20.37°C, dew point 20.41°C, and stored depression 0.0°C. It appears once in `historical_dewpoint_review.csv`, and only that row has `qc_dewpoint_inconsistency = 1` in `BS_HISTORICAL_WITH_QC_FLAGS.csv`.

All 25 original historical columns in the QC copy match the source numerically within 1e-12; the QC file adds only the flag column. The source NASA CSV was not changed. Minor CSV floating-point serialization differences do not alter values at scientific precision.

## 8. Unresolved limitations and recommendations

1. The full continuous label function is unavailable, so exact target reproducibility and the six small score conflicts remain unresolved.
2. Donor-time splitting removes known timestamp overlap, but donor time is mainly provenance for matched weather values; the scenarios are not a continuous chronological disease series.
3. All splits share one synthetic scoring mechanism. Random Forest and XGBoost can achieve strong scores by reproducing it; this does not demonstrate disease prediction.
4. Repeated baseline input vectors cross splits, and donor timestamps repeat within each split. Report these dependencies with evaluation results.
5. Class proportions differ substantially across revised splits. Use regression metrics plus class-stratified summaries, and do not infer field prevalence from them.
6. NASA weather is gridded, leaf wetness is an uncalibrated proxy, synthetic wetness prevalence differs sharply from historical data, and historical data contain no High Risk cases.
7. No plantation-level canopy measurements, leaf-wetness sensor validation, or confirmed Black Sigatoka outcomes are available.
8. A longer temporal buffer is not demanded by the stored 24/72/120-hour features, but no analysis establishes that 120 hours removes all environmental autocorrelation.

Proceed with a leakage-controlled preliminary baseline only. Treat validation/test results as performance on held-out synthetic scenarios grouped by donor period. External field evaluation remains necessary before making claims about real Black Sigatoka risk prediction.
