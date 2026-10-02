# Black Sigatoka EDA Report

Scope: Cavendish banana environmental risk research, Bingiriya, Sri Lanka. All results below are computed from the supplied CSVs. Labels are research-informed mathematical proxies, not confirmed disease observations.

## Dataset overview

| index | rows | columns |
| --- | --- | --- |
| train | 4200 | 25 |
| validation | 900 | 25 |
| test | 900 | 25 |
| historical | 26304 | 25 |

All four files have the same column names. CSV-inferred types differ where columns are entirely empty. No README, scoring generator, or threshold specification was available in the inspected project.

## Risk distribution

| index | No Risk | Low | Moderate | High |
| --- | --- | --- | --- | --- |
| train | 1050 | 1050 | 1050 | 1050 |
| validation | 225 | 225 | 225 | 225 |
| test | 225 | 225 | 225 | 225 |
| historical | 24990 | 1288 | 26 | 0 |

Training score range: 0.0000–99.9950; historical score range: 0.0000–47.2873. Simulated splits each allocate 25% to every level. This class balance is constructed and is not an estimate of disease prevalence.

Hypothesized bins: zero = No Risk, (0,25) = Low, [25,75) = Moderate, [75,100] = High. These are inferred, not documented or independently verified. Boundary conventions, near-zero tolerance, and the continuous label formula remain unresolved.

| dataset | missing_score | outside_0_100 | unknown_or_missing_level | hypothesized_rule_mismatches | exact_score_0 | exact_score_25 | exact_score_75 | exact_score_100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | 0 | 0 | 1050 | 0 | 0 | 0 |
| validation | 0 | 0 | 0 | 0 | 225 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 225 | 0 | 0 | 0 |
| historical | 0 | 0 | 0 | 0 | 24990 | 0 | 0 | 0 |

## Data quality

Missing values are retained. Missingness includes structural unavailable fields and historical rolling warmup. See missing_values.csv for all columns and percentages.

| dataset | column | missing | percent |
| --- | --- | --- | --- |
| train | timestamp_utc | 4200 | 100.0 |
| train | rain_previous_24h_mm | 4200 | 100.0 |
| train | mean_temp_previous_24h_c | 4200 | 100.0 |
| train | mean_rh_previous_24h_pct | 4200 | 100.0 |
| train | canopy_temperature_c | 4200 | 100.0 |
| validation | timestamp_utc | 900 | 100.0 |
| validation | rain_previous_24h_mm | 900 | 100.0 |
| validation | mean_temp_previous_24h_c | 900 | 100.0 |
| validation | mean_rh_previous_24h_pct | 900 | 100.0 |
| validation | canopy_temperature_c | 900 | 100.0 |
| test | timestamp_utc | 900 | 100.0 |
| test | rain_previous_24h_mm | 900 | 100.0 |
| test | mean_temp_previous_24h_c | 900 | 100.0 |
| test | mean_rh_previous_24h_pct | 900 | 100.0 |
| test | canopy_temperature_c | 900 | 100.0 |
| historical | wet_hours_previous_24h | 23 | 0.08743917274939172 |
| historical | wet_hours_previous_72h | 71 | 0.26992092457420924 |
| historical | rain_previous_24h_mm | 23 | 0.08743917274939172 |
| historical | mean_temp_previous_24h_c | 23 | 0.08743917274939172 |
| historical | mean_rh_previous_24h_pct | 23 | 0.08743917274939172 |
| historical | canopy_temperature_c | 26304 | 100.0 |
| historical | weather_analogue_timestamp_utc | 26304 | 100.0 |

Nonzero validity/consistency flags (empty table means none under the explicit checks):

| index | dataset | check | flagged_rows |
| --- | --- | --- | --- |
| 159 | historical | dew point above air + 0.02°C | 1 |
| 160 | historical | dew depression discrepancy > 0.02°C | 1 |

Historical dew-point inconsistency examples:

| index | scenario_id | timestamp_utc | air_temp_c | dew_point_c | dew_point_depression_c |
| --- | --- | --- | --- | --- | --- |
| 18119 | NASA_18119 | 2024-01-25 23:00:00+00:00 | 20.37 | 20.41 | 0.0 |

| dataset | full_row_duplicates_after_first | duplicate_scenario_ids_after_first | environment_duplicates_after_first | three_input_duplicates_after_first |
| --- | --- | --- | --- | --- |
| train | 0 | 0 | 0 | 36 |
| validation | 0 | 0 | 0 | 3 |
| test | 0 | 0 | 0 | 4 |
| historical | 0 | 0 | 0 | 23773 |

IQR screening is saved in iqr_outlier_screen.csv; flags do not establish bad observations. Zero-inflated rainfall and wetness can produce narrow or zero IQRs. No rows or values were altered.

## Leakage and split independence

risk_level is target-derived. scenario_id encodes the risk class and must not be a predictor. Keep risk_score only as the target. Exclude provenance, split, and identifier fields from predictors. Temperature/wetness associations with risk reflect the synthetic scoring mechanism, not evidence of biological predictive validity.

| index | left | right | full_row_left_rows | full_row_right_rows | excluding_split_left_rows | excluding_split_right_rows | scenario_id_left_rows | scenario_id_right_rows | environment_left_rows | environment_right_rows | three_inputs_left_rows | three_inputs_right_rows | three_inputs_and_score_left_rows | three_inputs_and_score_right_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | train | validation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 9 | 10 | 9 |
| 1 | train | test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 | 12 | 12 | 11 |
| 2 | validation | test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 | 1 | 1 |

| index | left | right | left_rows | exact_shared_analogue_left_rows | left_rows_within_24h | left_rows_within_72h | left_rows_within_120h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | train | validation | 4200 | 1134 | 3330 | 4082 | 4174 |
| 1 | train | test | 4200 | 1108 | 3276 | 4059 | 4158 |
| 2 | validation | test | 900 | 243 | 707 | 871 | 892 |

| dataset | missing_or_invalid_analogue_times | unique_analogues | analogues_matching_historical_rows | historical_split_differs_from_simulated |
| --- | --- | --- | --- | --- |
| train | 0 | 2834 | 4200 | 1061 |
| validation | 0 | 728 | 900 | 762 |
| test | 0 | 720 | 900 | 832 |

No shared scenario IDs or exact complete environmental records were found between simulated splits. However, 1134 training rows reuse validation analogue timestamps and 1108 reuse test analogue timestamps. These observations are not additive because a training row may match both splits.

Exact overlap counts are directional row counts. Feature matches need not be duplicate observations. Analogue proximity uses strict distances <24, <72, and <120 hours and indicates possible shared context; it does not prove actual shared scenario windows. Random scenario splits cannot be assumed independent of historical source weather. Source generation and grouping must be recovered before deciding how to evaluate.

Historical chronology and boundaries:

| index | rows | invalid_timestamps | duplicate_timestamps_after_first | start_utc | end_utc | non_hourly_steps_in_file_order |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 26304 | 0 | 0 | 2022-01-01 00:00:00+00:00 | 2024-12-31 23:00:00+00:00 | 0 |

| block | split | start | end | rows |
| --- | --- | --- | --- | --- |
| 1 | excluded_warmup | 2022-01-01 00:00:00+00:00 | 2022-01-03 23:00:00+00:00 | 72 |
| 2 | train | 2022-01-04 00:00:00+00:00 | 2023-12-31 23:00:00+00:00 | 17448 |
| 3 | excluded_warmup | 2024-01-01 00:00:00+00:00 | 2024-01-03 23:00:00+00:00 | 72 |
| 4 | validation | 2024-01-04 00:00:00+00:00 | 2024-06-30 23:00:00+00:00 | 4296 |
| 5 | excluded_warmup | 2024-07-01 00:00:00+00:00 | 2024-07-03 23:00:00+00:00 | 72 |
| 6 | test | 2024-07-04 00:00:00+00:00 | 2024-12-31 23:00:00+00:00 | 4344 |

Rolling checks compare both inclusive-current and previous-only window interpretations:

| index | feature | excluded_current_hour | comparable_rows | mismatches_tolerance_1e-5 |
| --- | --- | --- | --- | --- |
| 0 | wet_hours_previous_24h | False | 26281 | 0 |
| 1 | wet_hours_previous_24h | True | 26280 | 2244 |
| 2 | wet_hours_previous_72h | False | 26233 | 0 |
| 3 | wet_hours_previous_72h | True | 26232 | 2358 |
| 4 | rain_previous_24h_mm | False | 26281 | 0 |
| 5 | rain_previous_24h_mm | True | 26280 | 21788 |
| 6 | mean_temp_previous_24h_c | False | 26281 | 0 |
| 7 | mean_temp_previous_24h_c | True | 26280 | 26046 |
| 8 | mean_rh_previous_24h_pct | False | 26281 | 0 |
| 9 | mean_rh_previous_24h_pct | True | 26280 | 26247 |

All five historical rolling-summary checks match the inclusive-current-hour interpretation on comparable rows. The names previous_24h/previous_72h therefore do not establish a strictly previous-only window. Historical split blocks include 72 excluded warmup hours at each partition start; do not discard these protections in a future design.

## Historical distribution comparison

| feature | train_n | historical_n | train_mean | historical_mean | train_median | historical_median | train_min | train_max | historical_min | historical_max | train_outside_historical_range_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| air_temp_c | 4200 | 26304 | 24.246660476190474 | 27.022466545012165 | 23.941000000000003 | 26.86 | 17.015 | 33.988 | 19.56 | 36.33 | 14.11904761904762 |
| estimated_leaf_wet | 4200 | 26304 | 0.75 | 0.05090480535279805 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 |
| consecutive_wetness_hours | 4200 | 26304 | 44.105714285714285 | 0.2568430656934307 | 39.0 | 0.0 | 0.0 | 120.0 | 0.0 | 26.0 | 58.30952380952381 |
| relative_humidity_pct | 4200 | 26304 | 92.34773095238094 | 83.26911724452556 | 98.56 | 86.21 | 57.02 | 99.7 | 34.74 | 100.0 | 0.0 |
| rainfall_mm_hr | 4200 | 26304 | 0.3770047619047619 | 0.22195369525547443 | 0.0 | 0.06 | 0.0 | 5.99 | 0.0 | 9.32 | 0.0 |
| risk_score | 4200 | 26304 | 37.88368288095238 | 0.21602288245133822 | 25.03525 | 0.0 | 0.0 | 99.995 | 0.0 | 47.2873 | 38.30952380952381 |

Wet-row fraction: training 75.00%, historical 5.09%. Historical No Risk: 95.00%; historical High rows: 0.

Balanced synthetic scenarios cover different temperature/wetness conditions from the historical baseline. The historical data are gridded weather with estimated, uncalibrated wetness; air temperature is a canopy-temperature proxy. They are not field disease observations or an independent disease test set.

## Preliminary feature assessment

| index | missing_train | unique_train | train_min | train_max | historical_min | historical_max | pearson_with_risk | spearman_with_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| air_temp_c | 0.0 | 3609.0 | 17.015 | 33.988 | 19.56 | 36.33 | 0.2495935529497897 | 0.19180304454229066 |
| estimated_leaf_wet | 0.0 | 2.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.5710731676042229 | 0.7559289632452129 |
| consecutive_wetness_hours | 0.0 | 121.0 | 0.0 | 120.0 | 0.0 | 26.0 | 0.7659405384925577 | 0.8568201840346539 |

Wet flag versus positive consecutive wetness mismatches by file: {'train': 0, 'validation': 0, 'test': 0, 'historical': 0}. Repeated three-input combinations with more than one score across simulated files: 6 groups.

Associations and the joint temperature–wetness figure support examining a nonlinear response. Marginal correlations cannot rank causal importance. Clipped wet-window features may be redundant with consecutive duration. No model was trained and no features were selected automatically.

## Recommendations for the next stage

1. Recover the generator, exact score formula, thresholds, rounding policy, and scientific references. Confirm proxy wetness and rolling-window conventions; resolve flagged inconsistencies before changing data.

2. Define whether the task estimates contemporaneous proxy risk or forecasts a future horizon. Confirm all candidate inputs are available at that prediction time.

3. Plan source-aware grouped/temporal validation using analogue identities and weather blocks, with a gap appropriate to every feature window (audit includes up to 120-hour scenario spells). Audit again after any future split redesign; do not mix simulated rows and their source history as independent evaluation records.

4. Review structural missingness, all-empty variables, redundant wetness features, and outlier context. Any future imputation or learned transformation must be fitted on training data only. No such processing was done in this EDA.

5. Obtain local measured leaf wetness, canopy conditions, and confirmed disease outcomes for biological calibration and external validation. High accuracy against synthetic labels would demonstrate reproduction of the scoring function, not real disease prediction accuracy.

## Artifacts and verification

The notebook contains all code, explanations, outputs, and figures. Detailed numerical tables are saved as CSV alongside this report. Figures:

- [environment_and_score_distributions.png](environment_and_score_distributions.png)

- [risk_level_distributions.png](risk_level_distributions.png)

- [distribution_boxplots.png](distribution_boxplots.png)

- [environment_correlation_heatmap.png](environment_correlation_heatmap.png)

- [features_versus_risk.png](features_versus_risk.png)

- [joint_temperature_wetness_risk.png](joint_temperature_wetness_risk.png)

- [training_historical_ecdfs.png](training_historical_ecdfs.png)

Source SHA-256 checksums are saved in source_manifest.json and checked on every complete EDA run. EDA imports are consolidated and Markdown is shortened; all EDA calculations are retained. No cleaning, preprocessing, or model training was performed.

Unverified: documented label thresholds/formula; field disease validity; NASA extraction provenance/location; actual shared generator weather windows. CSV metadata alone cannot establish these.
