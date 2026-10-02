# Black Sigatoka Feature Engineering Review

## Scope and evidence

This review used the revised training and validation CSVs only (4,501 and 937 rows). The held-out test CSV was not loaded. It also checked the feature contract, `src/black_sigatoka_preprocessing.py`, the EDA and preprocessing reports, `datasets/black_sigatoka/build_full_feature_four_class.py`, the Random Forest and XGBoost documentation, and the saved feature audits. No datasets, models, or existing reports were changed; no model was trained.

The revised modeling rows are explicitly marked `SYNTHETIC_SCENARIO_NOT_MEASURED`. Their `timestamp_utc` is blank in every row. `weather_analogue_timestamp_utc` records the historical weather donor used for scenario construction; it is not the observation time of a continuous wetness series. The current inputs must therefore not be described as plantation sensor measurements.

## Current three model inputs

| Feature | Raw / Estimated / Engineered | Used in Model | Reason |
|---|---|---:|---|
| `air_temp_c` | Synthetic environmental input / air-temperature proxy | Yes | Complete and numeric in both partitions (train range 17.011-33.866 C; validation 17.029-33.988 C). Air temperature is measurable by an IoT temperature sensor, but these scenario records are not field readings. |
| `estimated_leaf_wet` | Estimated binary wet/dry indicator | Yes | Complete, values are 0 or 1. The current scenario method says `SCENARIO_SET_RH_GE_98_WET_ELSE_DRY; NOT SENSOR`; it is not a leaf-wetness sensor measurement. |
| `consecutive_wetness_hours` | Engineered wet-spell duration | Yes | Complete, integer values 0-120. The available builder copies `wetness_duration_hours` from a generated scenario. The indicator equals 1 exactly when this duration is greater than zero in all reviewed train/validation rows. However, no scenario timestamp or chronological sensor series exists here to independently reconstruct the duration. |
| `risk_score` | Synthetic target, not an input feature | Target | The input contract correctly keeps this as the target only. |
| `risk_level`, `scenario_id`, `weather_analogue_timestamp_utc`, `label_source`, split/provenance fields | Target-derived label or metadata | No | The input contract excludes labels, identifiers, donor times, and provenance fields to avoid direct leakage. |

All three approved inputs are numeric, finite, and have zero missing values in train and validation. Their use is reasonable for a preliminary synthetic rule-imitation baseline. For IoT use, temperature can be measured, wetness should be clearly labeled as either measured or estimated, and wet-spell duration must be computed from a real chronological wetness history available at prediction time.

### How to calculate a wetness streak from IoT readings

Sort readings for the same sensor/leaf by timestamp. If the current hourly wetness reading is wet, set the streak to the previous streak plus one hour; if dry, reset it to zero. Define how missing hours, irregular sampling, sensor resets, and timezone changes are handled. Calculate the value only from readings available at the prediction time. A single instantaneous wet/dry reading cannot determine a consecutive duration.

The data do not establish whether `consecutive_wetness_hours` means an elapsed streak known at prediction time or the eventual duration of a completed wet spell. That timing is important: if the supplied duration includes later hours, it would be future information for a real-time prediction. The timestamp-free synthetic scenarios prevent this audit from resolving that question.

## Existing feature engineering and additional candidates

The project already has engineered values. The full-feature builder takes `wetness_duration_hours` from the generated synthetic scenarios. It assigns `wet_hours_previous_24h` as `min(duration, 24)` and `wet_hours_previous_72h` as `min(duration, 72)` for wet scenarios (zero for dry scenarios). In the synthetic scenario files these fields are complete, and the current revised train/validation values match those capped-duration formulas exactly. They are scenario summaries, not independent rolling counts from prior hourly readings.

The existing EDA also examined chronological historical NASA-derived data. It found the named `wet_hours_previous_24h`, `wet_hours_previous_72h`, rainfall, and mean-temperature/humidity summaries matched rolling windows that include the current hour; previous-only versions did not match. This is not automatically leakage if a prediction is made after the current-hour reading arrives, but the time convention should be stated and the `previous` names are ambiguous. These historical-source audits do not make the current synthetic rows chronological sensor records.

| Feature | Exists in revised files | Train / validation completeness | Type and source notes | Recommendation |
|---|---|---|---|---|
| Wet hours in previous 24 hours | Yes | Complete / complete | Synthetic `min(duration, 24)` summary; same information is contained in the current streak up to its cap. | Do not add in this experiment. Redundant and not an actual trailing-window count here. |
| Wet hours in previous 72 hours | Yes | Complete / complete | Synthetic `min(duration, 72)` summary; redundant with wet-spell duration. | Do not add in this experiment. |
| Mean temperature during wet periods | No exact field | Not available / not available | `mean_temp_previous_24h_c` is a different feature and is entirely missing in both revised partitions. Historical EDA has a rolling 24-hour mean, not a mean restricted to wet periods. | Consider only after a chronological source series exists and the window is defined. |
| Rainfall in previous 24 hours | Yes (`rain_previous_24h_mm`) | 100% missing / 100% missing | The synthetic builder intentionally leaves it blank. Historical NASA-derived rolling rainfall exists in the separate historical source, with 23 warm-up rows missing, but it is not plantation rain-gauge data. | Do not impute or manufacture it for these scenarios. Possible future sensor feature after sourcing chronological rainfall. |
| Relative humidity | Yes | Complete / complete | Current synthetic RH was constructed by wet/dry scenario (wet >=98%, dry 57-88%); separate historical values are gridded weather. It is not independent of the wetness assignment. | Could support a future wetness estimator; do not add now as it may teach a synthetic shortcut. |
| Canopy temperature | Yes | 100% missing / 100% missing | No canopy temperature values are supplied; the full historical file is also documented as blank for this field. | Do not manufacture. Could be measured with an appropriate canopy sensor later. |
| Dew-point depression | Yes | Complete / complete | Derived as air temperature minus dew-point temperature. In synthetic scenarios dew point was recomputed from constructed temperature and humidity, so the feature is redundant/constructed. The EDA flags one historical dew-point inconsistency. | Do not add now; reassess after source/provenance checks. |
| Air temperature | Yes | Complete / complete | Current scenarios identify it as an air-temperature proxy; not field-measured in these rows. | Keep as the approved baseline input; collect actual local sensor data for IoT validation. |
| Estimated leaf wetness | Yes | Complete / complete | Synthetic wet/dry scenario indicator; historical wetness is separately estimated from a humidity/rain proxy and is not a direct sensor reading. | Keep for the current baseline, but preserve the measured-versus-estimated distinction. |
| Consecutive wetness hours | Yes | Complete / complete | Synthetic wet-spell duration carried from the scenario source; no chronological sequence in these rows. | Keep for reproducing this synthetic task, but verify its prediction-time availability before real-time or test claims. |

The historical comparison reports 23 missing warm-up rows for historical 24-hour features and 71 for 72-hour wetness summaries. The historical file is NASA-gridded weather, not plantation sensor data; the EDA notes that extraction/location provenance is not fully verified. It does not provide confirmed disease outcomes.

## Relationships, leakage, and calculation review

The selected inputs are associated with the synthetic score because the scenario labels are described as a project adaptation of a temperature and wet-spell response. The README attributes the modeling idea to published temperature/wetness research, but the upstream continuous label generator is unavailable. Six repeated three-input combinations have slightly different risk scores. Therefore the models can imitate the synthetic scoring process, but the exact dependency cannot be independently reproduced from this project.

No direct target column, risk category, scenario ID, donor timestamp, or label metadata is included among the approved inputs. No confirmed conventional target leakage or arithmetic error was found in the three selected columns. However, time leakage cannot be ruled out for the wet-spell-duration feature because its observation-time semantics are absent. Treat that as an unresolved prediction-time risk, not proof that the current rows contain future data.

A useful external biological rationale exists for considering temperature and leaf-wetness duration: Jacome and Schuh report that Black Sigatoka severity was influenced by these factors. Bebber's published climate model also represents hourly infection conditions using canopy wetness and temperature. Those papers do not validate this project's proxy rules, synthetic labels, or fitted scores. See [Jacome and Schuh (1992)](https://doi.org/10.1094/Phyto-82-515) and [Bebber (2019)](https://doi.org/10.1098/rstb.2018.0269).

## Decisions

- Feature engineering has already been performed: wetness-duration/streak values and rolling summaries exist, and dew-point depression is derived. The current approved preprocessing module validates and selects the three inputs; it does not calculate new rolling features.
- Keep the current three features for the existing synthetic baseline. They are complete and correspond to the intended synthetic score inputs, but they are not evidence of validated disease-risk prediction.
- Do not add the other candidates now. Several are missing, constructed from the same synthetic wetness assumptions, redundant with duration or humidity, or unavailable from the current scenario chronology.
- No changes or retraining are recommended for Random Forest or XGBoost based on this review. Their scores describe agreement with synthetic labels.
- Before calling this a real-time IoT model, confirm `consecutive_wetness_hours` is computed only from past/current sensor readings. If it is the completed future wet-spell duration, it must not be used for a real-time prediction.
- A one-time held-out synthetic test evaluation can assess performance on the reserved synthetic records, but it cannot resolve the wetness timing uncertainty or establish field disease accuracy. Confirm the duration semantics first if the intended claim is real-time prediction; keep the test set untouched until model selection is final.

## Existing files used as evidence

`reports/black_sigatoka/EDA_Report.md`, `reports/black_sigatoka/preprocessing/Preprocessing_Report.md`, `reports/black_sigatoka/final_safeguards/model_input_contract.json`, `reports/black_sigatoka/historical_rolling_consistency.csv`, `reports/black_sigatoka/training_historical_comparison.csv`, `reports/black_sigatoka/three_input_conflicting_scores.csv`, `datasets/black_sigatoka/BS_FULL_FEATURE_README.md`, and `datasets/black_sigatoka/build_full_feature_four_class.py`.
