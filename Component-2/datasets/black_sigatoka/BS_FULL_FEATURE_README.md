# Black Sigatoka FULL-COLUMN four-class dataset — READ FIRST

## Which file should I train on?
`BS_FULL_FEATURE_SIMULATED_train.csv` has 4,200 rows, 1,050 per class, with temperature, humidity, rain, dew point, wind, solar radiation, estimated wetness, consecutive wetness hours, risk score and risk level. Matching 900-row validation and 900-row test files contain 225 per class. THESE ARE SYNTHETIC MATHEMATICAL SCENARIOS, NOT REAL WEATHER TIME SERIES OR DISEASE EVENTS. The NASA weather file has 26,304 historical hourly records and all real available environmental measurements.

## Honest caveat on the requested full columns
To give all columns without pretending to have measured data, each synthetic scenario uses its target temperature and wetness duration from the existing project-generated four-class scenarios; wet/dry RH is constructed consistently (wet >=98%, dry 57–88%), dew point is recomputed from T/RH via Magnus equation, rain is 0 for dry scenarios and 0 or >=1 mm/hr for wet ones, wind/solar use a temperature-matched NASA weather analogue (solar reduced for wet scenarios). Donor timestamp is disclosed. These are PLAUSIBLE constructed point scenarios, not a physically validated weather generator. The historical NASA file retains actual weather columns. Synthetic 24-hour mean temperature, RH and rainfall are intentionally blank because one isolated scenario cannot support an authentic preceding 24-hour history; 24/72h wet hours assume an uninterrupted wet spell. Canopy temperature is blank because it was not measured or reliably estimated.

## Targets and feature selection
Target: `risk_score` (0–100 relative infection response); derive `risk_level` from score (0 No Risk, >0–<25 Low, >=25–<75 Moderate, >=75 High). For a robust BASELINE model train on `air_temp_c`, `estimated_leaf_wet`, `consecutive_wetness_hours`. Other weather columns are useful for a subsequent wetness-estimation submodel, but were partly constructed conditionally on wetness in simulated data; adding them to the risk regressor will cause synthetic shortcut learning, not independently validated improvement. Never include risk_level, risk_score, split, scenario_id, donor timestamp or source metadata as inputs. No guaranteed absence of over/underfitting.

## Scientific model and provenance
Risk labels copied unchanged from prior `Black_Sigatoka_Simulated_Training_Only.csv`, generated with a PROJECT ADAPTATION of Bebber (2019), DOI 10.1098/rstb.2018.0269: temperature-dependent Weibull wet-spell response (Tmin 16.6, Topt 27.2, Tmax 30.3 C; scale 32.6 h, shape 1.76). This is not the paper's complete annual hourly spore-cohort infection-pressure calculation, and the 0/25/75 class cutoffs are project-defined. Jacome & Schuh (1992), DOI 10.1094/Phyto-82-515 supports temperature/wetness dependence. NASA gridded estimates are not plantation sensors. Historical wetness is inferred by RH>=98% OR hourly rain>=1 mm/h (uncalibrated proxy). No actual disease-occurrence probabilities or confirmed disease outcomes are present.

## Evaluation
Synthetic splits have distinct scenario IDs, but all are generated from the same mathematical model. Report synthetic performance as equation imitation only. Evaluate separately on chronological NASA splits; its historical data contains 24,990 No Risk, 1,288 Low, 26 Moderate and 0 High records under the selected wetness proxy. A real-weather High Risk accuracy cannot be estimated. Obtain measured leaf wetness and independently confirmed disease outcomes for field validation.

## Future outlook
Future 7–14-day environmental risk requires HOURLY WEATHER FORECASTS and the same wetness estimation and historical feature pipeline. The current NASA historical CSV is not a future forecast.
