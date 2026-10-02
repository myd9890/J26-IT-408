# Yellow Sigatoka environmental suitability dataset — methodological notes

Location: Bingiriya, Sri Lanka (7.59832 N, 79.93734 E). NASA POWER hourly weather 2022–2024 (UTC).

## Crucial interpretation
Risk score is a **project-designed relative environmental suitability index**, NOT an empirically calibrated infection probability, NOT measured disease severity. Risk categories are project-defined. No real disease occurrence labels are included. Synthetic rows are counterfactual NASA-analogue scenarios, NOT real plantation weather. No model trained on these labels can be claimed to predict actual field infection without independent disease observations.

## Research anchors
1. Bonfim Junior et al. (2024), DOI 10.52945/rac.v37i3.1862. Field associations: 13–21 C with >160 wet hours in preceding 14 days, OR >21–25 C with >130 wet hours. Regression R2=0.55 reported. Study conducted in Brazilian subtropical conditions on different banana cultivar.
2. Pérez & Fernández (1978), FAO AGRIS record 647245ed2c1d629bc9796815. Conidial germination min 8 C, optimum 25 C, max 37.5 C; these are laboratory germination limits, not field infection probabilities.
3. NASA POWER hourly point dataset, https://power.larc.nasa.gov/ . Gridded estimates, not field station observations.

## Explicit project-defined heuristic (NOT published equation)
T14 = 14-day mean air temperature. LW14 = estimated wet hours during previous 336 hours. C = current consecutive wet hours.
TempSuit = clip((T14-8)/17,0,1) if T14<=25, otherwise clip((37.5-T14)/12.5,0,1).
WetAnchor = 130 hours if 21<T14<=25, otherwise 160 hours.
WetSuit = clip((LW14-30)/(WetAnchor+45-30),0,1).
Continuity = clip(C/18,0,1).
RiskScore = 100 * TempSuit * (0.88*WetSuit + 0.12*Continuity).
Risk categories: No Risk <5; Low 5–<30; Moderate 30–<65; High >=65. All score cutoffs and interpolation parameters are **engineering choices**, not claims from Bonfim or Pérez. Favourable flag uses Bonfim published empirical thresholds separately.

## Leaf wetness proxy
Wet = RH>=95% OR (T - dewpoint <=1.5 C AND solar radiation <0.15 in original NASA units) OR rainfall >=0.5 mm/hr. This is a PROJECT ESTIMATION RULE, NOT calibrated against leaf-wetness sensors. NASA solar radiation field is preserved in its original source units; confirm units against the file header before converting.

## Data generation and leakage
Historical dataset retains genuine NASA environmental values and computes rolling 24h,72h,14d features; first 335 records excluded for warmup. Simulated scenarios each draw a 336-hour historical weather analogue, apply temperature offsets and create internally coherent RH/dewpoint/rain/solar multi-hour regimes, and retain ONLY the final hour. Each scenario ID appears in only one split; 1050/225/225 per class. The underlying NASA analogue windows CAN overlap across scenarios, so **the simulation splits are not fully independent of source weather**; score-reproduction evaluation must not be interpreted as out-of-sample field disease prediction.

## Recommended ML use
Train regression on risk_score with input features T14, LW14, consecutive wet hours and optionally the other environmental features; exclude risk_score, risk_level, favourable flag, temperature_suitability, wetness_suitability, metadata IDs, and origin. Compare model to the exact heuristic baseline (the baseline is deterministic and exact). For rigorous future validation, acquire observed Yellow Sigatoka onset/severity and measured leaf wetness, and evaluate on new site/year data. No guarantee against over/underfitting.

## Class balancing
Balanced SIMULATED files have 4200 train, 900 validation, 900 test with all four classes. Historical weather is NOT relabelled or artificially balanced.
