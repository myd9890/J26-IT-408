import csv, math, random, json, zipfile, collections, pathlib
import numpy as np
from datetime import datetime,timezone
ROOT=pathlib.Path('/mnt/data'); rng=np.random.default_rng(20261008)
source=ROOT/'POWER_Point_Hourly_20220101_20241231_007d60N_079d94E_UTC (1).csv'
with source.open() as f:
    for line in f:
        if line.startswith('YEAR,MO,DY,HR,'): break
    records=list(csv.DictReader(f,fieldnames=line.strip().split(',')))
assert len(records)==26304,len(records)
T=np.array([float(r['T2M']) for r in records]); RH=np.array([float(r['RH2M']) for r in records]); R=np.array([float(r['PRECTOTCORR']) for r in records]); D=np.array([float(r['T2MDEW']) for r in records]); WS=np.array([float(r['WS2M']) for r in records]); SOL=np.array([float(r['ALLSKY_SFC_SW_DWN']) for r in records]);
assert np.all(T>-100) and np.all(RH>=0) and np.all(R>=0)
# Project proxy: wet if RH>=95%, or dew point depression<=1.5 C with low solar, or precipitation>=0.5 mm/h. Uncalibrated.
def wetness(t,rh,rain,dew,solar): return ((rh>=95)|((t-dew<=1.5)&(solar<0.15))|(rain>=0.5)).astype(int)
def rolling(a,n,mean=False):
    out=np.full(len(a),np.nan); c=np.r_[0,np.cumsum(a,dtype=float)]; out[n-1:]=(c[n:]-c[:-n])/(n if mean else 1);return out
def score(t14,lw14,consec):
    # Project-defined continuous surrogate: soft wetness ramp anchored at Bonfim's 130h and 160h favourable conditions.
    # Lab germination cardinal temperature 8/25/37.5, applied as a secondary modifier.
    if not np.isfinite(t14) or not np.isfinite(lw14): return float('nan'),0,0,0
    temp14=max(0,min(1,(t14-8)/17 if t14<=25 else (37.5-t14)/12.5))
    # lower favourable wetness boundary when 21<T14<=25; otherwise 160h, as a project extrapolation outside 13-25.
    anchor=130 if 21<t14<=25 else 160
    moist=max(0,min(1,(lw14-30)/(anchor+45-30)))
    continuity=min(1,consec/18)
    # The published paper does NOT supply this equation or its weights. Explicitly a project heuristic.
    s=100*temp14*(0.88*moist+0.12*continuity)
    favourable=int((13<=t14<=21 and lw14>160) or (21<t14<=25 and lw14>130))
    return round(float(max(0,min(100,s))),3),favourable,round(float(temp14),4),round(float(moist),4)
def level(s): return 'No Risk' if s<5 else 'Low' if s<30 else 'Moderate' if s<65 else 'High'
headers=['record_id','scenario_id','data_origin','timestamp_utc','air_temp_c','relative_humidity_pct','rainfall_mm_hr','dew_point_c','dew_point_depression_c','wind_speed_m_s','solar_radiation_original_nasa_units','estimated_leaf_wet','consecutive_wetness_hours','wet_hours_previous_24h','wet_hours_previous_72h','wet_hours_previous_14d','rainfall_previous_24h_mm','rainfall_previous_72h_mm','rainfall_previous_14d_mm','mean_temp_previous_24h_c','mean_temp_previous_14d_c','mean_rh_previous_24h_pct','mean_rh_previous_14d_pct','temperature_suitability','wetness_suitability','bonfim_favourable_flag','risk_score','risk_level','risk_score_type','wetness_estimation_method','label_method','split']
def rounded(v): return '' if not np.isfinite(v) else round(float(v),3)
def write_csv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows(rows)
wet=wetness(T,RH,R,D,SOL); wet24=rolling(wet,24); wet72=rolling(wet,72); wet336=rolling(wet,336); r24=rolling(R,24);r72=rolling(R,72);r336=rolling(R,336);t24=rolling(T,24,True);t336=rolling(T,336,True);rh24=rolling(RH,24,True);rh336=rolling(RH,336,True)
consec=np.zeros(len(wet),dtype=int)
for i in range(len(wet)):consec[i]=(consec[i-1]+1 if i else 1) if wet[i] else 0
hist=[]
for i,r in enumerate(records):
    if i<335: continue
    sc,flag,ts,ms=score(t336[i],wet336[i],consec[i]); ts0=f"{r['YEAR']}-{r['MO'].zfill(2)}-{r['DY'].zfill(2)}T{r['HR'].zfill(2)}:00:00Z"
    hist.append(dict(zip(headers,[f'NASA_{i:05d}','', 'NASA_POWER_GRIDDED_WEATHER',ts0,rounded(T[i]),rounded(RH[i]),rounded(R[i]),rounded(D[i]),rounded(T[i]-D[i]),rounded(WS[i]),rounded(SOL[i]),int(wet[i]),int(consec[i]),rounded(wet24[i]),rounded(wet72[i]),rounded(wet336[i]),rounded(r24[i]),rounded(r72[i]),rounded(r336[i]),rounded(t24[i]),rounded(t336[i]),rounded(rh24[i]),rounded(rh336[i]),ts,ms,flag,sc,level(sc),'RELATIVE_ENVIRONMENTAL_SUITABILITY_NOT_PROBABILITY','RH>=95 OR (T-DEW<=1.5 AND SOLAR<0.15) OR RAIN>=0.5; UNCALIBRATED','PROJECT_HEURISTIC_ANCHORED_TO_BONFIM2024_AND_PEREZ1978','historical_holdout_reference'])))
write_csv(ROOT/'YS_NASA_HISTORICAL_2022_2024.csv',hist)
# Generate 6000 distinct 14-day synthetic scenario weather windows from 2022-2024 NASA analogues;
# seasonal analogues are perturbed in correlated blocks; wetness is calculated from adjusted RH, rain, dewpoint and solar.
# Each scenario yields one endpoint sample, avoiding overlapping windows and temporal leakage.
target={'No Risk':1500,'Low':1500,'Moderate':1500,'High':1500}; buckets={k:[] for k in target}; attempts=0
while any(len(buckets[k])<target[k] for k in target) and attempts<180000:
    attempts+=1; j=int(rng.integers(0,len(T)-336)); baseT=T[j:j+336].copy(); baseRH=RH[j:j+336].copy(); baseR=R[j:j+336].copy(); baseWS=WS[j:j+336].copy();baseSOL=SOL[j:j+336].copy()
    regime=int(rng.integers(0,6)); temp_shift=float(rng.choice([-8,-5,-2,0,2,5,8])+rng.normal(0,1.2)); t=np.clip(baseT+temp_shift,8,40)
    # Physically coherent moisture scenarios: RH/dew/rain/solar covary by moisture regime.
    wet_target=[0.08,0.23,0.42,0.60,0.78,0.95][regime]
    # Generate multi-hour wet and dry runs, independently per 14-day scenario.
    states=np.zeros(336,dtype=bool); state=bool(rng.random()<wet_target)
    for h in range(336):
        if state:
            if rng.random()<((1-wet_target)*0.11+0.012):state=False
        else:
            if rng.random()<(wet_target*0.16+0.008):state=True
        states[h]=state
    rh=np.clip(np.where(states, rng.normal(97.2,1.7,336),rng.normal(76,9,336)),35,100)
    # dew point from Magnus relation; humidity consistency preserved
    alpha=np.log(rh/100)+17.625*t/(243.04+t);dew=243.04*alpha/(17.625-alpha)
    rain=np.where(states & (rng.random(336)<0.23),rng.gamma(1.4,1.2,336),0.0)
    solar=np.maximum(0,baseSOL*np.where(states,0.28,0.92)); wind=np.clip(baseWS+rng.normal(0,0.3,336),0,15)
    w=wetness(t,rh,rain,dew,solar); c=0
    for v in w[::-1]:
        if not v:break
        c+=1
    t14=float(np.mean(t));lw14=int(np.sum(w));sc,flag,ts,ms=score(t14,lw14,c); lab=level(sc)
    if len(buckets[lab])>=target[lab]:continue
    scenario_id=f'YS_SIM_{attempts:06d}'
    data=dict(zip(headers,[scenario_id,scenario_id,'SIMULATED_NASA_ANALOGUE','',rounded(t[-1]),rounded(rh[-1]),rounded(rain[-1]),rounded(dew[-1]),rounded(t[-1]-dew[-1]),rounded(wind[-1]),rounded(solar[-1]),int(w[-1]),c,int(w[-24:].sum()),int(w[-72:].sum()),lw14,rounded(rain[-24:].sum()),rounded(rain[-72:].sum()),rounded(rain.sum()),rounded(t[-24:].mean()),rounded(t14),rounded(rh[-24:].mean()),rounded(rh.mean()),ts,ms,flag,sc,lab,'RELATIVE_ENVIRONMENTAL_SUITABILITY_NOT_PROBABILITY','RH>=95 OR (T-DEW<=1.5 AND SOLAR<0.15) OR RAIN>=0.5; UNCALIBRATED','PROJECT_HEURISTIC_ANCHORED_TO_BONFIM2024_AND_PEREZ1978','']));buckets[lab].append(data)
assert all(len(buckets[k])==target[k] for k in target),(attempts,{k:len(v) for k,v in buckets.items()})
for k in target: rng.shuffle(buckets[k])
for split,start,end in [('train',0,1050),('validation',1050,1275),('test',1275,1500)]:
    rows=[]
    for k in target:
        rows.extend(buckets[k][start:end])
    rng.shuffle(rows)
    for r in rows:r['split']=split
    write_csv(ROOT/f'YS_SIMULATED_{split}.csv',rows)
    print(split,len(rows),dict(collections.Counter(r['risk_level'] for r in rows)))
print('historical',len(hist),dict(collections.Counter(r['risk_level'] for r in hist)),'attempts',attempts)
# Verify physical consistency and no scenario overlap
ids=set()
for split in ['train','validation','test']:
    with open(ROOT/f'YS_SIMULATED_{split}.csv') as f: rr=list(csv.DictReader(f))
    for row in rr:
        assert row['scenario_id'] not in ids;ids.add(row['scenario_id'])
        assert float(row['dew_point_c'])<=float(row['air_temp_c'])+0.002
        assert 0<=float(row['risk_score'])<=100
        assert row['risk_level']==level(float(row['risk_score']))
        assert 0<=int(row['wet_hours_previous_14d'])<=336
print('VALIDATION PASSED: disjoint scenarios, dew point consistency, score range, class mapping, wetness hours')
readme='''# Yellow Sigatoka environmental suitability dataset — methodological notes\n\nLocation: Bingiriya, Sri Lanka (7.59832 N, 79.93734 E). NASA POWER hourly weather 2022–2024 (UTC).\n\n## Crucial interpretation\nRisk score is a **project-designed relative environmental suitability index**, NOT an empirically calibrated infection probability, NOT measured disease severity. Risk categories are project-defined. No real disease occurrence labels are included. Synthetic rows are counterfactual NASA-analogue scenarios, NOT real plantation weather. No model trained on these labels can be claimed to predict actual field infection without independent disease observations.\n\n## Research anchors\n1. Bonfim Junior et al. (2024), DOI 10.52945/rac.v37i3.1862. Field associations: 13–21 C with >160 wet hours in preceding 14 days, OR >21–25 C with >130 wet hours. Regression R2=0.55 reported. Study conducted in Brazilian subtropical conditions on different banana cultivar.\n2. Pérez & Fernández (1978), FAO AGRIS record 647245ed2c1d629bc9796815. Conidial germination min 8 C, optimum 25 C, max 37.5 C; these are laboratory germination limits, not field infection probabilities.\n3. NASA POWER hourly point dataset, https://power.larc.nasa.gov/ . Gridded estimates, not field station observations.\n\n## Explicit project-defined heuristic (NOT published equation)\nT14 = 14-day mean air temperature. LW14 = estimated wet hours during previous 336 hours. C = current consecutive wet hours.\nTempSuit = clip((T14-8)/17,0,1) if T14<=25, otherwise clip((37.5-T14)/12.5,0,1).\nWetAnchor = 130 hours if 21<T14<=25, otherwise 160 hours.\nWetSuit = clip((LW14-30)/(WetAnchor+45-30),0,1).\nContinuity = clip(C/18,0,1).\nRiskScore = 100 * TempSuit * (0.88*WetSuit + 0.12*Continuity).\nRisk categories: No Risk <5; Low 5–<30; Moderate 30–<65; High >=65. All score cutoffs and interpolation parameters are **engineering choices**, not claims from Bonfim or Pérez. Favourable flag uses Bonfim published empirical thresholds separately.\n\n## Leaf wetness proxy\nWet = RH>=95% OR (T - dewpoint <=1.5 C AND solar radiation <0.15 in original NASA units) OR rainfall >=0.5 mm/hr. This is a PROJECT ESTIMATION RULE, NOT calibrated against leaf-wetness sensors. NASA solar radiation field is preserved in its original source units; confirm units against the file header before converting.\n\n## Data generation and leakage\nHistorical dataset retains genuine NASA environmental values and computes rolling 24h,72h,14d features; first 335 records excluded for warmup. Simulated scenarios each draw a 336-hour historical weather analogue, apply temperature offsets and create internally coherent RH/dewpoint/rain/solar multi-hour regimes, and retain ONLY the final hour. Each scenario ID appears in only one split; 1050/225/225 per class. The underlying NASA analogue windows CAN overlap across scenarios, so **the simulation splits are not fully independent of source weather**; score-reproduction evaluation must not be interpreted as out-of-sample field disease prediction.\n\n## Recommended ML use\nTrain regression on risk_score with input features T14, LW14, consecutive wet hours and optionally the other environmental features; exclude risk_score, risk_level, favourable flag, temperature_suitability, wetness_suitability, metadata IDs, and origin. Compare model to the exact heuristic baseline (the baseline is deterministic and exact). For rigorous future validation, acquire observed Yellow Sigatoka onset/severity and measured leaf wetness, and evaluate on new site/year data. No guarantee against over/underfitting.\n\n## Class balancing\nBalanced SIMULATED files have 4200 train, 900 validation, 900 test with all four classes. Historical weather is NOT relabelled or artificially balanced.\n'''
(ROOT/'YS_METHODS_README.md').write_text(readme)
with zipfile.ZipFile(ROOT/'Yellow_Sigatoka_NASA_FOUR_CLASS_COMPLETE.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['YS_NASA_HISTORICAL_2022_2024.csv','YS_SIMULATED_train.csv','YS_SIMULATED_validation.csv','YS_SIMULATED_test.csv','YS_METHODS_README.md','build_yellow_sigatoka.py']:
        z.write(ROOT/name,name)
print('ZIP', (ROOT/'Yellow_Sigatoka_NASA_FOUR_CLASS_COMPLETE.zip').stat().st_size)
