import csv,random,math,collections,zipfile,json
from pathlib import Path
p=Path('/mnt/data');rng=random.Random(20260929)
source=p/'Black_Sigatoka_Simulated_Training_Only.csv'; weather=p/'Black_Sigatoka_Research_Complete_Hourly_2022_2024.csv'
real=list(csv.DictReader(weather.open(newline='')))
valid=[r for r in real if r['air_temp_c'] and r['relative_humidity_pct'] and r['wind_speed_m_s'] and r['solar_radiation_mj_hr'] and r['split'] not in ('excluded_warmup','')]
by=collections.defaultdict(list)
for r in valid:by[int(float(r['air_temp_c'])//2)*2].append(r)
allreal=valid
sim=list(csv.DictReader(source.open(newline='')))
classes=['No Risk','Low','Moderate','High']
for c in classes: assert sum(r['risk_level']==c for r in sim)==1500
# Weather-analogue donor selection for wind and broad meteorological plausibility; synthetic RH and dew point explicitly constructed.
def donor(temp):
 k=int(temp//2)*2; pool=by.get(k) or allreal
 return rng.choice(pool)
def dewpoint(t,rh):
 a,b=17.625,243.04;g=math.log(rh/100)+a*t/(b+t);return b*g/(a-g)
cols=['scenario_id','record_type','timestamp_utc','air_temp_c','relative_humidity_pct','rainfall_mm_hr','dew_point_c','dew_point_depression_c','wind_speed_m_s','solar_radiation_mj_hr','estimated_leaf_wet','consecutive_wetness_hours','wet_hours_previous_24h','wet_hours_previous_72h','rain_previous_24h_mm','mean_temp_previous_24h_c','mean_rh_previous_24h_pct','canopy_temperature_c','temperature_input_type','wetness_estimation_method','risk_score','risk_level','label_source','split','weather_analogue_timestamp_utc']
rows=[]
for r in sim:
 t=float(r['air_temp_c']);wet=int(r['estimated_wet_primary']);dur=int(float(r['wetness_duration_hours'])); score=float(r['infection_response_score']); d=donor(t)
 # Conditions are scenario assumptions, NOT recorded hourly time series. Wet hours may result from humidity/dew without rain.
 rh=round(rng.uniform(98.0,99.7),2) if wet else round(rng.uniform(57,88),2)
 rain=0.0 if rng.random()<.85 else (round(rng.uniform(1,6),2) if wet else 0.0)
 if not wet: rain=0.0
 wind=round(float(d['wind_speed_m_s']),2)
 solar=round(min(float(d['solar_radiation_mj_hr']),rng.uniform(0,0.3)),3) if wet else round(float(d['solar_radiation_mj_hr']),3)
 dp=round(dewpoint(t,rh),2)
 wet24=min(dur,24) if wet else 0;wet72=min(dur,72) if wet else 0
 # Rolling means are stylized constant-state assumptions, not actual reconstructed historical sequences.
 row={'scenario_id':r['scenario_id'],'record_type':'SYNTHETIC_SCENARIO_NOT_MEASURED','timestamp_utc':'','air_temp_c':round(t,3),'relative_humidity_pct':rh,'rainfall_mm_hr':rain,'dew_point_c':dp,'dew_point_depression_c':round(t-dp,2),'wind_speed_m_s':wind,'solar_radiation_mj_hr':solar,'estimated_leaf_wet':wet,'consecutive_wetness_hours':dur,'wet_hours_previous_24h':wet24,'wet_hours_previous_72h':wet72,'rain_previous_24h_mm':'','mean_temp_previous_24h_c':'','mean_rh_previous_24h_pct':'','canopy_temperature_c':'','temperature_input_type':'AIR_TEMPERATURE_PROXY','wetness_estimation_method':'SCENARIO_SET_RH_GE_98_WET_ELSE_DRY; NOT SENSOR','risk_score':score,'risk_level':r['risk_level'],'label_source':'PROJECT_ADAPTED_BEBBER_WEIBULL_WET_SPELL; NOT DISEASE_INCIDENCE','split':'','weather_analogue_timestamp_utc':d['timestamp_utc']}
 # No fake 24h rain or averages: unavailable for independent point scenarios.
 rows.append(row)
# Fixed splits per class
splits={k:[] for k in ['train','validation','test']}
for c in classes:
 subset=[r for r in rows if r['risk_level']==c];rng.shuffle(subset)
 for name,group in [('train',subset[:1050]),('validation',subset[1050:1275]),('test',subset[1275:])]:
  for x in group:x['split']=name
  splits[name]+=group
for name,group in splits.items():
 rng.shuffle(group)
 with (p/f'BS_FULL_FEATURE_SIMULATED_{name}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(group)
# Historical full-feature source with consistent names, genuine values and noninvented blanks
realcols=cols
with (p/'BS_FULL_FEATURE_HISTORICAL_NASA.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=realcols);w.writeheader()
 for i,r in enumerate(real):
  out={'scenario_id':f'NASA_{i:05d}','record_type':'NASA_GRIDDED_WEATHER_WITH_ESTIMATED_WETNESS','timestamp_utc':r['timestamp_utc'],'air_temp_c':r['air_temp_c'],'relative_humidity_pct':r['relative_humidity_pct'],'rainfall_mm_hr':r['rainfall_mm_hr'],'dew_point_c':r['dew_point_c'],'dew_point_depression_c':r['dew_point_depression_c'],'wind_speed_m_s':r['wind_speed_m_s'],'solar_radiation_mj_hr':r['solar_radiation_mj_hr'],'estimated_leaf_wet':r['estimated_wet_primary'],'consecutive_wetness_hours':r['wetness_duration_hours'],'wet_hours_previous_24h':r['wet_hours_previous_24h'],'wet_hours_previous_72h':r['wet_hours_previous_72h'],'rain_previous_24h_mm':r['rain_previous_24h_mm'],'mean_temp_previous_24h_c':r['mean_temp_previous_24h_c'],'mean_rh_previous_24h_pct':r['mean_rh_previous_24h_pct'],'canopy_temperature_c':'','temperature_input_type':'NASA_2M_AIR_TEMPERATURE_PROXY','wetness_estimation_method':r['wetness_method'],'risk_score':r['infection_response_score'],'risk_level':r['risk_level'],'label_source':r['label_source'],'split':r['split'],'weather_analogue_timestamp_utc':''}
  w.writerow(out)
readme='''# Black Sigatoka FULL-COLUMN four-class dataset — READ FIRST\n\n## Which file should I train on?\n`BS_FULL_FEATURE_SIMULATED_train.csv` has 4,200 rows, 1,050 per class, with temperature, humidity, rain, dew point, wind, solar radiation, estimated wetness, consecutive wetness hours, risk score and risk level. Matching 900-row validation and 900-row test files contain 225 per class. THESE ARE SYNTHETIC MATHEMATICAL SCENARIOS, NOT REAL WEATHER TIME SERIES OR DISEASE EVENTS. The NASA weather file has 26,304 historical hourly records and all real available environmental measurements.\n\n## Honest caveat on the requested full columns\nTo give all columns without pretending to have measured data, each synthetic scenario uses its target temperature and wetness duration from the existing project-generated four-class scenarios; wet/dry RH is constructed consistently (wet >=98%, dry 57–88%), dew point is recomputed from T/RH via Magnus equation, rain is 0 for dry scenarios and 0 or >=1 mm/hr for wet ones, wind/solar use a temperature-matched NASA weather analogue (solar reduced for wet scenarios). Donor timestamp is disclosed. These are PLAUSIBLE constructed point scenarios, not a physically validated weather generator. The historical NASA file retains actual weather columns. Synthetic 24-hour mean temperature, RH and rainfall are intentionally blank because one isolated scenario cannot support an authentic preceding 24-hour history; 24/72h wet hours assume an uninterrupted wet spell. Canopy temperature is blank because it was not measured or reliably estimated.\n\n## Targets and feature selection\nTarget: `risk_score` (0–100 relative infection response); derive `risk_level` from score (0 No Risk, >0–<25 Low, >=25–<75 Moderate, >=75 High). For a robust BASELINE model train on `air_temp_c`, `estimated_leaf_wet`, `consecutive_wetness_hours`. Other weather columns are useful for a subsequent wetness-estimation submodel, but were partly constructed conditionally on wetness in simulated data; adding them to the risk regressor will cause synthetic shortcut learning, not independently validated improvement. Never include risk_level, risk_score, split, scenario_id, donor timestamp or source metadata as inputs. No guaranteed absence of over/underfitting.\n\n## Scientific model and provenance\nRisk labels copied unchanged from prior `Black_Sigatoka_Simulated_Training_Only.csv`, generated with a PROJECT ADAPTATION of Bebber (2019), DOI 10.1098/rstb.2018.0269: temperature-dependent Weibull wet-spell response (Tmin 16.6, Topt 27.2, Tmax 30.3 C; scale 32.6 h, shape 1.76). This is not the paper's complete annual hourly spore-cohort infection-pressure calculation, and the 0/25/75 class cutoffs are project-defined. Jacome & Schuh (1992), DOI 10.1094/Phyto-82-515 supports temperature/wetness dependence. NASA gridded estimates are not plantation sensors. Historical wetness is inferred by RH>=98% OR hourly rain>=1 mm/h (uncalibrated proxy). No actual disease-occurrence probabilities or confirmed disease outcomes are present.\n\n## Evaluation\nSynthetic splits have distinct scenario IDs, but all are generated from the same mathematical model. Report synthetic performance as equation imitation only. Evaluate separately on chronological NASA splits; its historical data contains 24,990 No Risk, 1,288 Low, 26 Moderate and 0 High records under the selected wetness proxy. A real-weather High Risk accuracy cannot be estimated. Obtain measured leaf wetness and independently confirmed disease outcomes for field validation.\n\n## Future outlook\nFuture 7–14-day environmental risk requires HOURLY WEATHER FORECASTS and the same wetness estimation and historical feature pipeline. The current NASA historical CSV is not a future forecast.\n'''
(p/'BS_FULL_FEATURE_README.md').write_text(readme)
files=[f'BS_FULL_FEATURE_SIMULATED_{x}.csv' for x in splits]+['BS_FULL_FEATURE_HISTORICAL_NASA.csv','BS_FULL_FEATURE_README.md','build_full_feature_four_class.py']
with zipfile.ZipFile(p/'Black_Sigatoka_FULL_ENVIRONMENT_FOUR_CLASS.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for x in files:z.write(p/x,x)
for name,group in splits.items():
 counts=dict(collections.Counter(x['risk_level'] for x in group));assert all(counts[c]==(1050 if name=='train' else 225) for c in classes)
 assert all((float(x['risk_score'])==0 if x['risk_level']=='No Risk' else (0<float(x['risk_score'])<25 if x['risk_level']=='Low' else (25<=float(x['risk_score'])<75 if x['risk_level']=='Moderate' else 75<=float(x['risk_score'])<=100))) for x in group)
 assert all(float(x['dew_point_c'])<=float(x['air_temp_c'])+.02 for x in group)
 print(name,len(group),counts,'verified score bins and dew point')
print('Historical rows:',len(real),'Zip bytes:',(p/'Black_Sigatoka_FULL_ENVIRONMENT_FOUR_CLASS.zip').stat().st_size)
