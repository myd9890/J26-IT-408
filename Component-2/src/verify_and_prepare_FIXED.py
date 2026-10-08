"""Non-destructive Black Sigatoka safeguards. Run from ML training project root."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path.cwd()
SOURCE = ROOT / 'datasets' / 'black_sigatoka' / 'revised_splits'
OUT = ROOT / 'reports' / 'black_sigatoka' / 'final_safeguards'
OUT.mkdir(parents=True, exist_ok=True)
FEATURES = ['air_temp_c', 'estimated_leaf_wet', 'consecutive_wetness_hours']
TARGET = 'risk_score'
FILES = {name: SOURCE / f'BS_SOURCE_TIME_{name}.csv' for name in ('train', 'validation', 'test')}
frames = {}
for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f'Missing revised {name} dataset: {path}')
    df = pd.read_csv(path)
    required = FEATURES + [TARGET, 'risk_level', 'scenario_id', 'weather_analogue_timestamp_utc']
    missing = sorted(set(required) - set(df))
    if missing:
        raise ValueError(f'{name} lacks required columns: {missing}')
    if df[FEATURES + [TARGET]].isna().any().any() or not np.isfinite(df[FEATURES + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError(f'{name} has missing or nonfinite model values')
    if not df[TARGET].between(0,100).all():
        raise ValueError(f'{name} has out-of-range scores')
    if not df['estimated_leaf_wet'].isin([0,1]).all() or (df['consecutive_wetness_hours'] < 0).any():
        raise ValueError(f'{name} has invalid wetness inputs')
    expected = np.select([df[TARGET].eq(0),df[TARGET].lt(25),df[TARGET].lt(75)], ['No Risk','Low','Moderate'], default='High')
    if not (df['risk_level'].to_numpy()==expected).all():
        raise ValueError(f'{name} has inconsistent risk categories')
    frames[name] = df

# Repeated feature vectors with conflicting scores: audit, don't silently overwrite labels.
combined = pd.concat([df.assign(_dataset=k) for k,df in frames.items()], ignore_index=True)
groups = combined.groupby(FEATURES,dropna=False)[TARGET].agg(['count','min','max']).reset_index()
conflicts = groups[(groups['count']>1) & (groups['max']>groups['min'])].copy()
conflicts['score_spread'] = conflicts['max']-conflicts['min']
conflicts.to_csv(OUT/'same_features_different_scores.csv', index=False)

# Source-time checks using pandas time deltas (avoids integer/nanosecond assumptions).
# Always save diagnostic results before reporting any failure.
times = {}
for name, df in frames.items():
    parsed = pd.to_datetime(df['weather_analogue_timestamp_utc'], utc=True, errors='coerce')
    if parsed.isna().any():
        raise ValueError(f'{name}: {parsed.isna().sum()} missing/invalid donor timestamps')
    times[name] = pd.DatetimeIndex(parsed.drop_duplicates().sort_values())

separation = []
for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
    a, b = times[left], times[right]
    shared = len(a.intersection(b))
    if len(a) == 0 or len(b) == 0:
        raise ValueError(f'Empty donor-timestamp set: {left}, {right}')
    # Nearest known donor in b for each timestamp in a, in timestamp units.
    nearest_pos = b.get_indexer(a, method='nearest')
    nearest_b = b.take(nearest_pos)
    differences = abs(a - nearest_b)
    min_gap_h = float(differences.min() / pd.Timedelta(hours=1))
    separation.append({
        'pair': f'{left}-{right}',
        'left_donor_count': len(a),
        'right_donor_count': len(b),
        'shared_donor_timestamps': shared,
        'minimum_gap_hours': round(min_gap_h, 4),
        'passes_120_hour_gap': bool(min_gap_h > 120 and shared == 0)
    })
sep = pd.DataFrame(separation)
sep.to_csv(OUT / 'source_time_separation.csv', index=False)
print('Source-time gap diagnostic:')
print(sep.to_string(index=False))
if not sep['passes_120_hour_gap'].all():
    failed_pairs = sep.loc[~sep['passes_120_hour_gap'], ['pair', 'minimum_gap_hours', 'shared_donor_timestamps']]
    raise ValueError('Source-time separation failed for these pairs:\n' + failed_pairs.to_string(index=False) +
                     '\nReview input files and the saved source_time_separation.csv; do not change the CSVs blindly.')

# Operational contract: strictly use only three inputs and train-only transformations.
contract = {
    'purpose':'PRELIMINARY SYNTHETIC RULE-IMITATION REGRESSION, NOT FIELD DISEASE PREDICTION',
    'feature_order':FEATURES, 'target':TARGET,
    'risk_category_bounds':{'No Risk':'score == 0','Low':'0 < score < 25','Moderate':'25 <= score < 75','High':'75 <= score <= 100'},
    'allowed_training_files':{k:str(v.relative_to(ROOT)) for k,v in FILES.items()},
    'must_exclude':['risk_level','scenario_id','split','weather_analogue_timestamp_utc','record_type','temperature_input_type','wetness_estimation_method','label_source'],
    'do_not_do':['modify original or revised csv','drop contradictory scores without provenance','recompute labels from unavailable formula','tune on test','claim independently verified infection accuracy']
}
(OUT/'model_input_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')

report = f'''# Black Sigatoka remaining-issues safeguards\n\n## Confirmed outcomes\n- Revised source-time splits verified. {len(combined):,} retained rows.\n- Donor-time separation: {sep.to_dict(orient='records')}.\n- Same baseline inputs with differing target scores: {len(conflicts)} distinct groups. Maximum spread: {conflicts['score_spread'].max() if len(conflicts) else 0:.4f}. **Preserved unchanged.**\n- Feature contract restricts predictors to {FEATURES}, target `{TARGET}`; no direct target metadata.\n\n## Issues that cannot be honestly 'fixed' from these data\n1. Slight score disagreement may be due to rounded inputs or unavailable upstream generator; rewriting scores invents evidence. Keep records and report uncertainty.\n2. Separation greater than 120 hours protects known windows but cannot establish all weather-series independence; test more conservative source-time blocking if needed.\n3. Simulated labels cannot be converted into confirmed disease observations without new field data; future validation needs measured leaf wetness and infection outcomes.\n\n## Next safe action\nUse only revised splits and three-feature contract for preliminary regression. Compare validation MAE/RMSE across models, and report held-out test metrics once after model selection. Keep field-validity claims separate.\n'''
(OUT/'Remaining_Issues_Report.md').write_text(report,encoding='utf-8')
print(report)
print('Saved safeguards to:',OUT)
