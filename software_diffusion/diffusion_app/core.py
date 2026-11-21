from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
INPUTS = ['φ', 'PLD', 'LCD', 'Density', 'VSA', 'Dia', 'Pol', 'Dip', 'M', 'Longest 1D persistence']
FEATURES = INPUTS + ['LCD/PLD']
ALIASES = {'ρ':'Density', 'density':'Density', 'porosity':'φ', 'HVF':'φ', 'Longest 1D persisitence':'Longest 1D persistence'}

def read_table(path):
    path = Path(path)
    return pd.read_csv(path) if path.suffix.lower() == '.csv' else pd.read_excel(path)

def features(frame):
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.rename(columns={k:v for k,v in ALIASES.items() if k in frame and v not in frame})
    if frame.columns.duplicated().any():
        raise ValueError('输入含重复列名。')
    missing = [c for c in INPUTS if c not in frame]
    if missing:
        raise ValueError('缺少输入列：' + ', '.join(missing))
    x = frame[INPUTS].apply(pd.to_numeric, errors='coerce')
    invalid = ~np.isfinite(x).all(axis=1)
    invalid |= (x['φ'] < 0) | (x['φ'] > 1)
    invalid |= (x[['PLD','LCD','Density','Dia','M']] <= 0).any(axis=1)
    invalid |= (x[['VSA','Pol','Dip','Longest 1D persistence']] < 0).any(axis=1)
    invalid |= x.LCD < x.PLD
    if invalid.any():
        rows = (np.flatnonzero(invalid) + 2).tolist()
        raise ValueError(f'输入缺失、非有限数或超出物理范围，文件行号：{rows[:20]}。请修正后预测。')
    x['LCD/PLD'] = x.LCD / x.PLD
    return x[FEATURES]

def load_bundle(path=None):
    path = Path(path or ROOT / 'models' / 'stacking.joblib')
    if not path.exists():
        raise FileNotFoundError('未找到训练模型：' + str(path))
    return joblib.load(path)

def predict(frame, bundle):
    x = features(frame)
    logd = bundle['model'].predict(x)
    result = frame.copy()
    result['predicted_log10_D'] = logd
    result['predicted_D_cm2_s'] = np.power(10.0, logd)
    warnings = []
    for _, row in x.iterrows():
        outside = [c for c in FEATURES if row[c] < bundle['ranges'][c][0] or row[c] > bundle['ranges'][c][1]]
        warnings.append(', '.join(outside))
    result['outside_training_range'] = warnings
    return result
