# ============================================================
# SmartGrid OS — Flask Backend
# SG02 · Dr. Boutheina BEN ISMAIL
# ============================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import io
import os
import time
import re
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)  # Allow HTML frontend to call this API

def load_env_file(path=".env"):
    env_path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

load_env_file()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
ALLOW_LOCAL_AI_FALLBACK = os.environ.get("AI_ALLOW_LOCAL_FALLBACK", "0").strip() == "1"
client = genai.Client(api_key=GEMINI_API_KEY) if genai and GEMINI_API_KEY else None
# ── Global state (in-memory per session) ─────────────────────
state = {
    'df': None,
    'vals': [],
    'energy_col': None,
    'time_col': None,
    'temporal_source': None,
    'hour_buckets': None,
    'hour_cnt': None,
    'day_buckets': None,
    'day_cnt': None,
    'month_buckets': None,
    'month_cnt': None,
    'stream_index': 0,
    'stream_active': False,
    'stream_started_at': None,
}

# ── Load ML model ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'energy_model.pkl')

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

try:
    artifacts = load_model()
    ml_model  = artifacts['model']
    ml_scaler = artifacts['scaler']
    ml_feats  = artifacts['features']
    ml_mae    = artifacts['mae']
    ml_rmse   = artifacts['rmse']
    ml_r2     = artifacts['r2']
    print(f"[OK] Model loaded | Features: {ml_feats}")
    print(f"   MAE={ml_mae:.4f}  RMSE={ml_rmse:.4f}  R2={ml_r2:.4f}")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    artifacts = ml_model = ml_scaler = ml_feats = None
    ml_mae = ml_rmse = ml_r2 = 0

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def avg(lst):
    return float(np.mean(lst)) if len(lst) else 0.0

def percentile(lst, p):
    return float(np.percentile(lst, p)) if len(lst) else 0.0

def parse_csv_bytes(raw_bytes):
    sample = raw_bytes[:3000].decode('utf-8', errors='ignore')
    sep = ';' if sample.count(';') > sample.count(',') else ','
    df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, low_memory=False)
    df = df.iloc[:50_000]
    return df, sep

def detect_columns(df):
    energy_col = None
    time_col   = None
    for c in df.columns:
        cl = c.lower()
        if not energy_col and any(k in cl for k in ['power','consumption','kwh','energy','load','kw','mw','watt','demand']):
            energy_col = c
        if not time_col and any(k in cl for k in ['time','date','stamp','ts','datetime']):
            time_col = c
    return energy_col, time_col

def extract_temporal(df, time_col):
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        df['hour']        = df[time_col].dt.hour
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['month']       = df[time_col].dt.month
        df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
        df['season']      = df['month'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
    else:
        n = len(df)
        df['hour']        = np.tile(range(24), n // 24 + 1)[:n]
        df['day_of_week'] = df.index % 7
        df['month']       = (df.index // 720) % 12 + 1
        df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
        df['season']      = df['month'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
    return df

def build_buckets(df, energy_col):
    hB = np.zeros(24); hC = np.zeros(24)
    dB = np.zeros(7);  dC = np.zeros(7)
    mB = np.zeros(12); mC = np.zeros(12)
    for _, row in df.iterrows():
        v = row[energy_col]
        if pd.isna(v): continue
        h = int(row.get('hour', -1))
        d = int(row.get('day_of_week', -1))
        m = int(row.get('month', 0)) - 1
        if 0 <= h < 24: hB[h] += v; hC[h] += 1
        if 0 <= d < 7:  dB[d] += v; dC[d] += 1
        if 0 <= m < 12: mB[m] += v; mC[m] += 1
    return hB, hC, dB, dC, mB, mC

def build_day_hour_heatmap(df, energy_col):
    heat = np.zeros((7, 24))
    counts = np.zeros((7, 24))
    for _, row in df.iterrows():
        v = row[energy_col]
        if pd.isna(v):
            continue
        d = int(row.get('day_of_week', -1))
        h = int(row.get('hour', -1))
        if 0 <= d < 7 and 0 <= h < 24:
            heat[d][h] += v
            counts[d][h] += 1
    avg_heat = np.divide(heat, counts, out=np.zeros_like(heat), where=counts > 0)
    return avg_heat.tolist(), counts.astype(int).tolist()

# ─────────────────────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

# ─────────────────────────────────────────────────────────────
# ROUTE: /upload
# ─────────────────────────────────────────────────────────────
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file sent'}), 400
    raw = request.files['file'].read()
    try:
        df, sep = parse_csv_bytes(raw)
    except Exception as e:
        return jsonify({'error': f'CSV parse error: {e}'}), 400

    energy_col, time_col = detect_columns(df)
    # Allow client to override
    if request.form.get('energy_col'):
        energy_col = request.form['energy_col']
    if request.form.get('time_col'):
        time_col = request.form['time_col']

    if not energy_col:
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        energy_col = nums[0] if nums else None
    if not energy_col:
        return jsonify({'error': 'Could not detect energy column'}), 400

    temporal_source = 'timestamp' if time_col and time_col in df.columns else 'generated'
    df = extract_temporal(df, time_col)
    df[energy_col] = pd.to_numeric(df[energy_col], errors='coerce')
    df = df.dropna(subset=[energy_col]).drop_duplicates().reset_index(drop=True)

    # Remove outliers (1st–99th percentile)
    q1  = df[energy_col].quantile(0.01)
    q99 = df[energy_col].quantile(0.99)
    df  = df[(df[energy_col] >= q1) & (df[energy_col] <= q99)].reset_index(drop=True)

    hB, hC, dB, dC, mB, mC = build_buckets(df, energy_col)
    heatmap, heatmap_counts = build_day_hour_heatmap(df, energy_col)

    # Store in global state
    state['df']           = df
    state['vals']         = df[energy_col].tolist()
    state['energy_col']   = energy_col
    state['time_col']     = time_col
    state['temporal_source'] = temporal_source
    state['hour_buckets'] = hB.tolist()
    state['hour_cnt']     = hC.tolist()
    state['day_buckets']  = dB.tolist()
    state['day_cnt']      = dC.tolist()
    state['month_buckets']= mB.tolist()
    state['month_cnt']    = mC.tolist()
    state['stream_index'] = 0
    state['stream_active'] = False
    state['stream_started_at'] = None

    vals = state['vals']
    mean = float(np.mean(vals))
    std  = float(np.std(vals))
    peak = float(np.max(vals))
    total= float(np.sum(vals))
    score= max(0, min(100, round(100 - (std / mean) * 50))) if mean else 0

    # Timeline sample (max 300 points)
    step = max(1, len(vals) // 300)
    timeline = vals[::step]

    # Hourly avg
    hAvg = [float(hB[i] / hC[i]) if hC[i] > 0 else 0.0 for i in range(24)]
    dAvg = [float(dB[i] / dC[i]) if dC[i] > 0 else 0.0 for i in range(7)]
    mAvg = [float(mB[i] / mC[i]) if mC[i] > 0 else 0.0 for i in range(12)]

    lo = int(sum(1 for v in vals if v < mean * 0.75))
    md = int(sum(1 for v in vals if mean * 0.75 <= v <= mean * 1.25))
    hi = int(sum(1 for v in vals if v > mean * 1.25))

    # Preview rows
    preview_cols = df.columns.tolist()[:6]
    preview_rows = df[preview_cols].head(6).fillna('—').to_dict('records')

    # Rolling avg (24h)
    series = df[energy_col].values
    roll24 = pd.Series(series).rolling(24, min_periods=1).mean().tolist()
    roll_step = max(1, len(roll24) // 300)
    raw_sample  = series[::roll_step].tolist()
    roll_sample = roll24[::roll_step]

    return jsonify({
        'ok': True,
        'rows': len(vals),
        'cols': df.shape[1],
        'sep': sep,
        'energy_col': energy_col,
        'time_col': time_col,
        'temporal_source': temporal_source,
        'headers': df.columns.tolist(),
        'kpi': {
            'total': round(total / 1000, 2),
            'mean':  round(mean, 2),
            'peak':  round(peak, 2),
            'score': score,
        },
        'timeline': timeline,
        'donut': {'low': lo, 'normal': md, 'high': hi},
        'hourly_avg': hAvg,
        'daily_avg':  dAvg,
        'monthly_avg':mAvg,
        'heatmap': heatmap,
        'heatmap_counts': heatmap_counts,
        'preview_cols': preview_cols,
        'preview_rows': preview_rows,
        'rolling': {'raw': raw_sample, 'roll24': roll_sample},
        'trend': 'Rising' if float(np.mean(vals[-100:])) > float(np.mean(vals[:100])) else 'Falling',
        'std': round(std, 3),
    })

# ─────────────────────────────────────────────────────────────
# ROUTE: /stats  (anomaly detection)
# ─────────────────────────────────────────────────────────────
@app.route('/stats', methods=['GET'])
def stats():
    vals = state['vals']
    if not vals:
        return jsonify({'error': 'No data loaded'}), 400

    mean = float(np.mean(vals))
    std  = float(np.std(vals))
    threshold = 2.5

    step   = max(1, len(vals) // 300)
    sample = vals[::step]
    zscores = [(v - mean) / std if std else 0 for v in sample]
    anomalies = [sample[i] if abs(z) > threshold else None for i, z in enumerate(zscores)]
    upper = [mean + threshold * std] * len(sample)
    lower = [mean - threshold * std] * len(sample)

    anom_count = sum(1 for a in anomalies if a is not None)
    crit_count = sum(1 for z in zscores if abs(z) > 3.5)
    rate = round(anom_count / len(sample) * 100, 2) if sample else 0

    alerts = []
    for i, (z, v) in enumerate(zip(zscores, sample)):
        if abs(z) > threshold:
            sev = 'crit' if abs(z) > 3.5 else 'warn' if abs(z) > 3 else 'ok'
            alerts.append({'index': i, 'value': round(v, 2), 'zscore': round(z, 2), 'severity': sev})

    return jsonify({
        'ok': True,
        'mean': round(mean, 3),
        'std':  round(std, 3),
        'threshold': threshold,
        'anomaly_count': anom_count,
        'crit_count': crit_count,
        'rate': rate,
        'sample': sample,
        'anomalies': anomalies,
        'upper': upper,
        'lower': lower,
        'alerts': alerts[:15],
    })

# ─────────────────────────────────────────────────────────────
@app.route('/stream/start', methods=['POST'])
def stream_start():
    if state['df'] is None or not state['vals']:
        return jsonify({'error': 'No data loaded'}), 400
    state['stream_index'] = 0
    state['stream_active'] = True
    state['stream_started_at'] = time.time()
    return jsonify({
        'ok': True,
        'rows': len(state['vals']),
        'energy_col': state['energy_col'],
        'time_col': state['time_col'],
        'temporal_source': state['temporal_source'],
    })

@app.route('/stream/stop', methods=['POST'])
def stream_stop():
    state['stream_active'] = False
    return jsonify({'ok': True, 'index': state['stream_index']})

@app.route('/stream/next', methods=['GET'])
def stream_next():
    if state['df'] is None or not state['vals']:
        return jsonify({'error': 'No data loaded'}), 400
    if not state['stream_active']:
        return jsonify({'error': 'Stream is not active'}), 400

    batch_size = request.args.get('batch_size', default=50, type=int)
    batch_size = max(1, min(batch_size, 1000))
    start = int(state['stream_index'])
    end = min(start + batch_size, len(state['vals']))
    df = state['df']
    energy_col = state['energy_col']
    time_col = state['time_col']

    rows = []
    for idx in range(start, end):
        row = df.iloc[idx]
        stamp = row[time_col] if time_col and time_col in df.columns else idx
        rows.append({
            'index': idx,
            'timestamp': str(stamp),
            'value': float(row[energy_col]),
            'hour': int(row.get('hour', 0)),
            'day_of_week': int(row.get('day_of_week', 0)),
        })

    state['stream_index'] = end
    if end >= len(state['vals']):
        state['stream_active'] = False

    seen = state['vals'][:end]
    latest = rows[-1]['value'] if rows else None
    return jsonify({
        'ok': True,
        'active': state['stream_active'],
        'index': end,
        'total_rows': len(state['vals']),
        'progress': round(end / len(state['vals']) * 100, 2),
        'batch': rows,
        'latest': latest,
        'running': {
            'count': len(seen),
            'mean': round(float(np.mean(seen)), 2) if seen else 0,
            'peak': round(float(np.max(seen)), 2) if seen else 0,
            'total_mwh': round(float(np.sum(seen)) / 1000, 2) if seen else 0,
        }
    })

# ROUTE: /predict  (real ML inference)
# ─────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if not ml_model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    vals = state['vals']

    # Build feature row matching model's expected features
    row = {}
    for feat in ml_feats:
        row[feat] = float(data.get(feat, 0))

    input_df = pd.DataFrame([row])[ml_feats]
    scaled   = ml_scaler.transform(input_df)
    pred     = float(ml_model.predict(scaled)[0])

    mean = float(np.mean(vals)) if vals else pred
    level = 'high' if pred > mean * 1.25 else 'low' if pred < mean * 0.75 else 'med'
    pct   = min(100, round((pred / (mean * 2)) * 100, 1)) if mean else 50

    # 24h forecast using model
    h = int(data.get('hour', 12))
    d = int(data.get('day_of_week', 1))
    forecast = []
    hB = state['hour_buckets'] or [mean] * 24
    hC = state['hour_cnt']     or [1] * 24
    for fh in range(24):
        frow = {feat: float(data.get(feat, 0)) for feat in ml_feats}
        frow['hour'] = fh
        fdf   = pd.DataFrame([frow])[ml_feats]
        fsc   = ml_scaler.transform(fdf)
        fpred = float(ml_model.predict(fsc)[0])
        noise = float(np.random.normal(0, mean * 0.02)) if mean else 0
        forecast.append(round(fpred + noise, 2))

    tips = []
    if h >= 12 and h <= 16:
        tips.append('Peak window active — shift non-critical loads to off-peak hours (22h–06h)')
    if float(data.get('temp', 20)) > 30:
        tips.append('High temperature — HVAC load will increase, check cooling efficiency')
    if int(data.get('is_weekend', 0)):
        tips.append('Weekend mode — typically 15–30% lower than weekday baseline')
    if level == 'high':
        tips.append('High load predicted — consider activating demand-response protocols')
    if not tips:
        tips.append('Normal consumption expected — no immediate action required')

    return jsonify({
        'ok': True,
        'prediction': round(pred, 2),
        'level': level,
        'pct': pct,
        'mean': round(mean, 2),
        'forecast': forecast,
        'tips': tips,
        'model_info': {
            'mae': round(ml_mae, 4),
            'rmse': round(ml_rmse, 4),
            'r2': round(ml_r2, 4),
            'features': ml_feats,
        }
    })

# ─────────────────────────────────────────────────────────────
# ROUTE: /forecast  (24h forecast only, no input needed)
# ─────────────────────────────────────────────────────────────
@app.route('/forecast', methods=['GET'])
def forecast():
    if not ml_model:
        return jsonify({'error': 'Model not loaded'}), 500
    vals = state['vals']
    mean = float(np.mean(vals)) if vals else 100.0
    results = []
    for fh in range(24):
        row = {feat: 0 for feat in ml_feats}
        row['hour'] = fh
        row['day_of_week'] = 1
        row['month'] = 6
        row['is_weekend'] = 0
        row['season'] = 2
        fdf   = pd.DataFrame([row])[ml_feats]
        fsc   = ml_scaler.transform(fdf)
        fpred = float(ml_model.predict(fsc)[0])
        results.append(round(fpred, 2))
    return jsonify({'ok': True, 'forecast': results, 'baseline': round(mean, 2)})

# ─────────────────────────────────────────────────────────────
# ROUTE: /anomalies  (alias for /stats)
# ─────────────────────────────────────────────────────────────
@app.route('/anomalies', methods=['GET'])
def anomalies():
    return stats()

# ─────────────────────────────────────────────────────────────
# ROUTE: /model_info
# ─────────────────────────────────────────────────────────────
@app.route('/model_info', methods=['GET'])
def model_info():
    if not ml_model:
        return jsonify({'error': 'Model not loaded'}), 500
    return jsonify({
        'ok': True,
        'features': ml_feats,
        'mae': round(ml_mae, 4),
        'rmse': round(ml_rmse, 4),
        'r2': round(ml_r2, 4),
        'model_type': type(ml_model).__name__,
    })

def dataset_context_text():
    if not state['vals']:
        return "No dataset has been loaded yet."

    vals = state['vals']
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    peak = float(np.max(vals))
    minimum = float(np.min(vals))
    median = float(np.median(vals))
    p25 = percentile(vals, 25)
    p75 = percentile(vals, 75)
    total = float(np.sum(vals))
    upper = mean + 2.5 * std
    lower = mean - 2.5 * std
    anomalies = sum(1 for v in vals if v > upper or v < lower)

    hourly_summary = "not available"
    if state['hour_buckets'] and state['hour_cnt']:
        hourly = [
            state['hour_buckets'][i] / state['hour_cnt'][i]
            if state['hour_cnt'][i] else 0
            for i in range(24)
        ]
        peak_hour = int(np.argmax(hourly))
        low_hour = int(np.argmin(hourly))
        hourly_summary = (
            f"highest average hour {peak_hour}:00 ({hourly[peak_hour]:.2f} kWh), "
            f"lowest average hour {low_hour}:00 ({hourly[low_hour]:.2f} kWh)"
        )

    daily_summary = "not available"
    if state['day_buckets'] and state['day_cnt']:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily = [
            state['day_buckets'][i] / state['day_cnt'][i]
            if state['day_cnt'][i] else 0
            for i in range(7)
        ]
        daily_summary = ", ".join(f"{days[i]} {daily[i]:.2f}" for i in range(7))

    return f"""
Dataset loaded:
- Records: {len(vals):,}
- Columns: {state['df'].shape[1] if state['df'] is not None else 'unknown'}
- Energy column: {state['energy_col']}
- Time column: {state['time_col'] or 'none detected'}
- Time features: {'derived from the detected timestamp column' if state.get('temporal_source') == 'timestamp' else 'generated because no timestamp column was detected'}
- Mean: {mean:.2f} kWh
- Median: {median:.2f} kWh
- Min: {minimum:.2f} kWh
- Max/peak: {peak:.2f} kWh
- Std deviation: {std:.2f}
- 25th percentile: {p25:.2f}
- 75th percentile: {p75:.2f}
- Total: {total/1000:.2f} MWh
- Anomaly rule: values below {lower:.2f} or above {upper:.2f} kWh
- Estimated anomalies with that rule: {anomalies}
- Hourly pattern: {hourly_summary}
- Daily averages: {daily_summary}
""".strip()

def local_ai_reply(user_msg):
    msg = (user_msg or "").lower().strip()
    if not state['vals']:
        if any(word in msg for word in ["hi", "hello", "hey", "salam"]):
            return (
                "Hi. I am ready to help with your energy dataset. Upload a CSV, "
                "then ask me about anomalies, peak hours, consumption trends, or efficiency tips."
            )
        return (
            "I can help with anomalies, peaks, forecasts, and efficiency tips, "
            "but no dataset is loaded yet. Upload a CSV first, then ask me about "
            "the consumption pattern or unusual points."
        )

    vals = state['vals']
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    peak = float(np.max(vals))
    total = float(np.sum(vals))
    upper = mean + 2.5 * std
    lower = mean - 2.5 * std
    anomalies = sum(1 for v in vals if v > upper or v < lower)
    peak_hour = None
    low_hour = None
    hourly = []
    if state['hour_buckets'] and state['hour_cnt']:
        hourly = [
            state['hour_buckets'][i] / state['hour_cnt'][i]
            if state['hour_cnt'][i] else 0
            for i in range(24)
        ]
        peak_hour = int(np.argmax(hourly))
        low_hour = int(np.argmin(hourly))

    data_line = (
        f"Your loaded dataset has {len(vals):,} cleaned records, an average load of "
        f"{mean:,.2f} kWh, and a peak reading of {peak:,.2f} kWh."
    )

    if any(word in msg for word in ["hi", "hello", "hey", "salam"]):
        return (
            "Hi. I can see your dataset is loaded now. Ask me things like "
            "\"what are anomalies?\", \"where is the peak hour?\", or "
            "\"how can I reduce consumption?\""
        )

    if any(phrase in msg for phrase in ["what anomaly", "what anomalie", "what is anomaly", "what are anomalies", "anomalie mean", "anomaly mean"]):
        return (
            "**What an anomaly means**\n"
            "An anomaly is a reading that looks unusual compared with the normal pattern of your data. "
            "In this app, I estimate anomalies using the average load and standard deviation.\n\n"
            f"For your dataset, a value is treated as unusual if it is below {lower:,.2f} kWh "
            f"or above {upper:,.2f} kWh. With that rule, I currently find {anomalies} anomaly points.\n\n"
            "So if the anomaly count is 0, it does not mean the system is perfect. It means no point is extreme "
            "enough for this statistical threshold. A lower threshold could detect smaller irregular changes."
        )

    if any(word in msg for word in ["anomaly", "anomalies", "spike", "outlier", "unusual"]):
        return (
            "**Anomaly check**\n"
            f"I checked values outside mean +/- 2.5 standard deviations. That gives {anomalies} anomaly points.\n\n"
            f"- Lower limit: {lower:,.2f} kWh\n"
            f"- Upper limit: {upper:,.2f} kWh\n"
            f"- Peak reading: {peak:,.2f} kWh\n\n"
            "If you expected anomalies but see 0, your data may be smooth after outlier cleaning, or the threshold "
            "may be too strict for this dataset."
        )

    if any(word in msg for word in ["tip", "efficiency", "improve", "saving", "reduce", "optimization", "optimise", "optimize"]):
        peak_text = f"{peak_hour}:00" if peak_hour is not None else "the peak period"
        low_text = f"{low_hour}:00" if low_hour is not None else "lower-load hours"
        return (
            "**Efficiency tips for this dataset**\n"
            f"{data_line}\n\n"
            f"- Your highest average load is around {peak_text}. Move flexible consumption away from that hour when possible.\n"
            f"- The lowest average hour is around {low_text}. Schedule batch jobs, charging, or non-critical loads there.\n"
            f"- Since the current anomaly count is {anomalies}, focus first on peak reduction rather than fault investigation.\n"
            "- Compare weekday and weekend profiles in the heatmap to find standby loads that stay high when activity should be lower."
        )

    if any(word in msg for word in ["peak", "highest", "maximum", "hour"]):
        if peak_hour is None:
            return f"{data_line} I could not identify an hourly peak because no valid time/hour buckets are available."
        return (
            "**Peak load insight**\n"
            f"The highest average hour is {peak_hour}:00. The single highest reading is {peak:,.2f} kWh.\n\n"
            "This is the best place to start if your goal is demand reduction: check which processes, buildings, "
            "or devices are active around that hour."
        )

    if any(word in msg for word in ["summary", "summarize", "pattern", "overview", "describe"]):
        peak_text = f" The highest average hour is {peak_hour}:00." if peak_hour is not None else ""
        return (
            "**Dataset summary**\n"
            f"{data_line}{peak_text} Estimated anomaly points: {anomalies}.\n\n"
            "The main story in this dataset is the load level and its hourly profile. For a deeper diagnosis, "
            "ask about anomalies, peak hours, or efficiency improvements."
        )

    return (
        "I understand. For this dataset, I can answer best about anomalies, peak hours, summaries, "
        "and efficiency actions.\n\n"
        f"Quick context: {data_line}"
    )

def friendly_gemini_error(exc):
    raw = str(exc)
    retry_match = re.search(r"retryDelay': '(\d+)s'", raw) or re.search(r"retry in ([\d.]+)s", raw)
    retry = retry_match.group(1) if retry_match else None
    if "RESOURCE_EXHAUSTED" in raw or "quota" in raw.lower() or "429" in raw:
        wait = f" Wait about {retry} seconds and try again." if retry else ""
        return (
            "Gemini quota is exhausted for this API key/model. "
            "This is a Google free-tier limit, not a dataset or dashboard bug."
            f"{wait} You can also wait for quota reset, use another API key, upgrade billing, "
            "or set AI_ALLOW_LOCAL_FALLBACK=1 in .env if you want the dashboard to answer locally when Gemini is unavailable."
        ), 429
    if "API key" in raw or "permission" in raw.lower() or "unauthorized" in raw.lower():
        return "Gemini rejected the API key. Check GEMINI_API_KEY in .env, then restart Flask.", 401
    return f"Gemini API error: {raw}", 503

@app.route('/ai', methods=['POST'])
def ai_chat():
    data = request.get_json(silent=True) or {}
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({'error': 'No message'}), 400

    ctx = dataset_context_text()

    system_prompt = (
        "You are SmartGrid AI, a sharp, natural energy-data analyst inside a Flask dashboard. "
        "Talk like a helpful expert, not like a template. Answer the user's exact question first. "
        "Use the provided dataset statistics when relevant, explain technical words simply, and give "
        "practical next steps. If the user asks in imperfect English, understand the intent and answer kindly. "
        "Do not claim to inspect raw rows beyond the summarized context. Do not repeat the full stats block unless "
        "the user asks for a summary. Keep most answers short, clear, and useful.\n\n"
        f"Current dataset context:\n{ctx}"
    )

    if not client:
        if ALLOW_LOCAL_AI_FALLBACK:
            return jsonify({'ok': True, 'reply': local_ai_reply(user_msg), 'source': 'local'})
        return jsonify({
            'error': (
                'Gemini API key is not configured. Create a .env file with '
                'GEMINI_API_KEY=your_key or set the GEMINI_API_KEY environment variable, then restart Flask.'
            )
        }), 503

    try:
        config = None
        if types:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.55,
                max_output_tokens=900,
            )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_msg,
            config=config,
        )
        reply = response.text
    except Exception as e:
        friendly, status = friendly_gemini_error(e)
        if ALLOW_LOCAL_AI_FALLBACK:
            return jsonify({
                'ok': True,
                'reply': local_ai_reply(user_msg),
                'source': 'local',
                'warning': friendly
            })
        return jsonify({'error': friendly}), status

    return jsonify({'ok': True, 'reply': reply, 'source': 'gemini', 'model': GEMINI_MODEL})   
# ─────────────────────────────────────────────
# RUN FLASK APP
# ─────────────────────────────────────────────
if __name__ == '__main__':

    print("=" * 50)
    print("SmartGrid OS - Flask Backend")
    print("http://localhost:5000")
    print("=" * 50)

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )
