# SmartGrid OS - SG02 Energy Consumption Analysis

SmartGrid OS is a Python data analysis project for **SG02: Prediction and Analysis of Energy Consumption**. It analyzes building power consumption data, builds decision indicators, runs anomaly detection, uses a regression model for prediction, and provides an interactive Flask dashboard.

Dataset used:
[Building Sites Power Consumption Dataset - Kaggle](https://www.kaggle.com/datasets/arashnic/building-sites-power-consumption-dataset)

## Project Features

- CSV upload and automatic parsing.
- Data cleaning with Pandas and NumPy.
- Missing values, duplicate handling, and outlier filtering.
- Time-series features: hour, day of week, month, weekend, season.
- KPIs: total consumption, average load, peak load, efficiency score.
- Charts: timeline, rolling average, distribution, heatmap, monthly average.
- Real day/hour heatmap when a timestamp column exists.
- Anomaly detection with z-score thresholds.
- Regression prediction using a saved Scikit-Learn model.
- Gemini AI assistant using the uploaded dataset context.
- Live replay mode: streams real rows from the uploaded CSV to simulate real-time data.

## Project Structure

```text
SmartGrid-OS/
  app.py
  requirements.txt
  .env.example
  .gitignore
  README.md
  frontend/
    index.html
  models/
    energy_model.pkl
  Project_Building_dataset_.ipynb
```

## Important: Real-Time Explanation

This dashboard does **not** connect to real IoT sensors. The real-time feature is a **live replay**:

> The app reads real rows from the uploaded CSV and streams them gradually to the dashboard.

This is useful for demonstration because it simulates a real-time energy feed while still using real dataset values.

## Fresh Laptop Installation

Use these steps after cloning the project from GitHub.

### 1. Clone the repository

```powershell
git clone https://github.com/Shouaib-gf/SmartGrid
cd SmartGrid-OS
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```



### Quick command to change only the Gemini API key

If you want to quickly replace the API key, run this in PowerShell:

```powershell
Set-Content .env "GEMINI_API_KEY=YOUR_NEW_API_KEY_HERE`nGEMINI_MODEL=gemini-2.5-flash`nAI_ALLOW_LOCAL_FALLBACK=1"
```

Then restart the app.

### 5. Run the Flask app

```powershell
python app.py
```

Open:

[http://localhost:5000](http://localhost:5000)

## How To Use The Dashboard

1. Start the app with `python app.py`.
2. Open [http://localhost:5000](http://localhost:5000).
3. Go to **Upload Data**.
4. Upload the Kaggle consumption CSV.
5. If needed, choose the correct columns:
   - Energy column: `Value`
   - Time column: `Timestamp`
6. Click re-analyze if you changed the mapping.
7. Use:
   - Dashboard for KPIs.
   - Time Series for rolling analysis.
   - Heatmap for day/hour patterns.
   - Prediction for ML inference.
   - Anomaly Detector for unusual points.
   - AI Assistant for Gemini analysis.
   - Live Replay for simulated real-time streaming from real CSV rows.

## Changing The Gemini API Key

The app reads the key from `.env`.

To change it:

```powershell
Set-Content .env "GEMINI_API_KEY=YOUR_NEW_API_KEY_HERE`nGEMINI_MODEL=gemini-2.5-flash`nAI_ALLOW_LOCAL_FALLBACK=1"
python app.py
```

If Gemini shows a quota error, it means the free limit of your key was reached. You can:

- wait for the quota reset,
- use another API key,
- enable billing,
- or keep `AI_ALLOW_LOCAL_FALLBACK=1` so the app can answer locally when Gemini is unavailable.


## Notes For The Report

Recommended wording:

> SmartGrid OS is an interactive dashboard for analyzing building energy consumption. It transforms uploaded CSV data into decision indicators, visualizations, anomaly alerts, and prediction outputs. The real-time module is implemented as a live replay of real dataset rows, simulating continuous data arrival for demonstration purposes.

French wording:

> SmartGrid OS est un tableau de bord interactif pour l'analyse de la consommation énergétique des bâtiments. Il transforme les données CSV en indicateurs décisionnels, visualisations, alertes d'anomalies et prédictions. Le module temps réel est un replay progressif des vraies lignes du dataset, simulant l'arrivée continue des données pour la démonstration.

