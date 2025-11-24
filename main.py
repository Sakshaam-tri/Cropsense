from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests
import math
import joblib
import json
import numpy as np
import os

app = FastAPI()

# ------------------------------------------------------------------
# CORS
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for development; later restrict to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# PATHS & GOOGLE DRIVE CONFIG
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_CACHE_DIR = BASE_DIR / "data_cache"
MODEL_DIR = DATA_CACHE_DIR / "models"
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def gdrive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

# TODO: put your actual Google Drive file IDs here
GDRIVE_FILES = {
    "profit_csv": gdrive_url("1U2-xdy7cpc38HJcP28wVsPidmO1sFI17"),  # remains same
    "price_csv": gdrive_url("13Ah_-KlGfWLRcsM_qFX8ScxE1xruGxs6"),   # UPDATED ✔
    "profit_model": gdrive_url("193jLIjrEvRDsufh-qVwMQSvkcRdCdysV"),
    "profit_features": gdrive_url("1sCWaMerNLSVnR_SRxU8m4rZAM5AtK7rr"),
    "crop_encoder": gdrive_url("1zNMRZ6IIqXJnXDWDx-E2LMzncebLDBNy"),
    "season_encoder": gdrive_url("16_tfjGLWCSEywiBBSeSGhwyAYuyQuzQi"),
}


def download_file_if_needed(url: str, path: Path):
    """Download from Google Drive only if local file is missing."""
    if path.exists():
        return
    if not url or "YOUR_" in url:
        # user hasn't configured this yet
        raise RuntimeError(f"Google Drive URL for {path.name} is not configured.")
    resp = requests.get(url, timeout=6000)
    resp.raise_for_status()
    path.write_bytes(resp.content)

# local cached file paths
PROFIT_CSV_PATH = DATA_CACHE_DIR / "crop_profit_dataset.csv"
PRICE_CSV_PATH = DATA_CACHE_DIR / "Price_Agriculture_commodities_Week.csv"
PROFIT_MODEL_PATH = MODEL_DIR / "profit_model.pkl"
PROFIT_FEATURES_PATH = MODEL_DIR / "profit_features.json"
CROP_ENCODER_PATH = MODEL_DIR / "crop_encoder.pkl"
SEASON_ENCODER_PATH = MODEL_DIR / "season_encoder.pkl"

# ------------------------------------------------------------------
# DOWNLOAD DATASETS & MODELS FROM GOOGLE DRIVE
# ------------------------------------------------------------------
df_profit = pd.DataFrame()
df_price = pd.DataFrame()

try:
    download_file_if_needed(GDRIVE_FILES["profit_csv"], PROFIT_CSV_PATH)
    df_profit = pd.read_csv(PROFIT_CSV_PATH)
except Exception as e:
    print("ERROR: could not load profit dataset:", e)

try:
    download_file_if_needed(GDRIVE_FILES["price_csv"], PRICE_CSV_PATH)
    df_price = pd.read_csv(PRICE_CSV_PATH)
except Exception as e:
    print("ERROR: could not load price dataset:", e)

# Normalize columns if data loaded
if not df_profit.empty:
    df_profit["STATE_U"] = df_profit["State"].astype(str).str.strip().str.upper()
    df_profit["DISTRICT_U"] = df_profit["District"].astype(str).str.strip().str.upper()
    df_profit["CROP_U"] = df_profit["Crop"].astype(str).str.strip().str.upper()

    if "Season" in df_profit.columns:
        df_profit["SEASON_U"] = df_profit["Season"].astype(str).str.strip().str.upper()
    else:
        df_profit["Season"] = "UNKNOWN"
        df_profit["SEASON_U"] = "UNKNOWN"

if not df_price.empty:
    df_price["State"] = df_price["State"].astype(str).str.strip()
    df_price["District"] = df_price["District"].astype(str).str.strip()
    df_price["Commodity"] = df_price["Commodity"].astype(str).str.strip()

    df_price["STATE_U"] = df_price["State"].str.upper()
    df_price["DISTRICT_U"] = df_price["District"].str.upper()
    df_price["CROP_U"] = df_price["Commodity"].str.upper()

# ------------------------------------------------------------------
# LOAD PROFIT ML MODEL (RandomForestRegressor)
# ------------------------------------------------------------------
PROFIT_MODEL_AVAILABLE = False
PROFIT_FEATURES: list[str] = []
crop_le_profit = None
season_le_profit = None
feature_means = None

try:
    # Download model artifacts if they exist
    download_file_if_needed(GDRIVE_FILES["profit_model"], PROFIT_MODEL_PATH)
    download_file_if_needed(GDRIVE_FILES["profit_features"], PROFIT_FEATURES_PATH)
    download_file_if_needed(GDRIVE_FILES["crop_encoder"], CROP_ENCODER_PATH)

    # season encoder is optional
    try:
        download_file_if_needed(GDRIVE_FILES["season_encoder"], SEASON_ENCODER_PATH)
        has_season_encoder = True
    except Exception:
        has_season_encoder = False

    profit_model = joblib.load(PROFIT_MODEL_PATH)

    with open(PROFIT_FEATURES_PATH, "r") as f:
        PROFIT_FEATURES = json.load(f)

    crop_le_profit = joblib.load(CROP_ENCODER_PATH)
    season_le_profit = joblib.load(SEASON_ENCODER_PATH) if has_season_encoder else None

    if not df_profit.empty:
        df_profit["Crop_id"] = crop_le_profit.transform(df_profit["Crop"].astype(str))
        if season_le_profit is not None and "Season" in df_profit.columns:
            df_profit["Season_id"] = season_le_profit.transform(df_profit["Season"].astype(str))
        else:
            df_profit["Season_id"] = 0

        missing_cols = [c for c in PROFIT_FEATURES if c not in df_profit.columns]
        if missing_cols:
            print("WARNING: Missing columns for profit model:", missing_cols)
        else:
            PROFIT_MODEL_AVAILABLE = True
            feature_means = df_profit[PROFIT_FEATURES].mean()
            print("Profit model and encoders loaded successfully.")
    else:
        print("WARNING: df_profit is empty; ML model disabled.")

except Exception as e:
    print("Could not load profit model or encoders:", e)
    PROFIT_MODEL_AVAILABLE = False
    PROFIT_FEATURES = []
    feature_means = None

# ------------------------------------------------------------------
# USER & ADMIN DATA (CSV STORAGE - LOCAL)
# ------------------------------------------------------------------
USER_DATA_DIR = BASE_DIR / "user_data"
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

USER_CSV_PATH = USER_DATA_DIR / "users.csv"
ADMIN_CSV_PATH = USER_DATA_DIR / "admins.csv"

USER_COLUMNS = ["username", "password", "created_at", "role"]
ADMIN_COLUMNS = ["username", "password", "created_at", "role"]

def load_users() -> pd.DataFrame:
    if USER_CSV_PATH.exists():
        df_users = pd.read_csv(USER_CSV_PATH)
        for col in USER_COLUMNS:
            if col not in df_users.columns:
                df_users[col] = None
        return df_users[USER_COLUMNS]
    else:
        return pd.DataFrame(columns=USER_COLUMNS)

def save_users(df_users: pd.DataFrame):
    df_users.to_csv(USER_CSV_PATH, index=False)

def load_admins() -> pd.DataFrame:
    if ADMIN_CSV_PATH.exists():
        df_admins = pd.read_csv(ADMIN_CSV_PATH)
    else:
        df_admins = pd.DataFrame(columns=ADMIN_COLUMNS)

    for col in ADMIN_COLUMNS:
        if col not in df_admins.columns:
            df_admins[col] = None

    mask = df_admins["username"].astype(str) == "Saksham"
    if not mask.any():
        new_row = {
            "username": "Saksham",
            "password": "1234",
            "created_at": datetime.utcnow().isoformat(),
            "role": "admin",
        }
        df_admins = pd.concat([df_admins, pd.DataFrame([new_row])], ignore_index=True)
        df_admins.to_csv(ADMIN_CSV_PATH, index=False)
    else:
        df_admins.loc[mask, "password"] = "1234"
        df_admins.loc[mask, "role"] = "admin"
        df_admins.to_csv(ADMIN_CSV_PATH, index=False)

    return df_admins[ADMIN_COLUMNS]

def save_admins(df_admins: pd.DataFrame):
    df_admins.to_csv(ADMIN_CSV_PATH, index=False)

# ------------------------------------------------------------------
# GROWTH & SEASON DATA
# ------------------------------------------------------------------
GROWTH_DAYS = {
    "RICE": 120,
    "WHEAT": 140,
    "MAIZE": 110,
    "COTTON": 160,
    "SUGARCANE": 300,
    "GRAM": 110,
    "PULSES": 100,
}

CROP_SEASON_PREF = {
    "RICE": "KHARIF",
    "PADDY": "KHARIF",
    "MAIZE": "KHARIF",
    "COTTON": "KHARIF",
    "SOYABEAN": "KHARIF",
    "BAJRA": "KHARIF",
    "JOWAR": "KHARIF",

    "WHEAT": "RABI",
    "GRAM": "RABI",
    "MUSTARD": "RABI",
    "LENTIL": "RABI",
    "CHICKPEA": "RABI",
    "BARLEY": "RABI",
    "PEA": "RABI",

    "POTATO": "ZAID",
    "VEGETABLE": "ZAID",
    "VEGETABLES": "ZAID",
    "ONION": "ZAID",
    "TOMATO": "ZAID",
}

IRRIGATION_INFO = {
    "RICE": {
        "irrigation": "Flood / canal / tube-well irrigation; standing water in field",
        "fertilizers": "Basal NPK + urea top-dressing; FYM if available",
    },
    "WHEAT": {
        "irrigation": "Irrigation at CRI, tillering, booting & grain filling",
        "fertilizers": "Balanced NPK + zinc if deficient",
    },
    "MAIZE": {
        "irrigation": "Furrow / drip at knee-height, tasseling & grain-filling",
        "fertilizers": "High NPK with split N doses",
    },
    "COTTON": {
        "irrigation": "Furrow / drip; avoid waterlogging",
        "fertilizers": "Balanced NPK with potash; micronutrients where needed",
    },
    "GRAM": {
        "irrigation": "Mostly rainfed; one irrigation at pod-initiation",
        "fertilizers": "Starter N + P with Rhizobium",
    },
}

DEFAULT_IRRIGATION = {
    "irrigation": "Based on local source: canal / tube-well / rainfed",
    "fertilizers": "Soil-test based NPK + organic manure/FYM",
}

MONTH_MAP = {
    "JANUARY": 1,
    "FEBRUARY": 2,
    "MARCH": 3,
    "APRIL": 4,
    "MAY": 5,
    "JUNE": 6,
    "JULY": 7,
    "AUGUST": 8,
    "SEPTEMBER": 9,
    "OCTOBER": 10,
    "NOVEMBER": 11,
    "DECEMBER": 12,
}

def month_to_number(m: str) -> int:
    if not m:
        return 1
    m = m.strip().upper()
    return MONTH_MAP.get(m, 1)

def month_to_season(m_num: int) -> str:
    if 6 <= m_num <= 9:
        return "KHARIF"
    elif m_num >= 10 or m_num <= 2:
        return "RABI"
    else:
        return "ZAID"

def crop_pref_season(crop_u: str) -> Optional[str]:
    for key, season in CROP_SEASON_PREF.items():
        if key in crop_u:
            return season
    return None

def adjust_profit(base_profit: float, crop_name: str, month: str) -> float:
    month_num = month_to_number(month)
    season_u = month_to_season(month_num)
    crop_u = crop_name.upper()
    pref = crop_pref_season(crop_u)

    if pref is None:
        season_factor = 1.0
    elif pref == season_u:
        season_factor = 1.2
    else:
        season_factor = 0.8

    angle = (month_num - 1) / 12.0 * 2 * math.pi
    month_factor = 1.0 + 0.1 * math.sin(angle)

    return base_profit * season_factor * month_factor

def adjust_market_price(base_price: Optional[float], crop_name: str, month: str) -> Optional[float]:
    if base_price is None:
        return None
    month_num = month_to_number(month)
    season_u = month_to_season(month_num)
    crop_u = crop_name.upper()
    pref = crop_pref_season(crop_u)

    if pref is None:
        season_factor = 1.0
    elif pref == season_u:
        season_factor = 1.2
    else:
        season_factor = 0.8

    angle = (month_num - 1) / 12.0 * 2 * math.pi
    month_factor = 1.0 + 0.05 * math.sin(angle)

    return base_price * season_factor * month_factor

# ------------------------------------------------------------------
# REQUEST MODELS
# ------------------------------------------------------------------
class PredictRequest(BaseModel):
    state: str
    district: str
    month: str

class UserCredentials(BaseModel):
    username: str
    password: str

# ------------------------------------------------------------------
# OPEN-METEO HELPERS
# ------------------------------------------------------------------
def geocode(state: str, district: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    q = f"{district}, {state}, India"
    try:
        r = requests.get(url, params={"name": q, "count": 1}, timeout=5)
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        if not results:
            return None, None
        first = results[0]
        return float(first["latitude"]), float(first["longitude"])
    except Exception:
        return None, None

def get_weather(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,pressure_msl,precipitation",
        "timezone": "auto",
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        cur = r.json().get("current", {})
        return {
            "temp": cur.get("temperature_2m"),
            "humidity": cur.get("relative_humidity_2m"),
            "pressure": cur.get("pressure_msl"),
            "precip_mm": cur.get("precipitation"),
        }
    except Exception:
        return {
            "temp": None,
            "humidity": None,
            "pressure": None,
            "precip_mm": None,
        }

def get_river_discharge(lat: float, lon: float):
    url = "https://flood-api.open-meteo.com/v1/flood"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "river_discharge",
        "forecast_days": 1,
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        daily = r.json().get("daily", {})
        discharge = daily.get("river_discharge")
        if discharge and len(discharge) > 0:
            return float(discharge[0])
        return None
    except Exception:
        return None

def compute_risk(weather: dict, discharge: Optional[float]):
    temp = weather.get("temp") or 30.0
    humidity = weather.get("humidity") or 60.0
    precip = weather.get("precip_mm") or 0.0
    if discharge is None:
        discharge = 100.0

    flood_raw = max(0.0, discharge / 2000.0 + precip / 40.0)
    drought_raw = 0.0
    if temp > 32:
        drought_raw += (temp - 32) / 18.0
    if humidity < 60:
        drought_raw += (60 - humidity) / 60.0
    if precip > 5:
        drought_raw -= precip / 20.0
    drought_raw = max(0.0, drought_raw)

    normal_raw = 1.0
    total = flood_raw + drought_raw + normal_raw
    if total == 0:
        flood_raw = drought_raw = 0.0
        normal_raw = 1.0
        total = 1.0

    flood_pct = round(flood_raw / total * 100, 1)
    drought_pct = round(drought_raw / total * 100, 1)
    normal_pct = round(normal_raw / total * 100, 1)

    adjust = 100.0 - (flood_pct + drought_pct + normal_pct)
    if abs(adjust) >= 0.2:
        normal_pct = round(normal_pct + adjust, 1)

    if flood_pct > 50:
        warning = "⚠️ High flood risk – be careful."
    elif drought_pct > 50:
        warning = "⚠️ High drought risk – be careful."
    else:
        warning = "✔️ Conditions look reasonably normal."

    return {
        "normal": normal_pct,
        "flood": flood_pct,
        "drought": drought_pct,
        "warning": warning,
    }

# ------------------------------------------------------------------
# PRECOMPUTE LOCATIONS FOR DROPDOWNS
# ------------------------------------------------------------------
LOCATIONS: List[Dict[str, Any]] = []
if not df_profit.empty:
    loc_df = df_profit[
        ["State", "STATE_U", "District", "DISTRICT_U"]
    ].dropna().drop_duplicates()

    locations_map: Dict[str, Dict[str, Any]] = {}
    for _, row in loc_df.iterrows():
        state_name = str(row["State"]).strip()
        state_u = str(row["STATE_U"]).strip()
        district_name = str(row["District"]).strip()

        if state_u not in locations_map:
            locations_map[state_u] = {
                "state": state_name,
                "state_u": state_u,
                "districts": set(),
            }
        locations_map[state_u]["districts"].add(district_name)

    for st_u, info in locations_map.items():
        LOCATIONS.append(
            {
                "state": info["state"],
                "state_u": info["state_u"],
                "districts": sorted(list(info["districts"])),
            }
        )
    LOCATIONS = sorted(LOCATIONS, key=lambda x: x["state"])

# ------------------------------------------------------------------
# AUTH ROUTES
# ------------------------------------------------------------------
@app.post("/api/user/signup")
def user_signup(user: UserCredentials):
    df_users = load_users()
    if not df_users.empty and (df_users["username"].astype(str).str.lower() == user.username.lower()).any():
        return {"error": "Username already exists. Please choose another."}

    new_row = {
        "username": user.username,
        "password": user.password,
        "created_at": datetime.utcnow().isoformat(),
        "role": "user",
    }
    df_users = pd.concat([df_users, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df_users)
    return {
        "message": "Account created successfully.",
        "username": user.username,
        "role": "user",
    }

@app.post("/api/user/login")
def user_login(user: UserCredentials):
    df_users = load_users()
    if df_users.empty:
        return {"error": "No users registered yet."}

    mask = (df_users["username"].astype(str) == user.username) & (
        df_users["password"].astype(str) == user.password
    )
    if not mask.any():
        return {"error": "Invalid username or password."}

    return {
        "message": "User login successful.",
        "username": user.username,
        "role": "user",
    }

@app.post("/api/user/delete")
def user_delete(user: UserCredentials):
    df_users = load_users()
    if df_users.empty:
        return {"error": "No users registered."}

    mask = (df_users["username"].astype(str) == user.username) & (
        df_users["password"].astype(str) == user.password
    )
    if not mask.any():
        return {"error": "Username or password incorrect. Cannot delete account."}

    df_users = df_users.loc[~mask].reset_index(drop=True)
    save_users(df_users)
    return {"message": "Account deleted successfully."}

@app.post("/api/admin/login")
def admin_login(user: UserCredentials):
    df_admins = load_admins()
    mask = (df_admins["username"].astype(str) == user.username) & (
        df_admins["password"].astype(str) == user.password
    )
    if not mask.any():
        return {"error": "Invalid admin credentials."}

    return {
        "message": "Admin login successful.",
        "username": user.username,
        "role": "admin",
    }

# ------------------------------------------------------------------
# ROOT & LOCATIONS ROUTE
# ------------------------------------------------------------------
@app.get("/")
def root():
    if PROFIT_MODEL_AVAILABLE:
        return {"message": "Crop backend using ML profit model + mandi price + live weather/river"}
    else:
        return {"message": "Crop backend using historical profit + mandi price + live weather/river (ML model not loaded)"}

@app.get("/api/locations")
def get_locations():
    return {"locations": LOCATIONS}

# ------------------------------------------------------------------
# MAIN PREDICT ROUTE
# ------------------------------------------------------------------
@app.post("/api/predict")
def predict(req: PredictRequest):
    if df_profit.empty:
        return {"error": "Profit dataset not loaded on server."}

    state_u = req.state.strip().upper()
    district_u = req.district.strip().upper()
    month_num = month_to_number(req.month)
    season_u = month_to_season(month_num)

    sub = df_profit[(df_profit["STATE_U"] == state_u) & (df_profit["DISTRICT_U"] == district_u)]
    if sub.empty:
        return {"error": "No historical data for this state & district."}

    if "SEASON_U" in sub.columns:
        sub_season = sub[sub["SEASON_U"] == season_u]
        if not sub_season.empty:
            sub = sub_season

    if sub.empty:
        return {"error": "No data for this state, district and season."}

    # ---------------- ML PROFIT PREDICTION ----------------
    if PROFIT_MODEL_AVAILABLE and PROFIT_FEATURES and feature_means is not None:
        X_sub = sub[PROFIT_FEATURES].copy()
        X_sub = X_sub.fillna(feature_means)
        y_pred_sub = profit_model.predict(X_sub)

        sub = sub.copy()
        sub["profit_pred_model"] = y_pred_sub

        agg_dict = {"profit_pred_model": "mean"}
        if "cost_per_ha" in sub.columns:
            agg_dict["cost_per_ha"] = "mean"

        grouped = sub.groupby("Crop", as_index=False).agg(agg_dict)
        grouped = grouped.rename(columns={"profit_pred_model": "avg_profit_per_ha"})
        if "cost_per_ha" in grouped.columns:
            grouped = grouped.rename(columns={"cost_per_ha": "avg_cost_per_ha"})
    else:
        agg_dict = {"profit_per_ha": "mean"}
        if "cost_per_ha" in sub.columns:
            agg_dict["cost_per_ha"] = "mean"

        grouped = sub.groupby("Crop", as_index=False).agg(agg_dict)
        grouped = grouped.rename(columns={"profit_per_ha": "avg_profit_per_ha"})
        if "cost_per_ha" in grouped.columns:
            grouped = grouped.rename(columns={"cost_per_ha": "avg_cost_per_ha"})

    if grouped.empty:
        return {"error": "No crops found after aggregation."}

    grouped["adj_profit_per_ha"] = grouped.apply(
        lambda r: adjust_profit(float(r["avg_profit_per_ha"]), str(r["Crop"]), req.month),
        axis=1,
    )

    grouped = grouped.sort_values("adj_profit_per_ha", ascending=False)
    top3 = grouped.head(3)

    lat, lon = geocode(req.state, req.district)
    if lat is None or lon is None:
        lat, lon = 23.5, 80.5

    weather = get_weather(lat, lon)
    discharge = get_river_discharge(lat, lon)
    risk = compute_risk(weather, discharge)

    crops_out = []
    for _, row in top3.iterrows():
        crop_name = str(row["Crop"])
        crop_u = crop_name.upper()
        base_profit = float(row["avg_profit_per_ha"])
        profit_adj = float(row["adj_profit_per_ha"])
        cost = (
            float(row["avg_cost_per_ha"])
            if "avg_cost_per_ha" in row and not pd.isna(row["avg_cost_per_ha"])
            else None
        )

        sub_price = pd.DataFrame()
        if not df_price.empty:
            sub_price = df_price[
                (df_price["STATE_U"] == state_u)
                & (df_price["DISTRICT_U"] == district_u)
                & (df_price["CROP_U"] == crop_u)
            ]
            if sub_price.empty:
                sub_price = df_price[df_price["CROP_U"] == crop_u]

        if not df_price.empty and "Modal Price" in df_price.columns:
            if not sub_price.empty:
                base_price = float(sub_price["Modal Price"].mean())
            else:
                base_price = float(df_price["Modal Price"].mean())
        else:
            base_price = None

        market_price_adj = adjust_market_price(base_price, crop_name, req.month)

        days = GROWTH_DAYS.get(crop_u, 120)
        months = round(days / 30, 1)

        info = IRRIGATION_INFO.get(crop_u, DEFAULT_IRRIGATION)

        crops_out.append(
            {
                "crop": crop_name,
                "profit_per_ha": round(profit_adj, 2),
                "profit_for_1_ha": round(profit_adj, 2),
                "base_profit_per_ha": round(base_profit, 2),
                "cost_per_ha": cost,
                "market_price": round(market_price_adj, 2) if market_price_adj is not None else None,
                "growth_days": days,
                "growth_months": months,
                "irrigation": info["irrigation"],
                "fertilizers": info["fertilizers"],
            }
        )

    return {
        "state": req.state,
        "district": req.district,
        "month": req.month,
        "month_num": month_num,
        "season": season_u,
        "lat": lat,
        "lon": lon,
        "weather": weather,
        "river_discharge_m3s": discharge,
        "risk": risk,
        "best_crops": crops_out,
    }

