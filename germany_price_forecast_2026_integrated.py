from __future__ import annotations

import os
import sys
import warnings
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, DatetimeIndex, Timestamp, Timedelta
import requests

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

TIMEZONE = "Europe/Berlin"
TRAIN_START = "2023-01-01"
TRAIN_END = "2025-12-31"
FORECAST_START = "2026-01-01"
FORECAST_END = "2026-12-31"

# API endpoints
EC_BASE = "https://api.energy-charts.info"
DE_CTRL_AREA = "10Y1001A1001A83F"  # ENTSO-E EIC for imbalance
DE_LU_CC = "DE_LU"  # ENTSO-E country code for DA + wind/solar
DE_CC = "DE"  # ENTSO-E country code for load

# DA forecast parameters
DA_SCENARIO = 1.00  # 1.00=P50, 1.20=P10, 0.80=P90

USE_EQUAL_WEIGHTS = True  # Set False to revert to exponential weighting
LAMBDA_EW = 0.87  # Only used if USE_EQUAL_WEIGHTS = False


SPREAD_ENHANCEMENT = 1.15  # Applied after profile construction

HIST_YEARS = [2023, 2024, 2025]

# IDC parameters
BID_ASK_SPREAD = 0.02  # ±1% spread
CHUNK_DAYS = 31  # reBAP chunk size

# FCR parameters
FCR_BLOCKS_PER_DAY = 6
BLOCK_STARTS = [0, 4, 8, 12, 16, 20]

# XGBoost model configuration (for IDC)
XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'tree_method': 'hist',
    'random_state': 42,
}

# Output path
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "Germany_priceforecast_2026.xlsx")

print(f"""
{'='*70}
GERMANY ELECTRICITY MARKET PRICE FORECAST 2026
Integrated Pipeline - All Markets
{'='*70}
""")


def _to_dti(idx: object) -> DatetimeIndex:
    """Cast to DatetimeIndex."""
    if isinstance(idx, DatetimeIndex):
        return idx
    return pd.DatetimeIndex(idx)


def _to_15min(df: DataFrame) -> DataFrame:
    """Forward-fill hourly to 15-min."""
    try:
        freq = pd.infer_freq(_to_dti(df.index))
    except Exception:
        freq = None
    if freq in ("15T", "15min", "QH"):
        return df
    return df.resample("15min").ffill()


def _find_col(df: DataFrame, keywords: List[str]) -> Optional[str]:
    """Find column matching keywords."""
    for col in df.columns:
        if any(kw.lower() in str(col).lower() for kw in keywords):
            return col
    return None


def _make_full_index(start: str, end: str) -> DatetimeIndex:
    """Create complete 15-min DatetimeIndex."""
    return pd.date_range(
        start=f"{start} 00:00",
        end=f"{end} 23:45",
        freq="15min",
        tz=TIMEZONE,
    )


def _ec_get(endpoint: str, params: dict, label: str) -> dict:
    """Energy-Charts API request."""
    url = f"{EC_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    if not resp.ok:
        print(f"  [ERROR] {label}: HTTP {resp.status_code}")
        resp.raise_for_status()
    return resp.json()


def _ec_to_df(data: dict, col: str, label: str) -> DataFrame:
    """Parse Energy-Charts response to DataFrame."""
    for key in ("unix_seconds", "price"):
        if key not in data:
            raise ValueError(f"Missing '{key}' in {label} response")
    ts = pd.to_datetime(data["unix_seconds"], unit="s", utc=True).tz_convert(TIMEZONE)
    df = DataFrame({col: data["price"]}, index=ts)
    df.index.name = "timestamp"
    df = df.dropna(subset=[col])
    df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()


def fetch_da_ec(start_date: str, end_date: str) -> DataFrame:
    """Fetch DA prices from Energy-Charts."""
    print(f"  [DA prices] Energy-Charts: {start_date} → {end_date}")
    data = _ec_get("price", {
        "bzn": "DE-LU",
        "market": "day_ahead",
        "start": f"{start_date}T00:00Z",
        "end": f"{end_date}T23:59Z",
    }, "DA price")
    df = _ec_to_df(data, "da_price", "DA")
    df = _to_15min(df)
    print(f"    → {len(df):,} rows (15-min)")
    return df


def fetch_idc_ec(start_date: str, end_date: str) -> DataFrame:
    """Fetch IDC mid prices from Energy-Charts."""
    print(f"  [IDC mid] Energy-Charts: {start_date} → {end_date}")
    data = _ec_get("price", {
        "bzn": "DE-LU",
        "market": "intraday",
        "start": f"{start_date}T00:00Z",
        "end": f"{end_date}T23:59Z",
    }, "IDC mid")
    df = _ec_to_df(data, "idc_mid", "IDC")
    print(f"    → {len(df):,} rows (~44% coverage)")
    return df

def _entsoe_client(api_key: str):
    """Create ENTSO-E client."""
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        sys.exit("[ERROR] entsoe-py not installed. Run: pip install entsoe-py")
    return EntsoePandasClient(api_key=api_key)


def fetch_rebap(api_key: str, start_date: str, end_date: str) -> DataFrame:
    """Fetch German reBAP (imbalance price) from ENTSO-E."""
    try:
        from entsoe.exceptions import NoMatchingDataError
    except ImportError:
        NoMatchingDataError = Exception
    
    client = _entsoe_client(api_key)
    ts_start = Timestamp(start_date, tz=TIMEZONE)
    ts_end = Timestamp(end_date, tz=TIMEZONE) + Timedelta(days=1)
    
    print(f"  [reBAP] ENTSO-E (chunked): {start_date} → {end_date}")
    chunks: List[DataFrame] = []
    current = ts_start
    
    while current < ts_end:
        chunk_end = min(current + Timedelta(days=CHUNK_DAYS), ts_end)
        try:
            raw = client.query_imbalance_prices(
                country_code=DE_CTRL_AREA, start=current, end=chunk_end
            )
            pos_col = _find_col(raw, ["Positive", "Long", "A07"])
            pos_vals = raw[pos_col] if pos_col else raw.iloc[:, 0]
            dti = _to_dti(raw.index)
            chunk = DataFrame({"rebap": pos_vals.values}, index=dti)
            chunk.index.name = "timestamp"
            if not chunk.empty:
                chunks.append(chunk)
        except NoMatchingDataError:
            pass  # Expected for future dates
        except Exception as exc:
            print(f"    ⚠ {current.date()}: {exc}")
        current = chunk_end
    
    if not chunks:
        print("    → No reBAP data (will use DA proxy)")
        return DataFrame(dtype=float)
    
    combined = pd.concat(chunks)
    combined = combined[~combined.index.duplicated(keep="first")]
    print(f"    → {len(combined):,} rows")
    return combined.sort_index()


def fetch_wind_solar(api_key: str, start_date: str, end_date: str) -> Tuple[DataFrame, DataFrame]:
    """Fetch wind + solar forecasts from ENTSO-E."""
    client = _entsoe_client(api_key)
    ts_start = Timestamp(start_date, tz=TIMEZONE)
    ts_end = Timestamp(end_date, tz=TIMEZONE) + Timedelta(days=1)
    
    print(f"  [Wind+Solar] ENTSO-E: {start_date} → {end_date}")
    try:
        raw = client.query_wind_and_solar_forecast(
            country_code=DE_LU_CC, start=ts_start, end=ts_end, psr_type=None
        )
        
        # Extract series (prefer Day Ahead columns)
        def extract_series(keywords):
            if isinstance(raw, Series):
                return raw
            if isinstance(raw.columns, pd.MultiIndex):
                mask_da = [
                    any(kw.lower() in str(c[0]).lower() for kw in keywords)
                    and "day ahead" in str(c[1]).lower()
                    for c in raw.columns
                ]
                if any(mask_da):
                    return raw.loc[:, mask_da].sum(axis=1)
            mask = [any(kw.lower() in str(c).lower() for kw in keywords) for c in raw.columns]
            return raw.loc[:, mask].sum(axis=1) if any(mask) else raw.iloc[:, 0]
        
        wind_s = extract_series(["Wind"])
        solar_s = extract_series(["Solar"])
        dti = _to_dti(wind_s.index)
        
        wind_df = DataFrame({"wind_forecast_mw": wind_s.values}, index=dti)
        solar_df = DataFrame({"solar_forecast_mw": solar_s.values}, index=dti)
        wind_df.index.name = solar_df.index.name = "timestamp"
        
        print(f"    → {len(wind_df):,} rows")
        return wind_df, solar_df
    except Exception as exc:
        print(f"    ⚠ Error: {exc}")
        return DataFrame(dtype=float), DataFrame(dtype=float)


def fetch_load(api_key: str, start_date: str, end_date: str) -> DataFrame:
    """Fetch load forecast from ENTSO-E."""
    client = _entsoe_client(api_key)
    ts_start = Timestamp(start_date, tz=TIMEZONE)
    ts_end = Timestamp(end_date, tz=TIMEZONE) + Timedelta(days=1)
    
    print(f"  [Load] ENTSO-E: {start_date} → {end_date}")
    try:
        raw = client.query_load_forecast(country_code=DE_CC, start=ts_start, end=ts_end)
        series = raw.iloc[:, 0] if isinstance(raw, DataFrame) else raw
        dti = _to_dti(series.index)
        result = DataFrame({"load_forecast_mw": series.values}, index=dti)
        result.index.name = "timestamp"
        print(f"    → {len(result):,} rows")
        return result
    except Exception as exc:
        print(f"    ⚠ Error: {exc}")
        return DataFrame(dtype=float)

def fetch_fcr_from_regelleistung(date_from: str, date_to: str) -> DataFrame:

    print(f"  [FCR] regelleistung.net: {date_from} → {date_to}")
    
    url = "https://www.regelleistung.net/ext/data/tenders"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (research/bess-optimization)"
    }
    payload = {
        "exportFormat": "JSON",
        "dateFrom": date_from,
        "dateTo": date_to,
        "productTypes": ["FCR"],
        "regions": ["DE"],
    }
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        print(f"    ⚠ Cannot fetch FCR data: {e}")
        print("    → Will use synthetic FCR prices")
        return DataFrame()
    
    # Parse response
    records = []
    items = raw if isinstance(raw, list) else raw.get("tenders", [])
    
    for item in items:
        try:
            tender_date = pd.to_datetime(item.get("tenderDate") or item.get("date"))
            for block_info in item.get("blocks", [item]):
                block_start = int(block_info.get("deliveryFrom", "00:00").split(":")[0])
                block_idx = BLOCK_STARTS.index(block_start) if block_start in BLOCK_STARTS else -1
                if block_idx < 0:
                    continue
                clearing = float(block_info.get("clearingPrice", 0) or
                               block_info.get("settlementCapacityPrice", 0) or 0)
                records.append({
                    "date": tender_date.date(),
                    "block": block_idx,
                    "block_start_h": block_start,
                    "fcr_price": clearing,
                })
        except Exception:
            continue
    
    if not records:
        print("    → No FCR records found, using synthetic")
        return DataFrame()
    
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "block"]).reset_index(drop=True)
    print(f"    → {len(df):,} FCR block records")
    return df

print("[INFO] Data collection functions loaded")
print("[INFO] FCR capacity from regelleistung.net (if available)")
print("[INFO] aFRR capacity: synthetic (calibrated to 2024 market)")
print("[INFO] aFRR activation: REBAP from ENTSO-E")

def collect_all_data(api_key: str, start_date: str, end_date: str, for_training: bool = True) -> DataFrame:

    print(f"\n{'='*70}")
    print(f"COLLECTING DATA: {start_date} → {end_date}")
    print(f"{'='*70}\n")
    
    # Fetch all data
    da_df = fetch_da_ec(start_date, end_date)
    idc_df = fetch_idc_ec(start_date, end_date) if for_training else DataFrame()
    rebap_df = fetch_rebap(api_key, start_date, end_date)
    wind_df, solar_df = fetch_wind_solar(api_key, start_date, end_date)
    load_df = fetch_load(api_key, start_date, end_date)
    
    # Assembly
    if for_training:
        # Training: start with IDC index
        df = idc_df.copy()
        for extra in (da_df, rebap_df, wind_df, solar_df, load_df):
            if extra is not None and not extra.empty:
                df = df.join(extra, how="left")
    else:
        # Forecast: use full 15-min grid
        idx = _make_full_index(start_date, end_date)
        df = DataFrame(index=idx)
        for extra in (da_df, rebap_df, wind_df, solar_df, load_df):
            if extra is not None and not extra.empty:
                df = df.join(extra, how="left")
    
    # Ensure timezone
    dti = _to_dti(df.index)
    if dti.tz is None:
        dti = dti.tz_localize(TIMEZONE)
    df.index = dti
    
    # Resample to uniform 15-min (for training only)
    if for_training:
        idc_15 = df[["idc_mid"]].resample("15min").first() if "idc_mid" in df.columns else DataFrame()
        feat_cols = [c for c in df.columns if c != "idc_mid"]
        feat_15 = df[feat_cols].resample("15min").ffill() if feat_cols else DataFrame()
        df = feat_15.join(idc_15, how="left") if not idc_15.empty else feat_15
    
    print(f"\n✓ Data collection complete: {len(df):,} rows\n")
    return df


def engineer_features(df: DataFrame) -> DataFrame:
    """Build complete feature set."""
    out = df.copy().sort_index()
    dti = _to_dti(out.index)
    
    # Time features
    out["hour_of_day"] = dti.hour
    out["quarter_hour"] = dti.minute // 15
    out["day_of_week"] = dti.dayofweek
    out["month"] = dti.month
    out["is_weekend"] = (dti.dayofweek >= 5).astype(int)
    
    # Derived features
    if "da_price" in out.columns and "rebap" in out.columns:
        out["spread_vs_da"] = out["rebap"] - out["da_price"]
    
    if "wind_forecast_mw" in out.columns and "solar_forecast_mw" in out.columns:
        out["total_res_mw"] = (out["wind_forecast_mw"].fillna(0) +
                              out["solar_forecast_mw"].fillna(0))
    
    if "load_forecast_mw" in out.columns and "total_res_mw" in out.columns:
        out["residual_load"] = (out["load_forecast_mw"].fillna(0) -
                               out["total_res_mw"].fillna(0))
    
    # Lags
    if "rebap" in out.columns:
        out["rebap_lag1"] = out["rebap"].shift(1)
        out["rebap_lag4"] = out["rebap"].shift(4)
        out["rebap_lag96"] = out["rebap"].shift(96)
    
    return out


print("[INFO] Assembly & feature engineering functions loaded\n")

def _strip_leap_day(df: DataFrame) -> DataFrame:
    """Remove Feb 29 to keep DOY consistent across years."""
    dti = _to_dti(df.index)
    mask_leap = (dti.month == 2) & (dti.day == 29)
    return df[~mask_leap]


def build_da_profile(hist_raw: dict, lambda_ew: float = LAMBDA_EW, 
                    use_equal_weights: bool = USE_EQUAL_WEIGHTS,
                    spread_factor: float = SPREAD_ENHANCEMENT) -> Series:

    n = len(HIST_YEARS)
    
    if use_equal_weights:
        # Equal weighting for unbiased median
        w = np.ones(n) / n
        print(f"  Using EQUAL weights (true median): " + 
              "  ".join(f"{yr}={wt:.3f}" for yr, wt in zip(HIST_YEARS, w)))
    else:
        # Exponential weighting (biased toward recent years)
        raw_w = np.array([lambda_ew ** (n - 1 - i) for i in range(n)])
        w = raw_w / raw_w.sum()
        print(f"  Using EXPONENTIAL weights (λ={lambda_ew}): " + 
              "  ".join(f"{yr}={wt:.3f}" for yr, wt in zip(HIST_YEARS, w)))
    
    frames: List[DataFrame] = []
    for i, (yr, df_yr) in enumerate(hist_raw.items()):
        col = f"da_price_{yr}"
        tmp = df_yr[[col]].copy().rename(columns={col: "da_price"})
        tmp = _strip_leap_day(tmp)
        dti = _to_dti(tmp.index)
        tmp["doy"] = dti.dayofyear
        tmp["hour"] = dti.hour
        tmp["weight"] = w[i]
        frames.append(tmp)
    
    combined = pd.concat(frames)
    profile = (
        combined
        .groupby(["doy", "hour"])
        .apply(lambda g: np.average(g["da_price"].values, weights=g["weight"].values),
               include_groups=False)
        .rename("da_price_mean")
    )
    
    # Apply spread enhancement if requested
    if spread_factor != 1.0:
        print(f"  Applying spread enhancement factor: {spread_factor}")
        profile = enhance_profile_spreads(profile, spread_factor)
    
    print(f"  Profile cells: {len(profile):,}  |  NaN: {profile.isna().sum()}")
    return profile


def enhance_profile_spreads(profile: Series, factor: float = 1.15) -> Series:

    df = profile.reset_index()
    
    for doy in range(1, 366):
        mask = df['doy'] == doy
        if not mask.any():
            continue
        
        day_prices = df.loc[mask, 'da_price_mean'].values
        if len(day_prices) < 24:
            continue
        
        day_mean = day_prices.mean()
        
        # Enhance spread around daily mean
        enhanced = day_mean + (day_prices - day_mean) * factor
        df.loc[mask, 'da_price_mean'] = enhanced
    
    return df.set_index(['doy', 'hour'])['da_price_mean']


def forecast_da_prices(profile: Series, forecast_index: DatetimeIndex,
                      scenario_mult: float = DA_SCENARIO) -> Series:
    """Generate DA forecast from profile."""
    dti = _to_dti(forecast_index)
    fc_vals = np.array([
        profile.get((int(ts.dayofyear), int(ts.hour)), np.nan)
        for ts in dti
    ])
    fc_vals = fc_vals * scenario_mult
    
    s = Series(fc_vals, index=forecast_index, name="da_price_forecast")
    if s.isna().any():
        s = s.ffill().bfill()
    
    print(f"\n  DA Forecast (scenario ×{scenario_mult:.2f}):")
    print(f"    Mean: {s.mean():.2f} EUR/MWh")
    print(f"    Min: {s.min():.2f} | Max: {s.max():.2f}")
    return s


FEATURE_COLS = [
    "da_price", "rebap", "rebap_lag1", "rebap_lag4", "rebap_lag96",
    "spread_vs_da", "wind_forecast_mw", "solar_forecast_mw",
    "total_res_mw", "residual_load", "load_forecast_mw",
    "hour_of_day", "quarter_hour", "day_of_week", "month", "is_weekend",
]
TARGET_COL = "idc_mid"


def train_idc_model(df: DataFrame):
    """Train XGBoost model for IDC mid price."""
    try:
        import xgboost as xgb
    except ImportError:
        sys.exit("[ERROR] xgboost not installed. Run: pip install xgboost")
    
    print(f"\n{'='*70}")
    print("TRAINING IDC MODEL")
    print(f"{'='*70}\n")
    
    # Prepare training data
    available = [c for c in FEATURE_COLS if c in df.columns]
    df_train = df.dropna(subset=[TARGET_COL])
    
    X = df_train[available]
    y = df_train[TARGET_COL]
    
    # Compute training medians for imputation
    train_medians = {col: X[col].median() for col in available}
    
    # Handle NaN
    X_clean = X.copy()
    for col in available:
        if X_clean[col].isna().any():
            X_clean[col] = X_clean[col].fillna(train_medians[col])
    
    print(f"  Training samples: {len(X_clean):,}")
    print(f"  Features: {len(available)}")
    
    # Train model
    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_clean, y)
    
    print(f"  ✓ Model trained")
    return model, available, train_medians


def predict_idc(model, feature_cols, train_medians, df_features, da_forecast) -> DataFrame:

    available = [c for c in feature_cols if c in df_features.columns]
    df_pred = df_features[available].copy()
    
    # Inject DA forecast
    if "da_price" in df_pred.columns and da_forecast is not None:
        aligned = da_forecast.reindex(df_pred.index)
        df_pred["da_price"] = aligned.fillna(df_pred["da_price"])
    
    # Impute NaN
    for col in available:
        if df_pred[col].isna().any():
            df_pred[col] = df_pred[col].fillna(train_medians.get(col, 0.0))
    
    # Predict base IDC mid
    mid_base = Series(model.predict(df_pred.values), index=df_pred.index)
    
    # ENHANCEMENT: Ensure IDC spreads are at least 2x DA spreads
    # Market reality: Germany IDC 12-mo spread 231 EUR/MWh vs DA 124 EUR/MWh (~1.9x)
    # XGBoost model learns compressed relationship, so we enhance spreads post-prediction
    da_prices = da_forecast.reindex(df_pred.index)
    
    # Calculate daily DA spread for each IDC period
    da_daily = da_prices.resample('D')
    da_spread = (da_daily.max() - da_daily.min()).reindex(mid_base.index, method='ffill')
    
    # Calculate daily IDC spread from base prediction
    mid_daily = mid_base.resample('D')
    idc_spread_base = (mid_daily.max() - mid_daily.min()).reindex(mid_base.index, method='ffill')
    
    # Enhance IDC spread if it's less than 2x DA spread
    min_idc_spread = da_spread * 2.0  # Target: IDC spread ≥ 2x DA spread
    spread_ratio = (idc_spread_base / min_idc_spread).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    # Apply enhancement only when needed (don't reduce existing high spreads)
    enhancement_factor = np.where(spread_ratio < 1.0, 1.0 / spread_ratio, 1.0)
    
    # Get daily mean and enhance around it
    mid_daily_mean = mid_base.resample('D').mean().reindex(mid_base.index, method='ffill')
    mid = mid_daily_mean + (mid_base - mid_daily_mean) * enhancement_factor
    
    # Bid/ask spread
    h = BID_ASK_SPREAD / 2
    bid = np.minimum(mid * (1.0 - h), mid * (1.0 + h))
    ask = np.maximum(mid * (1.0 - h), mid * (1.0 + h))
    
    print(f"  IDC spread enhancement applied:")
    print(f"    Mean enhancement factor: {enhancement_factor.mean():.3f}")
    print(f"    Periods enhanced: {(enhancement_factor > 1.0).sum():,} / {len(enhancement_factor):,}")
    
    return DataFrame({
        "da_price_forecast": da_prices,
        "idc_mid_forecast": mid,
        "idc_bid": bid,
        "idc_ask": ask,
    }, index=df_features.index)


def forecast_fcr_2026(fcr_hist: Optional[DataFrame], features_2026: DataFrame) -> DataFrame:

    print(f"\n{'='*70}")
    print("FCR FORECAST")
    print(f"{'='*70}\n")
    
    if fcr_hist is None or len(fcr_hist) == 0:
        print("  Using synthetic FCR prices (calibrated to 2024 market)")
        # Base synthetic model
        idx = features_2026.index
        hours = idx.hour
        blocks = hours // 4
        months = idx.month
        
        # Base price: 25 EUR/MW/4h (2026 estimate)
        base = 18.0
        
        # Block factors (solar bathtub)
        block_map = {0: 0.79, 1: 0.64, 2: 1.25, 3: 1.43, 4: 1.00, 5: 0.71}
        block_factors = np.array([block_map[b] for b in blocks])
        
        # Month factors (seasonal)
        month_map = {1: 0.90, 2: 0.82, 3: 0.92, 4: 1.10, 5: 1.30, 6: 1.42,
                    7: 1.35, 8: 1.22, 9: 1.12, 10: 0.90, 11: 0.82, 12: 0.88}
        month_factors = np.array([month_map[m] for m in months])
        
        # Cannibalization (quarterly)
        # UPDATED: Less aggressive than original (floor 30% vs 10%, slope -25% vs -40%)
        # Market shows FCR still earning ~16 EUR/MW/4h despite saturation
        # Old: max(0.10, 1 - 0.40 × max(0, overbuild - 1.0))
        # New: max(0.30, 1 - 0.25 × max(0, overbuild - 1.0))
        quarters = (months - 1) // 3
        overbuild_map = {0: 1.65, 1: 1.80, 2: 1.95, 3: 2.05}  # Q1-Q4 2026
        discount_map = {q: max(0.20, 1 - 0.35 * max(0, ob - 1.0)) # Stronger cannibalization
                       for q, ob in overbuild_map.items()}
        discounts = np.array([discount_map[q] for q in quarters])
        
        fcr_p50 = base * block_factors * month_factors * discounts
        fcr_p10 = fcr_p50 * 0.7
        fcr_p90 = fcr_p50 * 1.3
        
    else:
        print(f"  Using historical FCR data: {len(fcr_hist):,} records")
        # Use actual historical data - simplified model
        # Group by block and month, take mean
        fcr_hist_copy = fcr_hist.copy()
        fcr_hist_copy["month"] = pd.to_datetime(fcr_hist_copy["date"]).dt.month
        block_month_avg = fcr_hist_copy.groupby(["block", "month"])["fcr_price"].mean()
        
        idx = features_2026.index
        hours = idx.hour
        months = idx.month
        blocks = hours // 4
        
        fcr_p50 = np.array([
            block_month_avg.get((b, m), 25.0)
            for b, m in zip(blocks, months)
        ])
        fcr_p10 = fcr_p50 * 0.7
        fcr_p90 = fcr_p50 * 1.3
    
    result = DataFrame({
        "fcr_p50_block": fcr_p50,
        "fcr_p10_block": fcr_p10,
        "fcr_p90_block": fcr_p90,
    }, index=features_2026.index)
    
    print(f"  FCR P50: Mean={fcr_p50.mean():.2f} EUR/MW/4h")
    return result


def forecast_afrr_2026(features_2026: DataFrame, rebap_hist: DataFrame) -> DataFrame:

    print(f"\n{'='*70}")
    print("aFRR FORECAST")
    print(f"{'='*70}\n")
    
    idx = features_2026.index
    hours = idx.hour
    months = idx.month

    base_pos = 20.0  # Up from 8.3 (2.4x increase)
    base_neg = 13.5  # Up from 6.8 (2.0x increase)
    
    # Hour factors
    hour_map = {
        0: 0.75, 1: 0.70, 2: 0.68, 3: 0.70, 4: 0.75, 5: 0.80,
        6: 0.88, 7: 1.05, 8: 1.15, 9: 1.20, 10: 1.18, 11: 1.12,
        12: 1.10, 13: 1.08, 14: 1.05, 15: 1.10, 16: 1.20, 17: 1.28,
        18: 1.25, 19: 1.15, 20: 1.05, 21: 0.95, 22: 0.85, 23: 0.78,
    }
    hour_factors = np.array([hour_map[h] for h in hours])
    
    # Month factors
    month_map = {1: 1.10, 2: 1.05, 3: 0.95, 4: 0.90, 5: 0.95, 6: 1.05,
                 7: 1.10, 8: 1.05, 9: 0.95, 10: 0.90, 11: 1.00, 12: 1.08}
    month_factors = np.array([month_map[m] for m in months])
    
    # Cannibalization
    quarters = (months - 1) // 3
    overbuild_map = {0: 0.75, 1: 0.86, 2: 1.00, 3: 1.12}  # aFRR less saturated
    discount_map = {q: max(0.25, 1 - 0.30 * max(0, ob - 1.0))
                   for q, ob in overbuild_map.items()}
    discounts = np.array([discount_map[q] for q in quarters])
    
    cap_pos_p50 = base_pos * hour_factors * month_factors * discounts
    cap_neg_p50 = base_neg * hour_factors * month_factors * discounts
    
    # ── REBAP (activation price) - empirical quantiles ──────────────────────
    if rebap_hist is not None and len(rebap_hist) > 100:
        print("  Using REBAP quantiles from historical data")
        rebap_hist_copy = rebap_hist.copy()
        dti = _to_dti(rebap_hist_copy.index)
        rebap_hist_copy["month"] = dti.month
        rebap_hist_copy["hour"] = dti.hour
        
        quantiles = rebap_hist_copy.groupby(["month", "hour"])["rebap"].quantile([0.1, 0.5, 0.9]).unstack()
        quantiles.columns = ["rebap_p10", "rebap_p50", "rebap_p90"]
        
        rebap_p10 = np.array([quantiles.loc[(m, h), "rebap_p10"] if (m, h) in quantiles.index else 40.0
                             for m, h in zip(months, hours)])
        rebap_p50 = np.array([quantiles.loc[(m, h), "rebap_p50"] if (m, h) in quantiles.index else 90.0
                             for m, h in zip(months, hours)])
        rebap_p90 = np.array([quantiles.loc[(m, h), "rebap_p90"] if (m, h) in quantiles.index else 140.0
                             for m, h in zip(months, hours)])
        
        prob_pos = np.array([
            (rebap_hist_copy[(rebap_hist_copy["month"] == m) & (rebap_hist_copy["hour"] == h)]["rebap"] > 0).mean()
            if len(rebap_hist_copy[(rebap_hist_copy["month"] == m) & (rebap_hist_copy["hour"] == h)]) > 0
            else 0.667
            for m, h in zip(months, hours)
        ])
    else:
        print("  Using default REBAP quantiles (insufficient history)")
        rebap_p10 = np.full(len(idx), 40.0)
        rebap_p50 = np.full(len(idx), 90.0)
        rebap_p90 = np.full(len(idx), 140.0)
        prob_pos = np.full(len(idx), 0.667)
    
    result = DataFrame({
        "afrr_cap_pos_p50": cap_pos_p50,
        "afrr_cap_pos_p10": cap_pos_p50 * 0.7,
        "afrr_cap_pos_p90": cap_pos_p50 * 1.3,
        "afrr_cap_neg_p50": cap_neg_p50,
        "afrr_cap_neg_p10": cap_neg_p50 * 0.7,
        "afrr_cap_neg_p90": cap_neg_p50 * 1.3,
        "rebap_p50": rebap_p50,
        "rebap_p10": rebap_p10,
        "rebap_p90": rebap_p90,
        "rebap_prob_positive": prob_pos,
    }, index=features_2026.index)
    
    print(f"  aFRR Capacity P50 (pos): Mean={cap_pos_p50.mean():.2f} EUR/MW/h")
    print(f"  REBAP P50: Mean={rebap_p50.mean():.2f} EUR/MWh")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT TO EXCEL
# ═════════════════════════════════════════════════════════════════════════════

def write_excel_output(df_combined: DataFrame, output_path: str):
    """Write combined forecast to Excel with multiple sheets."""
    print(f"\n{'='*70}")
    print("WRITING EXCEL OUTPUT")
    print(f"{'='*70}\n")
    
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"  Removed existing: {output_path}")
    
    # Convert timezone-aware index to naive for Excel compatibility
    df_excel = df_combined.copy()
    if df_excel.index.tz is not None:
        df_excel.index = df_excel.index.tz_localize(None)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: DA & IDC
        df_da_idc = df_excel[['da_price_forecast', 'idc_mid_forecast', 'idc_bid', 'idc_ask']].copy()
        df_da_idc.to_excel(writer, sheet_name='DA_IDC_Forecast')
        print("  ✓ Sheet 1: DA_IDC_Forecast")
        
        # Sheet 2: FCR
        fcr_cols = [c for c in df_excel.columns if c.startswith('fcr_')]
        if fcr_cols:
            df_fcr = df_excel[fcr_cols].copy()
            df_fcr.to_excel(writer, sheet_name='FCR_Forecast')
            print("  ✓ Sheet 2: FCR_Forecast")
        
        # Sheet 3: aFRR
        afrr_cols = [c for c in df_excel.columns if 'afrr' in c or 'rebap' in c]
        if afrr_cols:
            df_afrr = df_excel[afrr_cols].copy()
            df_afrr.to_excel(writer, sheet_name='aFRR_Forecast')
            print("  ✓ Sheet 3: aFRR_Forecast")
        
        # Sheet 4: All combined
        df_excel.to_excel(writer, sheet_name='All_Forecasts')
        print("  ✓ Sheet 4: All_Forecasts")
        
        # Sheet 5: Summary
        summary = []
        for col in df_excel.columns:
            s = df_excel[col].dropna()
            if len(s) > 0:
                summary.append({
                    'Variable': col,
                    'Count': len(s),
                    'Mean': s.mean(),
                    'Std': s.std(),
                    'Min': s.min(),
                    'P50': s.median(),
                    'Max': s.max(),
                })
        df_summary = DataFrame(summary)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        print("  ✓ Sheet 5: Summary")
    
    print(f"\n✓ Excel file written: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB\n")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Germany Electricity Price Forecast 2026 - Integrated Pipeline"
    )
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                       help='Output Excel file path')
    args = parser.parse_args()


    # Check API key
    api_key = "c510a984-1c78-46b8-a142-4063a47761d5"
    if not api_key:
        print("[ERROR] ENTSO-E API key required!")
        print("  Set: export ENTSOE_API_KEY=your_token")
        print("  Or hardcode in script (not recommended)")
        sys.exit(1)
    
    print(f"Output: {args.output}")
    print(f"Training: {TRAIN_START} → {TRAIN_END}")
    print(f"Forecast: {FORECAST_START} → {FORECAST_END}")

    df_raw_train = collect_all_data(api_key, TRAIN_START, TRAIN_END, for_training=True)

    print(f"\n{'='*70}")
    print("BUILDING DA PROFILE")
    print(f"{'='*70}\n")
    
    hist_raw = {}
    for yr in HIST_YEARS:
        col = f"da_price_{yr}"
        sub = df_raw_train.loc[df_raw_train.index.year == yr, ["da_price"]].copy()
        sub = sub.rename(columns={"da_price": col})
        sub = sub[~sub.index.duplicated(keep="first")].sort_index()
        sub = _to_15min(sub)
        hist_raw[yr] = sub
        print(f"  {yr}: {len(sub):,} rows")
    
    profile = build_da_profile(hist_raw, lambda_ew=LAMBDA_EW, 
                              use_equal_weights=USE_EQUAL_WEIGHTS,
                              spread_factor=SPREAD_ENHANCEMENT)

    print(f"\n{'='*70}")
    print("FEATURE ENGINEERING (TRAINING)")
    print(f"{'='*70}\n")
    
    df_train = engineer_features(df_raw_train)
    print(f"  Features: {len(df_train.columns)} columns, {len(df_train):,} rows")

    model, feat_cols, train_medians = train_idc_model(df_train)

    df_raw_2026 = collect_all_data(api_key, FORECAST_START, FORECAST_END, for_training=False)
    
    # Store REBAP for aFRR activation forecast
    rebap_hist = df_raw_train[["rebap"]].dropna() if "rebap" in df_raw_train.columns else DataFrame()

    print(f"\n{'='*70}")
    print("GENERATING DA FORECAST")
    print(f"{'='*70}\n")
    
    forecast_idx = _make_full_index(FORECAST_START, FORECAST_END)
    da_forecast = forecast_da_prices(profile, forecast_idx, scenario_mult=DA_SCENARIO)

    print(f"\n{'='*70}")
    print("FEATURE ENGINEERING (2026)")
    print(f"{'='*70}\n")
    
    df_features_2026 = engineer_features(df_raw_2026)
    print(f"  Features: {len(df_features_2026.columns)} columns")

    print(f"\n{'='*70}")
    print("PREDICTING IDC PRICES")
    print(f"{'='*70}\n")
    
    df_da_idc = predict_idc(model, feat_cols, train_medians, df_features_2026, da_forecast)
    print(f"  IDC Mid P50: Mean={df_da_idc['idc_mid_forecast'].mean():.2f} EUR/MWh")

    fcr_hist = fetch_fcr_from_regelleistung(TRAIN_START, TRAIN_END)
    df_fcr = forecast_fcr_2026(fcr_hist, df_features_2026)

    df_afrr = forecast_afrr_2026(df_features_2026, rebap_hist)

    print(f"\n{'='*70}")
    print("COMBINING ALL FORECASTS")
    print(f"{'='*70}\n")
    
    df_combined = pd.concat([df_da_idc, df_fcr, df_afrr], axis=1)
    df_combined.index.name = 'timestamp'
    
    print(f"  Combined: {len(df_combined):,} rows × {len(df_combined.columns)} columns")
    print(f"  Columns: {list(df_combined.columns)[:5]}...")

    write_excel_output(df_combined, args.output)

    print(f"\n{'='*70}")
    print("✓ FORECAST COMPLETE")
    print(f"{'='*70}\n")
    print(f"Output: {args.output}")
    print(f"Rows: {len(df_combined):,} (15-min intervals, full year 2026)")
    print(f"\nMarket Summaries:")
    print(f"  DA:   Mean={df_combined['da_price_forecast'].mean():.2f} EUR/MWh")
    print(f"  IDC:  Mean={df_combined['idc_mid_forecast'].mean():.2f} EUR/MWh")
    print(f"  FCR:  Mean={df_combined['fcr_p50_block'].mean():.2f} EUR/MW/4h")
    print(f"  aFRR: Mean={df_combined['afrr_cap_pos_p50'].mean():.2f} EUR/MW/h (capacity)")
    print(f"        Mean={df_combined['rebap_p50'].mean():.2f} EUR/MWh (activation)")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
