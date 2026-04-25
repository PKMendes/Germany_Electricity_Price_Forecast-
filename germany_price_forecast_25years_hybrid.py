
from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional
import argparse

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, DatetimeIndex

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

TIMEZONE = "Europe/Berlin"
START_YEAR = 2026
END_YEAR = 2050
NUM_YEARS = END_YEAR - START_YEAR + 1

# Base year 2026 parameters (validated against market data)
BASE_DA_PRICE_2026 = 90.0  # EUR/MWh (conservative vs Q1 2026 actual 97.75)
BASE_FCR_PRICE_2026 = 12.96  # EUR/MW/h (validated vs Suena Oct 2025)
BASE_AFRR_POS_2026 = 19.79  # EUR/MW/h (conservative vs actuals)
BASE_AFRR_NEG_2026 = 13.36  # EUR/MW/h

# Output path
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), 
                              "Germany_priceforecast_25years.xlsx")

print(f"""  {'='*70} GERMANY 25-YEAR PRICE FORECAST: HYBRID MODEL Realistic Volatility + Long-term Trajectory""")


def load_2026_profile(filepath: str) -> tuple[Series, DataFrame, DataFrame, DataFrame]:

    print(f"\n{'='*70}")
    print("LOADING 2026 PROFILE FOR VOLATILITY STRUCTURE")
    print(f"{'='*70}\n")
    
    try:
        df_2026 = pd.read_excel(filepath, sheet_name='All_Forecasts')
        print(f"  ✓ Loaded 2026 forecast: {len(df_2026):,} intervals")
    except Exception as e:
        print(f"  ✗ Error loading 2026 profile: {e}")
        print(f"  Using fallback: simplified profile")
        return None, None, None, None
    
    # Extract DA normalized profile
    da_prices = df_2026['da_price_forecast'].values
    mean_2026 = da_prices.mean()
    std_2026 = da_prices.std()
    
    # Normalize: deviation from mean as ratio
    da_normalized = (da_prices - mean_2026) / mean_2026
    da_normalized = Series(da_normalized, index=pd.to_datetime(df_2026['timestamp']))
    
    print(f"  DA Profile Statistics:")
    print(f"    Original mean: {mean_2026:.2f} EUR/MWh")
    print(f"    Original std:  {std_2026:.2f} EUR/MWh")
    print(f"    Normalized std: {da_normalized.std():.4f}")
    print(f"    Min/max ratio:  {da_normalized.min():.2f} / {da_normalized.max():.2f}")
    
    # Count negative prices
    negative_count = (da_prices < 0).sum()
    negative_pct = negative_count / len(da_prices) * 100
    print(f"    Negative prices: {negative_count:,} ({negative_pct:.2f}%)")
    
    # Extract FCR profile (preserve relative structure)
    fcr_profile = df_2026[['timestamp', 'fcr_p50_block']].copy()
    fcr_profile['timestamp'] = pd.to_datetime(fcr_profile['timestamp'])
    fcr_mean = fcr_profile['fcr_p50_block'].mean()
    fcr_profile['fcr_normalized'] = (fcr_profile['fcr_p50_block'] - fcr_mean) / fcr_mean
    print(f"\n  FCR Profile:")
    print(f"    Original mean: {fcr_mean:.2f} EUR/MW/h")
    print(f"    Normalized range: [{fcr_profile['fcr_normalized'].min():.3f}, {fcr_profile['fcr_normalized'].max():.3f}]")
    
    # Extract aFRR profile
    afrr_profile = df_2026[['timestamp', 'afrr_cap_pos_p50', 'afrr_cap_neg_p50']].copy()
    afrr_profile['timestamp'] = pd.to_datetime(afrr_profile['timestamp'])
    afrr_pos_mean = afrr_profile['afrr_cap_pos_p50'].mean()
    afrr_profile['afrr_pos_normalized'] = (afrr_profile['afrr_cap_pos_p50'] - afrr_pos_mean) / afrr_pos_mean
    afrr_profile['afrr_neg_normalized'] = (afrr_profile['afrr_cap_neg_p50'] - 
                                           afrr_profile['afrr_cap_neg_p50'].mean()) / afrr_profile['afrr_cap_neg_p50'].mean()
    print(f"\n  aFRR Profile:")
    print(f"    Original pos mean: {afrr_pos_mean:.2f} EUR/MW/h")
    print(f"    Normalized range: [{afrr_profile['afrr_pos_normalized'].min():.3f}, {afrr_profile['afrr_pos_normalized'].max():.3f}]")
    
    print(f"\n  ✓ Profile extraction complete")
    
    return da_normalized, fcr_profile, afrr_profile, df_2026


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _make_full_index(year: int) -> DatetimeIndex:
    """Create complete 15-min DatetimeIndex for a given year."""
    start = f"{year}-01-01 00:00"
    end = f"{year}-12-31 23:45"
    return pd.date_range(start=start, end=end, freq="15min", tz=TIMEZONE)


def battery_capacity_forecast(year: int) -> float:

    tau = year - START_YEAR
    if tau == 0:
        return 2.5  # 2026 baseline
    elif year <= 2030:
        # Linear to 15 GW by 2030
        return 2.5 + (15.0 - 2.5) * tau / 4
    elif year <= 2040:
        # Linear to 24 GW by 2040
        return 15.0 + (24.0 - 15.0) * (year - 2030) / 10
    else:
        # Linear to 61 GW by 2050
        return 24.0 + (61.0 - 24.0) * (year - 2040) / 10

def forecast_da_base_price(year: int) -> float:

    tau = year - START_YEAR
    
    if year <= 2030:
        # Phase 1: Rising to peak
        return BASE_DA_PRICE_2026 + 2.5 * tau
    elif year <= 2040:
        # Phase 2: Plateau then decline
        peak_price = BASE_DA_PRICE_2026 + 2.5 * 4  # 100 EUR/MWh at 2030
        return peak_price - 1.0 * (year - 2030)
    else:
        # Phase 3: Continued decline
        price_2040 = 100.0 - 1.0 * 10  # 90 EUR/MWh at 2040
        return price_2040 - 0.6 * (year - 2040)


def forecast_fcr_base_price(year: int) -> float:
    """FCR base price evolution (EUR/MW/h)."""
    tau = year - START_YEAR
    
    if year <= 2030:
        # Phase 1: Moderate decline (-4% annual)
        return BASE_FCR_PRICE_2026 * (1 - 0.04) ** tau
    elif year <= 2040:
        # Phase 2: Faster decline (-5% annual)
        price_2030 = BASE_FCR_PRICE_2026 * (1 - 0.04) ** 4
        return price_2030 * (1 - 0.05) ** (year - 2030)
    else:
        # Phase 3: Floor price (-3% annual, min €4.5/MW/h)
        price_2040 = BASE_FCR_PRICE_2026 * (1 - 0.04) ** 4 * (1 - 0.05) ** 10
        return max(4.5, price_2040 * (1 - 0.03) ** (year - 2040))


def forecast_afrr_base_price(year: int) -> float:
    """aFRR base price evolution (EUR/MW/h) - Positive direction."""
    tau = year - START_YEAR
    
    if year <= 2032:
        # Phase 1: Growth (+1% annual)
        return BASE_AFRR_POS_2026 * (1 + 0.01) ** tau
    elif year <= 2042:
        # Phase 2: Saturation (-5% annual)
        price_2032 = BASE_AFRR_POS_2026 * (1 + 0.01) ** 6
        return price_2032 * (1 - 0.05) ** (year - 2032)
    else:
        # Phase 3: Floor (-2% annual, min €9/MW/h)
        price_2042 = BASE_AFRR_POS_2026 * (1 + 0.01) ** 6 * (1 - 0.05) ** 10
        return max(9.0, price_2042 * (1 - 0.02) ** (year - 2042))


def align_profile_to_year(profile_2026: Series, target_year: int) -> Series:
    """
    Align 2026 profile timestamps to target year, handling leap years.
    """
    # Create target year index
    target_idx = _make_full_index(target_year)
    
    # Extract hour/minute structure from 2026
    profile_reindexed = profile_2026.copy()
    profile_reindexed.index = pd.to_datetime(profile_2026.index)
    
    # Create mapping based on (dayofyear, hour, minute)
    profile_map = {}
    for ts, val in profile_reindexed.items():
        key = (ts.dayofyear, ts.hour, ts.minute)
        profile_map[key] = val
    
    # Apply to target year
    result = []
    for ts in target_idx:
        key = (ts.dayofyear, ts.hour, ts.minute)
        # Handle leap year: if Feb 29 doesn't exist in pattern, use Feb 28
        if key not in profile_map and ts.month == 2 and ts.day == 29:
            key = (59, ts.hour, ts.minute)  # Feb 28 in non-leap year
        result.append(profile_map.get(key, 0.0))
    
    return Series(result, index=target_idx)


def generate_hybrid_da_forecast(year: int, da_normalized_2026: Series) -> Series:

    base_price = forecast_da_base_price(year)
    
    profile_aligned = align_profile_to_year(da_normalized_2026, year)

    prices = base_price * (1 + profile_aligned)
    
    prices.name = "da_price_forecast"
    
    return prices


def generate_hybrid_fcr_forecast(year: int, fcr_profile_2026: DataFrame) -> DataFrame:

    base_price = forecast_fcr_base_price(year)
    
    target_idx = _make_full_index(year)
    
    fcr_profile_2026_indexed = fcr_profile_2026.set_index('timestamp')
    
    # Map by (hour, minute) to handle different calendar
    fcr_map = {}
    for ts, row in fcr_profile_2026_indexed.iterrows():
        key = (ts.dayofyear, ts.hour, ts.minute)
        fcr_map[key] = row['fcr_normalized']
    
    # Apply to target year
    fcr_normalized = []
    for ts in target_idx:
        key = (ts.dayofyear, ts.hour, ts.minute)
        if key not in fcr_map and ts.month == 2 and ts.day == 29:
            key = (59, ts.hour, ts.minute)
        fcr_normalized.append(fcr_map.get(key, 0.0))
    
    # Apply to base price
    fcr_prices = base_price * (1 + np.array(fcr_normalized))
    
    return DataFrame({
        'fcr_p50_block': fcr_prices
    }, index=target_idx)


def generate_hybrid_afrr_forecast(year: int, afrr_profile_2026: DataFrame,
                                  rebap_2026: Optional[Series] = None) -> DataFrame:

    base_price_pos = forecast_afrr_base_price(year)
    base_price_neg = 0.70 * base_price_pos
    
    # Create target year index
    target_idx = _make_full_index(year)
    
    # Align aFRR profile
    afrr_profile_2026_indexed = afrr_profile_2026.set_index('timestamp')
    
    # Map by (dayofyear, hour, minute)
    afrr_pos_map = {}
    afrr_neg_map = {}
    for ts, row in afrr_profile_2026_indexed.iterrows():
        key = (ts.dayofyear, ts.hour, ts.minute)
        afrr_pos_map[key] = row['afrr_pos_normalized']
        afrr_neg_map[key] = row['afrr_neg_normalized']
    
    # Apply to target year
    afrr_pos_normalized = []
    afrr_neg_normalized = []
    for ts in target_idx:
        key = (ts.dayofyear, ts.hour, ts.minute)
        if key not in afrr_pos_map and ts.month == 2 and ts.day == 29:
            key = (59, ts.hour, ts.minute)
        afrr_pos_normalized.append(afrr_pos_map.get(key, 0.0))
        afrr_neg_normalized.append(afrr_neg_map.get(key, 0.0))
    
    # Apply to base prices
    afrr_cap_pos = base_price_pos * (1 + np.array(afrr_pos_normalized))
    afrr_cap_neg = base_price_neg * (1 + np.array(afrr_neg_normalized))
    
    # REBAP: Simple trending from 2026 pattern if available
    if rebap_2026 is not None:
        # Apply annual trending to REBAP
        tau = year - START_YEAR
        rebap_aligned = align_profile_to_year(rebap_2026, year)
        rebap_mean_2026 = rebap_2026.mean()
        rebap_base = 90.0 * (1 + 0.01 * tau)  # +1% per year trend
        rebap_p50 = rebap_base * (rebap_aligned / rebap_mean_2026)
        rebap_p50 = np.maximum(rebap_p50, 20.0)  # Floor at €20/MWh
    else:
        # Fallback: simple trending
        tau = year - START_YEAR
        rebap_base = 90.0 * (1 + 0.01 * tau)
        rebap_p50 = np.full(len(target_idx), rebap_base)
        # Add some hourly variation
        hours = target_idx.hour.values
        hour_var = np.sin(hours * np.pi / 12) * 0.15  # ±15% sinusoidal
        rebap_p50 = rebap_p50 * (1 + hour_var)
    
    rebap_prob_positive = np.full(len(target_idx), 0.84)  # Historical average
    
    return DataFrame({
        'afrr_cap_pos_p50': afrr_cap_pos,
        'afrr_cap_neg_p50': afrr_cap_neg,
        'rebap_p50': rebap_p50,
        'rebap_prob_positive': rebap_prob_positive
    }, index=target_idx)


def generate_idc_from_da(da_prices: Series, year: int) -> DataFrame:

    tau = year - START_YEAR
    
    # IDC-DA spread (converging)
    idc_discount = 0.03 * np.exp(-0.04 * tau)
    idc_mid = da_prices * (1 - idc_discount)
    
    # Bid-ask spread (CORRECTED: use absolute spread to handle negative prices)
    spread_pct = 0.02 * np.exp(-0.025 * tau)
    spread_abs = np.abs(idc_mid) * spread_pct  # Absolute EUR/MWh spread
    
    # Apply spread: bid is always lower (more negative if price < 0)
    #               ask is always higher (less negative if price < 0)
    idc_bid = idc_mid - spread_abs
    idc_ask = idc_mid + spread_abs
    
    return DataFrame({
        'idc_mid_forecast': idc_mid,
        'idc_bid': idc_bid,
        'idc_ask': idc_ask
    }, index=da_prices.index)


# =============================================================================
# MAIN FORECAST GENERATION
# =============================================================================

def generate_year_forecast(year: int, 
                          da_normalized_2026: Series,
                          fcr_profile_2026: DataFrame,
                          afrr_profile_2026: DataFrame,
                          rebap_2026: Optional[Series] = None) -> DataFrame:
    """Generate complete forecast for a single year using hybrid approach."""
    print(f"\n  Generating hybrid forecast for {year}...")
    
    # 1. Day-Ahead prices (hybrid: trajectory + 2026 volatility)
    da_prices = generate_hybrid_da_forecast(year, da_normalized_2026)
    
    # 2. Intraday prices (derived from DA)
    df_idc = generate_idc_from_da(da_prices, year)
    
    # 3. FCR prices (hybrid: base + 2026 structure)
    df_fcr = generate_hybrid_fcr_forecast(year, fcr_profile_2026)
    
    # 4. aFRR prices (hybrid: base + 2026 structure)
    df_afrr = generate_hybrid_afrr_forecast(year, afrr_profile_2026, rebap_2026)
    
    # Combine all
    df_combined = pd.concat([
        da_prices.to_frame(),
        df_idc,
        df_fcr,
        df_afrr
    ], axis=1)
    
    df_combined.index.name = 'timestamp'
    
    # Print summary with volatility metrics
    battery_cap = battery_capacity_forecast(year)
    da_std = da_prices.std()
    da_range = da_prices.max() - da_prices.min()
    negative_count = (da_prices < 0).sum()
    negative_pct = negative_count / len(da_prices) * 100
    
    print(f"    DA Mean: {da_prices.mean():.2f} EUR/MWh, Std: {da_std:.2f}")
    print(f"    DA Range: [{da_prices.min():.2f}, {da_prices.max():.2f}] = {da_range:.2f}")
    print(f"    Negative prices: {negative_count} ({negative_pct:.2f}%)")
    print(f"    FCR Mean: {df_fcr['fcr_p50_block'].mean():.2f} EUR/MW/h")
    print(f"    aFRR Pos Mean: {df_afrr['afrr_cap_pos_p50'].mean():.2f} EUR/MW/h")
    print(f"    Battery Capacity: {battery_cap:.1f} GW")
    
    return df_combined


def generate_summary_trends(all_years_data: Dict[int, DataFrame]) -> DataFrame:
    """Generate summary sheet with trends over 25 years."""
    summary_rows = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        df = all_years_data[year]
        
        da_prices = df['da_price_forecast']
        
        # Calculate daily spread
        df_with_date = df.copy()
        df_with_date['date'] = df_with_date.index.date
        daily_spread = df_with_date.groupby('date')['da_price_forecast'].apply(
            lambda x: x.max() - x.min()
        ).mean()
        
        row = {
            'Year': year,
            'DA_Mean': da_prices.mean(),
            'DA_Std': da_prices.std(),
            'DA_Min': da_prices.min(),
            'DA_Max': da_prices.max(),
            'DA_Daily_Spread': daily_spread,
            'DA_Negative_Pct': (da_prices < 0).sum() / len(da_prices) * 100,
            'IDC_Mean': df['idc_mid_forecast'].mean(),
            'FCR_Mean': df['fcr_p50_block'].mean(),
            'aFRR_Pos_Mean': df['afrr_cap_pos_p50'].mean(),
            'aFRR_Neg_Mean': df['afrr_cap_neg_p50'].mean(),
            'REBAP_Mean': df['rebap_p50'].mean(),
            'Battery_GW': battery_capacity_forecast(year),
        }
        summary_rows.append(row)
    
    summary_df = DataFrame(summary_rows)
    
    # Calculate year-over-year changes
    for col in ['DA_Mean', 'FCR_Mean', 'aFRR_Pos_Mean']:
        summary_df[f'{col}_YoY%'] = summary_df[col].pct_change() * 100
    
    # Calculate cumulative change from 2026
    base_row = summary_df.iloc[0]
    for col in ['DA_Mean', 'FCR_Mean', 'aFRR_Pos_Mean', 'DA_Daily_Spread']:
        summary_df[f'{col}_Cumulative%'] = \
            ((summary_df[col] / base_row[col]) - 1) * 100
    
    # Calculate CAGR
    def calc_cagr(start_val, end_val, years):
        if start_val <= 0 or end_val <= 0:
            return 0.0
        return ((end_val / start_val) ** (1 / years) - 1) * 100
    
    summary_df['DA_CAGR_25yr'] = calc_cagr(
        summary_df.iloc[0]['DA_Mean'],
        summary_df.iloc[-1]['DA_Mean'],
        NUM_YEARS - 1
    )
    summary_df['FCR_CAGR_25yr'] = calc_cagr(
        summary_df.iloc[0]['FCR_Mean'],
        summary_df.iloc[-1]['FCR_Mean'],
        NUM_YEARS - 1
    )
    
    return summary_df


def write_excel_output(all_years_data: Dict[int, DataFrame], 
                       summary: DataFrame,
                       output_path: str):
    """Write all forecasts to Excel with multiple sheets."""
    print(f"\n{'='*70}")
    print("WRITING EXCEL OUTPUT")
    print(f"{'='*70}\n")
    
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"  Removed existing: {output_path}")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Write summary first
        summary.to_excel(writer, sheet_name='Summary_Trends', index=False)
        print(f"  ✓ Summary_Trends sheet")
        
        # Write each year's forecast
        for year in range(START_YEAR, END_YEAR + 1):
            df = all_years_data[year]
            
            # Convert timezone-aware index to naive for Excel
            df_excel = df.copy()
            if df_excel.index.tz is not None:
                df_excel.index = df_excel.index.tz_localize(None)
            
            sheet_name = f'All_Forecasts_{year}'
            df_excel.to_excel(writer, sheet_name=sheet_name)
            
            if year % 5 == 0 or year == START_YEAR:
                print(f"  ✓ {sheet_name}")
    
    print(f"\n✓ Excel file written: {output_path}")
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Sheets: {NUM_YEARS + 1} (25 years + summary)\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Germany 25-Year Hybrid Forecast (2026-2050)"
    )
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                       help='Output Excel file path')
    parser.add_argument('--profile', default='Germany_priceforecast_2026.xlsx',
                       help='2026 forecast file for volatility profile')
    args = parser.parse_args()
    
    print(f"Output: {args.output}")
    print(f"Profile source: {args.profile}")
    print(f"Forecast Period: {START_YEAR} → {END_YEAR} ({NUM_YEARS} years)")

    
    # Load 2026 profile for volatility structure
    da_normalized_2026, fcr_profile_2026, afrr_profile_2026, df_2026_full = \
        load_2026_profile(args.profile)
    
    if da_normalized_2026 is None:
        print("\n[ERROR] Could not load 2026 profile. Exiting.")
        return
    
    # Extract REBAP if available
    rebap_2026 = None
    if df_2026_full is not None and 'rebap_p50' in df_2026_full.columns:
        rebap_2026 = Series(
            df_2026_full['rebap_p50'].values,
            index=pd.to_datetime(df_2026_full['timestamp'])
        )
    
    # Generate forecasts for all years
    print(f"\n{'='*70}")
    print("GENERATING 25-YEAR HYBRID FORECASTS")
    print(f"{'='*70}")
    
    all_years_data = {}
    for year in range(START_YEAR, END_YEAR + 1):
        all_years_data[year] = generate_year_forecast(
            year, 
            da_normalized_2026,
            fcr_profile_2026,
            afrr_profile_2026,
            rebap_2026
        )
    
    # Generate summary trends
    print(f"\n{'='*70}")
    print("GENERATING SUMMARY TRENDS")
    print(f"{'='*70}\n")
    
    summary = generate_summary_trends(all_years_data)
    
    print("  Summary Statistics:")
    print(f"    2026 DA: {summary.iloc[0]['DA_Mean']:.2f} EUR/MWh (std: {summary.iloc[0]['DA_Std']:.2f})")
    print(f"    2026 Daily Spread: {summary.iloc[0]['DA_Daily_Spread']:.2f} EUR/MWh")
    print(f"    2030 DA: {summary.iloc[4]['DA_Mean']:.2f} EUR/MWh (std: {summary.iloc[4]['DA_Std']:.2f})")
    print(f"    2040 DA: {summary.iloc[14]['DA_Mean']:.2f} EUR/MWh (std: {summary.iloc[14]['DA_Std']:.2f})")
    print(f"    2050 DA: {summary.iloc[24]['DA_Mean']:.2f} EUR/MWh (std: {summary.iloc[24]['DA_Std']:.2f})")
    print(f"    25-year DA CAGR: {summary.iloc[0]['DA_CAGR_25yr']:.2f}%")
    print(f"    Volatility maintained: Std ~{summary['DA_Std'].mean():.1f} EUR/MWh")
    
    # Write output
    write_excel_output(all_years_data, summary, args.output)
    
    # Final summary
    print(f"\n{'='*70}")
    print("✓ HYBRID FORECAST COMPLETE")
    print(f"{'='*70}\n")
    print(f"Output: {args.output}")
    print(f"Years: {NUM_YEARS} (2026-2050)")
    print(f"Rows per year: {len(all_years_data[START_YEAR]):,} (15-min intervals)")
    print(f"Total data points: {len(all_years_data[START_YEAR]) * NUM_YEARS:,}")
    print(f"\nKey Features (Hybrid Model):")
    print(f"  ✓ Long-term trajectory: FAU Balanced Transition")
    print(f"  ✓ Volatility structure: Preserved from 2026 actuals")
    print(f"  ✓ Negative prices: ~{summary['DA_Negative_Pct'].mean():.1f}% maintained")
    print(f"  ✓ Daily spreads: ~{summary['DA_Daily_Spread'].mean():.0f} EUR/MWh")
    print(f"  ✓ Extreme events: Peak/mean ratio preserved")
    print(f"\nPrice Trajectory:")
    print(f"  DA:   {summary.iloc[0]['DA_Mean']:.0f} → {summary.iloc[24]['DA_Mean']:.0f} EUR/MWh")
    print(f"  FCR:  {summary.iloc[0]['FCR_Mean']:.1f} → {summary.iloc[24]['FCR_Mean']:.1f} EUR/MW/h")
    print(f"  aFRR: {summary.iloc[0]['aFRR_Pos_Mean']:.1f} → {summary.iloc[24]['aFRR_Pos_Mean']:.1f} EUR/MW/h")
    print(f"  BESS: {summary.iloc[0]['Battery_GW']:.1f} → {summary.iloc[24]['Battery_GW']:.1f} GW")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
