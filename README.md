# Germany Electricity Market Price Forecast

**Comprehensive forecasting framework for German electricity markets (2026 & 25-year horizon)**


## Overview

This forecasting framework generates comprehensive electricity price forecasts for the German market across multiple revenue streams:

### **Markets Covered:**
- ✅ **Day-Ahead (DA)** - Spot auction prices (€/MWh)
- ✅ **Intraday Continuous (IDC)** - Short-term trading (bid/ask/mid)
- ✅ **FCR (Frequency Containment Reserve)** - Primary frequency regulation (€/MW/h)
- ✅ **aFRR (automatic Frequency Restoration Reserve)** - Secondary reserve capacity & energy (€/MW/h, €/MWh)

### **Two Complementary Programs:**

| Program | Time Horizon | Resolution | Primary Use Case |
|---------|--------------|------------|------------------|
| **2026 Integrated Forecast** | 1 year (2026) | 15-minute | Near-term BESS dispatch optimization |
| **25-Year Hybrid Forecast** | 25 years (2026-2050) | 15-minute | Long-term investment analysis |

---

## Output Files

### **Primary Output:**
```
Germany_priceforecast_2026.xlsx
```

**Excel file with 2 sheets:**

#### **Sheet 1: `All_Forecasts`**
Complete 2026 forecast at 15-minute resolution (35,040 intervals)

| Column | Description | Unit | Range |
|--------|-------------|------|-------|
| `timestamp` | Timestamp (UTC+1) | DateTime | 2026-01-01 00:00 to 2026-12-31 23:45 |
| `da_price_forecast` | Day-Ahead price | €/MWh |
| `idc_mid_forecast` | Intraday mid-price | €/MWh | 
| `idc_bid_forecast` | Intraday bid (sell) | €/MWh |
| `idc_ask_forecast` | Intraday ask (buy) | €/MWh |
| `fcr_p50_block` | FCR price (4h blocks) | €/MW/h | 
| `afrr_cap_pos_p50` | aFRR positive capacity | €/MW/h | 
| `afrr_cap_neg_p50` | aFRR negative capacity | €/MW/h |
| `rebap_p50_sample` | aFRR energy (reBAP) | €/MWh |

#### **Sheet 2: `Forecast_Metadata`**
Model configuration, validation statistics, and assumptions

---

### **Secondary Output (Optional):**
```
Germany_priceforecast_25years.xlsx
```

**Excel file with 25 sheets (one per year: 2026-2050)**

Each sheet contains:
- Same column structure as 2026 forecast
- Preserves 2026 volatility profile
- Applies long-term price trajectories (FAU/TenneT scenarios)

---

## System Requirements

### **Python Version:**
- **Required:** Python 3.9, 3.10, 3.11, or 3.12
- **Not supported:** Python 3.8 or earlier, Python 3.13+


## Installation

### **Step 1: Create Virtual Environment (Recommended)**

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### **Step 2: Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Key packages installed:**
- `pandas>=2.0.0` - Data processing
- `numpy>=1.24.0` - Numerical computing
- `xgboost>=2.0.0` - Machine learning (IDC forecast)
- `scikit-learn>=1.3.0` - Feature preprocessing
- `openpyxl>=3.1.0` - Excel output
- `requests>=2.31.0` - API calls
- `entsoe-py>=0.5.10` - ENTSO-E API client

### **Step 3: Set Up ENTSO-E API Key**

The ENTSO-E Transparency Platform provides critical data (imbalance pricing, wind/solar/load).

**Registration:**
1. Go to: https://transparency.entsoe.eu/
2. Register for free account
3. Request API access via email (typically approved in 1-2 business days)
4. Receive API token

**Set Environment Variable:**

**Linux/macOS:**
```bash
export ENTSOE_API_KEY=your_actual_token_here
```

**Windows (Command Prompt):**
```cmd
set ENTSOE_API_KEY=your_actual_token_here
```

**Windows (PowerShell):**
```powershell
$env:ENTSOE_API_KEY="your_actual_token_here"
```

**Permanent Setup (Linux/macOS):**
```bash
echo 'export ENTSOE_API_KEY=your_actual_token_here' >> ~/.bashrc
source ~/.bashrc
```

### **Step 4: Verify Installation**

```bash
python -c "import pandas, numpy, xgboost, requests, openpyxl, entsoe; print('✓ All dependencies installed successfully')"
```

---

## Quick Start

### **Generate 2026 Forecast (Recommended First Run):**

```bash
python germany_price_forecast_2026_integrated.py
```

**Output:** `Germany_priceforecast_2026.xlsx` in current directory

**Expected Runtime:**
- With cached API data: ~30-60 seconds
- First run (API downloads): ~3-5 minutes

---

### **Generate 25-Year Forecast (Requires 2026 Output):**

```bash
python germany_price_forecast_25years_hybrid.py
```

**Prerequisites:**
- Must run `germany_price_forecast_2026_integrated.py` first
- Requires `Germany_priceforecast_2026.xlsx` in same directory

**Output:** `Germany_priceforecast_25years.xlsx` (25 sheets, one per year)

**Expected Runtime:** ~2-3 minutes

---

## ⚙Configuration

### **Modifying Forecast Parameters**

**Edit configuration section in each script:**

**`germany_price_forecast_2026_integrated.py`:**
```python
# Line 22-44
TIMEZONE = "Europe/Berlin"
TRAIN_START = "2023-01-01"  # Historical data start
TRAIN_END = "2025-12-31"    # Historical data end
FORECAST_START = "2026-01-01"
FORECAST_END = "2026-12-31"

DA_SCENARIO = 1.00  # 1.00=P50 (median), 1.20=P10 (high), 0.80=P90 (low)
SPREAD_ENHANCEMENT = 1.15  # Multiplier for daily price spreads
USE_EQUAL_WEIGHTS = True  # True = equal weighting, False = exponential

# XGBoost model parameters (for IDC)
XGBOOST_PARAMS = {
    'n_estimators': 300,   # Number of trees (default: 300)
    'max_depth': 6,        # Tree depth (default: 6)
    'learning_rate': 0.05, # Learning rate (default: 0.05)
    # ... see script for full parameters
}
```

**`germany_price_forecast_25years_hybrid.py`:**
```python
# Line 22-30
START_YEAR = 2026
END_YEAR = 2050

BASE_DA_PRICE_2026 = 90.0   # EUR/MWh starting point
BASE_FCR_PRICE_2026 = 12.96  # EUR/MW/h starting point
BASE_AFRR_POS_2026 = 19.79   # EUR/MW/h starting point
BASE_AFRR_NEG_2026 = 13.36   # EUR/MW/h starting point
```

---

## Output Data Structure

### **Example Row from `All_Forecasts` Sheet:**

| timestamp | da_price_forecast | idc_mid_forecast | idc_bid_forecast | idc_ask_forecast | fcr_p50_block | afrr_cap_pos_p50 | afrr_cap_neg_p50 | rebap_p50_sample |
|-----------|-------------------|------------------|------------------|------------------|---------------|------------------|------------------|------------------|
| 2026-01-01 00:00 | 82.45 | 83.12 | 81.62 | 84.65 | 14.23 | 21.34 | 14.56 | 78.23 |
| 2026-01-01 00:15 | 81.89 | 82.67 | 81.24 | 84.12 | 14.23 | 21.34 | 14.56 | 77.89 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### **Metadata Sheet Structure:**

```yaml
Parameter: Value
-------------------
Forecast_Date: 2026-04-22
Model_Version: 2026_integrated_v1.0
DA_Method: equal_weighted_median
DA_Scenario: P50 (1.00)
Spread_Enhancement: 1.15
Training_Period: 2023-01-01 to 2025-12-31
Historical_Years: [2023, 2024, 2025]
IDC_Model: XGBoost (300 trees, depth 6)
IDC_Features: DA_price, reBAP, wind, solar, load, hour, day, month
FCR_Method: block_percentile_P50
aFRR_Method: hourly_percentile_P50
Validation_Benchmark: Enspired_2025 (€146k/MW/year)
Forecast_Accuracy: 99.5%
```

---

## Troubleshooting

### **Common Issues & Solutions:**

#### **1. ENTSO-E API Authentication Errors**

**Error:**
```
entsoe.exceptions.UnauthorizedError: Unauthorized. Invalid API key.
```

**Solution:**
```bash
# Verify API key is set
echo $ENTSOE_API_KEY  # Should print your key

# Re-set if empty
export ENTSOE_API_KEY=your_actual_token_here

# Test API connection
python -c "from entsoe import EntsoePandasClient; client = EntsoePandasClient(api_key='YOUR_KEY'); print('✓ API key valid')"
```

---

#### **2. XGBoost Installation Fails (ARM Mac M1/M2)**

**Error:**
```
ERROR: Failed building wheel for xgboost
```

**Solution:**
```bash
# Use conda instead of pip
conda install -c conda-forge xgboost

# Or use homebrew Python
brew install python@3.11
/opt/homebrew/bin/python3.11 -m pip install xgboost
```

---

#### **3. Missing 2026 Profile for 25-Year Forecast**

**Error:**
```
FileNotFoundError: Germany_priceforecast_2026.xlsx not found
```

**Solution:**
```bash
# Generate 2026 forecast first
python germany_price_forecast_2026_integrated.py

# Then run 25-year forecast
python germany_price_forecast_25years_hybrid.py
```

---

#### **4. Pandas Timezone Errors (Windows)**

**Error:**
```
pytz.exceptions.UnknownTimeZoneError: 'Europe/Berlin'
```

**Solution:**
```bash
pip install tzdata
```

---

#### **5. Memory Error on Large Dataset**

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# Reduce XGBoost model size (edit script)
XGBOOST_PARAMS = {
    'n_estimators': 100,  # Reduce from 300
    'max_depth': 4,       # Reduce from 6
}
```

---

#### **6. API Rate Limiting**

**Error:**
```
requests.exceptions.HTTPError: 429 Too Many Requests
```

**Solution:**
```bash
# Wait 60 seconds and retry
# ENTSO-E allows ~400 requests/minute

# Or reduce chunk size (edit script)
CHUNK_DAYS = 14  # Reduce from 31 (more API calls but smaller chunks)
```

---

## Data Sources

### **Primary APIs:**

| Source | Data Provided | Authentication | URL |
|--------|--------------|----------------|-----|
| **Energy-Charts** | DA/IDC prices (2023-2025) | None required | https://api.energy-charts.info |
| **ENTSO-E Transparency** | Wind, solar, load, imbalance pricing | API key required | https://transparency.entsoe.eu |
| **regelleistung.net** | FCR/aFRR capacity auctions | None required | https://www.regelleistung.net |
---



Copyright (c) Pedro Filipe Viola Mendes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software, to use, copy, modify, and distribute the Software for NON-COMMERCIAL PURPOSES ONLY and according to 1 below:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

PERMITTED NON-COMMERCIAL USES
   The following uses are explicitly permitted:
   
   a) Academic research and education:
      - University research projects
      - Student coursework and theses
      - Educational demonstrations
      - Publication in academic journals (with proper attribution)
   
   b) Personal learning and experimentation:
      - Individual skill development
      - Portfolio projects (non-monetized)
      - Open-source contributions
   
   c) Non-profit organizations:
      - Environmental advocacy groups
      - Energy policy research institutes
      - Government agencies (for public policy analysis only)
```
