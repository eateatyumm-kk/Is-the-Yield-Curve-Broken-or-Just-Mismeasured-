import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def get_fred_client():
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY environment variable not found. Check your .env file.")
    return Fred(api_key=api_key)

def fetch_project_series(start_date="1980-01-01"):
    fred = get_fred_client()
    
    series_map = {
        'USREC': 'recession_nber',       # Monthly NBER recession flag
        'T10Y3M': 'spread_10y3m',        # Daily 10y-3m yield spread
        'DRTSCILM': 'sloos_tightening',  # Quarterly bank lending standards (%)[cite: 1]
        'NFCI': 'financial_cond_index',  # Weekly financial conditions[cite: 1]
        'BAA10Y': 'credit_spread_baa',   # Daily Baa corporate - 10y yield spread[cite: 1]
        'FEDFUNDS': 'fed_funds_rate'     # Monthly effective Fed Funds rate[cite: 1]
    }
    
    data_frames = {}
    for series_id, name in series_map.items():
        print(f"Fetching {series_id}...")
        # Passing observation_start filters dates directly at the API level
        s = fred.get_series(series_id, observation_start=start_date)
        data_frames[name] = s
        
    df = pd.DataFrame(data_frames)
    df.index.name = 'date'
    return df

if __name__ == "__main__":
    df = fetch_project_series(start_date="1980-01-01")

    # Define absolute path relative to project root
    output_dir = BASE_DIR / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "raw_fred_data.csv"
    
    df.to_csv(output_path)
    print(f"\nSuccess! Raw data saved to {output_path}")
    print(df.tail())