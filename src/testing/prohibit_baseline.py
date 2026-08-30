import os
import io
import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
import openpyxl
import matplotlib.pyplot as plt

# Loading data for probit model fitting
def load_clean_data(shift=-12):
    df = pd.read_csv("data/raw_fred_data.csv")

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    monthly_spread = df['spread_10y3m'].resample('ME').mean()
    monthly_USREC = df['recession_nber'].resample('ME').last()
    target_shifted = monthly_USREC.shift(shift)

    df_cleaned = pd.DataFrame({
        'T10Y3M': monthly_spread,
        'USREC' : monthly_USREC,
        'target_12m' : target_shifted
    })

    return df_cleaned.dropna()

# Downloading/loading the fed probability data (filtered to start date)
def load_ny_fed_data_local(filepath: str = "data/allmonth.xls", start_date: str = "1982-01-31") -> pd.DataFrame:
    if not os.path.exists(filepath):
        # Fallback check for .xlsx extension
        if os.path.exists("data/allmonth.xlsx"):
            filepath = "data/allmonth.xlsx"
        else:
            raise FileNotFoundError(f"Could not find file at '{filepath}'")

    print(f"Loading official NY Fed data from '{filepath}'...")

    engine = "openpyxl" if filepath.endswith(".xlsx") else "xlrd"
    df_fed = pd.read_excel(filepath, sheet_name=0, engine=engine)

    # Clean up column headers
    df_fed.columns = [str(col).strip() for col in df_fed.columns]

    # Map the exact columns from allmonth.xls
    df_fed["date"] = pd.to_datetime(df_fed["Date"])
    df_fed["nyfed_prob"] = pd.to_numeric(df_fed["Rec_prob"], errors="coerce")

    # Normalize percentage values to 0.0 - 1.0 scale if necessary
    if df_fed["nyfed_prob"].max() > 1.0:
        df_fed["nyfed_prob"] = df_fed["nyfed_prob"] / 100.0

    df_fed = df_fed[["date", "nyfed_prob"]].dropna().set_index("date")
    
    # Filter NY Fed data to start from FRED's start date (1982-01-31)
    df_fed = df_fed[df_fed.index >= start_date]
    
    return df_fed.resample("ME").last()

# ----------- Running Model Fit ------------
df = load_clean_data(shift=-12)

X = sm.add_constant(df[['T10Y3M']])
y = df['target_12m']

probit_model = sm.Probit(y, X).fit(disp=False)

# 1. Predictions indexed at time t (spread date)
df["pred_raw"] = probit_model.predict(X)

# 2. Shift predictions forward by +12 months to match NY Fed target-date indexing
df["our_pred_prob"] = df["pred_raw"].shift(12)

# Load NY Fed data starting from the same date as df
df_nyfed = load_ny_fed_data_local(start_date=df.index.min().strftime("%Y-%m-%d"))

# Merge predictions with official Fed data and drop NaNs introduced by shift(12)
df_comparison = df.join(df_nyfed, how="inner").dropna(
    subset=["our_pred_prob", "nyfed_prob", "target_12m"]
)

# Compute alignment metrics on clean comparison data
mae = mean_absolute_error(
    df_comparison["nyfed_prob"], df_comparison["our_pred_prob"]
)
rmse = np.sqrt(
    mean_squared_error(
        df_comparison["nyfed_prob"], df_comparison["our_pred_prob"]
    )
)
corr = df_comparison["our_pred_prob"].corr(df_comparison["nyfed_prob"])
auc_score = roc_auc_score(df_comparison["target_12m"], df_comparison["our_pred_prob"])

# ------------ Model Summary and Accuracy Data -----------
print(probit_model.summary())

print("\n--- BASELINE MODEL PERFORMANCE ---")
print(f"McFadden's Pseudo R-squared: {probit_model.prsquared:.4f}")
print(f"ROC-AUC Score:             {auc_score:.4f}")
print(f"Log-Likelihood:            {probit_model.llf:.2f}")
print(f"AIC:                       {probit_model.aic:.2f}")

# ------------ Printing the FED Comparison Data --------------
print("\n" + "=" * 50)
print("      NY FED REPLICATION COMPARISON METRICS      ")
print("=" * 50)
print(f"Sample Size (Months): {len(df_comparison)}")
print(f"Mean Absolute Error (MAE):  {mae:.6f}")
print(f"Root Mean Squared Error:     {rmse:.6f}")
print(f"Pearson Correlation:         {corr:.6f}")
print("=" * 50 + "\n")

# ------------ Plotting the Comparison Result ---------------
plt.figure(figsize=(12, 6))
plt.plot(
    df_comparison.index,
    df_comparison["nyfed_prob"],
    label="Official NY Fed Series",
    color="black",
    linewidth=2,
)
plt.plot(
    df_comparison.index,
    df_comparison["our_pred_prob"],
    label="Replicated Probit Model",
    color="crimson",
    linestyle="--",
    linewidth=1.5,
)

plt.fill_between(
    df_comparison.index,
    0,
    1,
    where=df_comparison["USREC"] == 1,
    color="gray",
    alpha=0.3,
    transform=plt.gca().get_xaxis_transform(),
    label="Actual Recession (NBER)",
)

plt.title(
    "Model Validation: Replicated Probit vs. Official NY Fed Recession Probabilities"
)
plt.ylabel("Recession Probability")
plt.xlabel("Date")
plt.legend(loc="upper left")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()