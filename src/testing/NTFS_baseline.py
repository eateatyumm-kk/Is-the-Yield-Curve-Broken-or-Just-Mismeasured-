import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path

# --- Load GSW data ---
df = pd.read_csv("data/feds200628.csv", skiprows=9)
df = df[['Date', 'BETA0', 'BETA1', 'BETA2', 'BETA3', 'TAU1', 'TAU2']].copy()
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] >= '1982-01-01']


def zero_yield(tau, beta0, beta1, beta2, beta3, tau1, tau2):
    term1 = beta0
    term2 = beta1 * ((1 - np.exp(-tau / tau1)) / (tau / tau1))
    term3 = beta2 * (((1 - np.exp(-tau / tau1)) / (tau / tau1)) - np.exp(-tau / tau1))
    term4 = beta3 * (((1 - np.exp(-tau / tau2)) / (tau / tau2)) - np.exp(-tau / tau2))
    return term1 + term2 + term3 + term4


def compute_ntfs(beta0, beta1, beta2, beta3, tau1, tau2):
    yld6 = zero_yield(1.5, beta0, beta1, beta2, beta3, tau1, tau2)
    yld7 = zero_yield(1.75, beta0, beta1, beta2, beta3, tau1, tau2)
    yld1 = zero_yield(0.25, beta0, beta1, beta2, beta3, tau1, tau2)
    fwd6 = 7 * yld7 - 6 * yld6
    return fwd6 - yld1


# Vectorized computation
df['NTFS'] = compute_ntfs(df['BETA0'], df['BETA1'], df['BETA2'], df['BETA3'], df['TAU1'], df['TAU2'])

# --- Monthly average ---
df = df.set_index('Date')
monthly_NTFS = df['NTFS'].resample('ME').mean()

# --- Merge with recession data ---
df_fred = pd.read_csv("data/raw_fred_data.csv")
df_fred['date'] = pd.to_datetime(df_fred['date'])
df_fred = df_fred.set_index('date')

monthly_USREC = df_fred['recession_nber'].resample('ME').last()
target_shifted = monthly_USREC.shift(-12)

df_cleaned = pd.DataFrame({
    'NTFS': monthly_NTFS,
    'USREC': monthly_USREC,
    'target_12m': target_shifted
}).dropna()


# --- Loaders ---
def loading_probit_predictions(filepath: str = "data/results/probit_predictions.csv") -> pd.DataFrame:
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Could not find file at '{filepath}'")

    df_probit = pd.read_csv(filepath)
    df_probit['spread_date'] = pd.to_datetime(df_probit['spread_date'])
    df_probit['target_date'] = pd.to_datetime(df_probit['target_date'])
    df_probit = df_probit.set_index('spread_date')

    return df_probit


def load_ny_fed_data_local(filepath: str = "data/allmonth.xls", start_date: str = "1982-01-31") -> pd.DataFrame:
    if not os.path.exists(filepath):
        if os.path.exists("data/allmonth.xlsx"):
            filepath = "data/allmonth.xlsx"
        else:
            raise FileNotFoundError(f"Could not find file at '{filepath}'")

    print(f"Loading official NY Fed data from '{filepath}'...")

    engine = "openpyxl" if filepath.endswith(".xlsx") else "xlrd"
    df_fed = pd.read_excel(filepath, sheet_name=0, engine=engine)
    df_fed.columns = [str(col).strip() for col in df_fed.columns]

    df_fed["Date"] = pd.to_datetime(df_fed["Date"])
    df_fed["nyfed_prob"] = pd.to_numeric(df_fed["Rec_prob"], errors="coerce")

    if df_fed["nyfed_prob"].max() > 1.0:
        df_fed["nyfed_prob"] = df_fed["nyfed_prob"] / 100.0

    df_fed = df_fed[["Date", "nyfed_prob"]].dropna().set_index("Date")
    df_fed = df_fed[df_fed.index >= start_date]

    return df_fed.resample("ME").last()


# ----- fitting probit model on NTFS -----

X = sm.add_constant(df_cleaned[['NTFS']])
y = df_cleaned['target_12m']

probit_model = sm.Probit(y, X).fit(disp=False)

# Predictions are already aligned to df_cleaned's index — no shift needed
df_cleaned["our_pred_prob"] = probit_model.predict(X).shift(12)

# --- Load comparison series ---
df_T10Y3M = loading_probit_predictions()
df_fed = load_ny_fed_data_local()

# Merge NTFS predictions with T10Y3M predictions and NY Fed data
df_comparison = (
    df_cleaned[["NTFS", "USREC", "target_12m", "our_pred_prob"]]
    .join(df_T10Y3M[["T10Y3M", "recession_prob"]].rename(columns={"recession_prob": "t10y3m_pred_prob"}), how="inner")
    .join(df_fed, how="inner")
    .dropna(subset=["our_pred_prob", "t10y3m_pred_prob", "nyfed_prob", "target_12m"])
)

# --------- NTFS vs NY Fed (official benchmark) ----------
mae_fed = mean_absolute_error(df_comparison["nyfed_prob"], df_comparison["our_pred_prob"])
rmse_fed = np.sqrt(mean_squared_error(df_comparison["nyfed_prob"], df_comparison["our_pred_prob"]))
corr_fed = df_comparison["our_pred_prob"].corr(df_comparison["nyfed_prob"])
auc_full = roc_auc_score(df_comparison["target_12m"], df_comparison["our_pred_prob"])

# --------- NTFS vs T10Y3M (H1 vs H2 comparison) ----------
mae_t10y3m = mean_absolute_error(df_comparison["t10y3m_pred_prob"], df_comparison["our_pred_prob"])
rmse_t10y3m = np.sqrt(mean_squared_error(df_comparison["t10y3m_pred_prob"], df_comparison["our_pred_prob"]))
corr_t10y3m = df_comparison["our_pred_prob"].corr(df_comparison["t10y3m_pred_prob"])

# AUC on the common comparison window (fair apples-to-apples vs T10Y3M's own AUC)
auc_comparison_window = roc_auc_score(df_comparison["target_12m"], df_comparison["our_pred_prob"])

# ------- saving the prediction probabilities to CSV for further analysis ---------

PROJECT_ROOT = Path(__file__).resolve().parents[2] if "__file__" in dir() else Path(".")
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

output_file = RESULTS_DIR / "NTFS_predictions.csv"

df_export = pd.DataFrame({
    'spread_date': df_cleaned.index,
    'target_date': df_cleaned.index + pd.DateOffset(months=12),
    'NTFS': df_cleaned['NTFS'],
    'recession_prob': df_cleaned["our_pred_prob"]
}).dropna(subset=["recession_prob"])

df_export.to_csv(output_file, index=False)
print(f"Predictions saved to '{output_file}'")

# ------------ Model Summary and Accuracy Data -----------
print(probit_model.summary())

print("\n--- NTFS BASELINE MODEL PERFORMANCE ---")
print(f"McFadden's Pseudo R-squared: {probit_model.prsquared:.4f}")
print(f"ROC-AUC Score (full sample): {auc_full:.4f}")
print(f"ROC-AUC Score (comparison window): {auc_comparison_window:.4f}")
print(f"Log-Likelihood:            {probit_model.llf:.2f}")
print(f"AIC:                       {probit_model.aic:.2f}")

print("\n" + "=" * 50)
print("      NTFS vs NY FED COMPARISON METRICS      ")
print("=" * 50)
print(f"Sample Size (Months): {len(df_comparison)}")
print(f"MAE:  {mae_fed:.6f}")
print(f"RMSE: {rmse_fed:.6f}")
print(f"Pearson Correlation: {corr_fed:.6f}")
print("=" * 50)

print("\n" + "=" * 50)
print("      NTFS vs T10Y3M COMPARISON METRICS      ")
print("=" * 50)
print(f"Sample Size (Months): {len(df_comparison)}")
print(f"MAE:  {mae_t10y3m:.6f}")
print(f"RMSE: {rmse_t10y3m:.6f}")
print(f"Pearson Correlation: {corr_t10y3m:.6f}")
print("=" * 50 + "\n")

# ------------ Plotting the Comparison Result ---------------
plt.figure(figsize=(12, 6))
plt.plot(df_comparison.index, df_comparison["nyfed_prob"], label="Official NY Fed Series",
         color="black", linewidth=2)
plt.plot(df_comparison.index, df_comparison["t10y3m_pred_prob"], label="T10Y3M Probit (H1)",
         color="steelblue", linestyle="--", linewidth=1.5)
plt.plot(df_comparison.index, df_comparison["our_pred_prob"], label="NTFS Probit (H2)",
         color="crimson", linestyle="--", linewidth=1.5)

plt.fill_between(
    df_comparison.index, 0, 1,
    where=df_comparison["USREC"] == 1,
    color="gray", alpha=0.3,
    transform=plt.gca().get_xaxis_transform(),
    label="Actual Recession (NBER)",
)

plt.title("Model Comparison: NTFS vs T10Y3M vs Official NY Fed Recession Probabilities")
plt.ylabel("Recession Probability")
plt.xlabel("Date")
plt.legend(loc="upper left")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()