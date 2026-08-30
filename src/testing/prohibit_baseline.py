import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

def load_clean_data():
    df = pd.read_csv("data/raw_fred_data.csv")

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    monthly_spread = df['spread_10y3m'].resample('ME').mean()
    monthly_USREC = df['recession_nber'].resample('ME').last()
    target_12m = monthly_USREC.shift(-12)

    df_cleaned = pd.DataFrame({
        'T10Y3M': monthly_spread,
        'USREC' : monthly_USREC,
        'target_12m' : target_12m
    })

    df_cleaned = df_cleaned.dropna()
    return df_cleaned

df = load_clean_data()

X = sm.add_constant(df[['T10Y3M']])
y = df['target_12m']

model_h1 = sm.Probit(y, X).fit()

print(model_h1.summary())

df['pred_prob'] = model_h1.predict(X)
auc_score = roc_auc_score(y, df['pred_prob'])

print("\n--- BASELINE MODEL PERFORMANCE ---")
print(f"McFadden's Pseudo R-squared: {model_h1.prsquared:.4f}")
print(f"ROC-AUC Score:             {auc_score:.4f}")
print(f"Log-Likelihood:            {model_h1.llf:.2f}")
print(f"AIC:                       {model_h1.aic:.2f}")



