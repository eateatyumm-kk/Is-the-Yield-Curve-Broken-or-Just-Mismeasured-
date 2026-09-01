- data ingestion
downloaded 
'USREC': 'recession_nber',       # Monthly NBER recession flag
'T10Y3M': 'spread_10y3m',        # Daily 10y-3m yield spread
'DRTSCILM': 'sloos_tightening',  # Quarterly bank lending standards (%)[cite: 1]
'NFCI': 'financial_cond_index',  # Weekly financial conditions[cite: 1]
'BAA10Y': 'credit_spread_baa',   # Daily Baa corporate - 10y yield spread[cite: 1]
'FEDFUNDS': 'fed_funds_rate'     # Monthly effective Fed Funds rate[cite: 1]

these are enough data for H1 and H2 testing

- prohibit model testing
This is a model follows the research by Estrella & Mishkin they used prohibit regression to predict future recession using yield curve difference. model combines linear regression with standardised cdf. 

-> I first loaded the data and adjusted the data to monthly data. to prevent data leakage the average was recorded as end of month mean. 
-> predict the percentage of incoming recesion 12 months in future -> calculate the R2 data by comparing with the y value
-> compare the prediction with FED prediction -> make sure to adjust the time, and there are some slight deviation because FED original data prediction starts from 1950 while FRED data starts from 1980
-> 

