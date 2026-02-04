
import pandas as pd
import os

FILE = 'herbal_claims_final_scored.csv'
if not os.path.exists(FILE):
    print("FILE NOT FOUND")
    exit()

try:
    df = pd.read_csv(FILE)
    # Filter for MENA
    mena_mask = df['region_or_culture'].astype(str).str.contains('MENA', case=False) | \
                df['region_or_culture'].astype(str).str.contains('Middle East', case=False)
    
    mena_df = df[mena_mask]
    n = len(mena_df)
    print(f"MENA N={n}")
    
    if n > 0:
        falcon = mena_df['falcon_TVS'].mean()
        gpt = mena_df['chatgpt_TVS'].mean()
        print(f"FALCON_MENA: {falcon:.4f}")
        print(f"CHATGPT_MENA: {gpt:.4f}")
except Exception as e:
    print(e)
