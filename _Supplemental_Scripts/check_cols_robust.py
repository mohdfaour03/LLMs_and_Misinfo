
import pandas as pd
df = pd.read_csv('herbal_claims_mistral_updated_scored.csv')
cols = df.columns.tolist()

print("--- SEARCHING COLUMNS ---")
for k in ['mistral', 'deepseek', 'gemini', 'chatgpt', 'llama', 'falcon']:
    matches = [c for c in cols if k in c.lower()]
    print(f"MATCHES FOR {k}: {matches}")
