
import pandas as pd

df = pd.read_csv('herbal_claims_mistral_updated_scored.csv')

# Find Falcon Column
fal_cols = [c for c in df.columns if 'falcon' in c.lower() and 'tvs' in c.lower()]
print(f"Falcon Columns Found: {fal_cols}")

if fal_cols:
    col = fal_cols[0] # Take the first one (e.g. falcon_TVS or falcon_TVS_y)
    
    # Global Score
    global_score = df[col].mean()
    
    # MENA Score
    mena_mask = df['region_or_culture'].astype(str).str.contains('MENA', case=False) | \
                df['region_or_culture'].astype(str).str.contains('Middle East', case=False)
    
    mena_score = df[mena_mask][col].mean()
    
    print(f"FALCON GLOBAL: {global_score:.4f}")
    print(f"FALCON MENA: {mena_score:.4f}")
    print(f"GAP: {mena_score - global_score:.4f}")

else:
    print("NO FALCON COLUMNS FOUND (Double Check Case?)")
