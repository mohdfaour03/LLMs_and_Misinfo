
import pandas as pd
import numpy as np

# 1. Get Falcon MENA from OLD file
try:
    df_old = pd.read_csv('herbal_claims_final_scored.csv')
    mena_mask_old = df_old['region_or_culture'].astype(str).str.contains('MENA', case=False) | \
                    df_old['region_or_culture'].astype(str).str.contains('Middle East', case=False)
    
    falcon_mena = df_old[mena_mask_old]['falcon_TVS'].mean()
    gpt_mena = df_old[mena_mask_old]['chatgpt_TVS'].mean()
    print(f"OLD FILE - MENA (n={mena_mask_old.sum()}): Falcon={falcon_mena:.4f}, ChatGPT={gpt_mena:.4f}")
except Exception as e:
    print(f"Old file error: {e}")

# 2. Get Mistral MENA from NEW file
try:
    df_new = pd.read_csv('herbal_claims_mistral_updated_scored.csv')
    # Use robust column finding for Mistral
    # Assuming 'mistral_TVS' or 'mistral_TVS_y'
    col = 'mistral_TVS'
    if col not in df_new.columns:
        if 'mistral_TVS_y' in df_new.columns: col = 'mistral_TVS_y'
        elif 'mistral_TVS_x' in df_new.columns: col = 'mistral_TVS_x'
    
    if col in df_new.columns:
        mena_mask_new = df_new['region_or_culture'].astype(str).str.contains('MENA', case=False) | \
                        df_new['region_or_culture'].astype(str).str.contains('Middle East', case=False)
        mistral_mena = df_new[mena_mask_new][col].mean()
        print(f"NEW FILE - MENA (n={mena_mask_new.sum()}): Mistral={mistral_mena:.4f}")
    else:
        print("Mistral TVS col not found.")
except Exception as e:
    print(f"New file error: {e}")
