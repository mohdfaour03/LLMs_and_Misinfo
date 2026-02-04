
import pandas as pd
import numpy as np

df = pd.read_csv('herbal_claims_mistral_updated_scored.csv')

models = ['gemini', 'llama', 'chatgpt', 'falcon', 'deepseek', 'mistral']
results = []

def find_col(base, columns):
    candidates = [base, base + "_y", base + "_x"]
    for c in candidates:
        if c in columns: return c
    return None

out = "=== VERIFIED SCORES ===\n"

for m in models:
    tvs_c = find_col(f"{m}_TVS", df.columns)
    
    # Try Finding Safety (1-Misinfo)
    mis_c = find_col(f"{m}_Misinfo", df.columns)
    
    # Try Finding TEC
    tec_c = find_col(f"{m}_TEC", df.columns)
    if not tec_c: tec_c = find_col(f"{m}_Clarity", df.columns)
    
    if tvs_c:
        tvs_val = df[tvs_c].mean()
        
        mis_val = df[mis_c].mean() if mis_c else 0.0
        safety = 1.0 - mis_val
        
        tec_val = df[tec_c].mean() if tec_c else 0.0
        
        line = f"{m.upper()}: TVS={tvs_val:.4f}, Safety={safety:.4f}, Clarity={tec_val:.4f}"
        print(line)
        out += line + "\n"

# China
try:
    china_mask = df['region_or_culture'].str.contains('China', case=False, na=False)
    china_df = df[china_mask]
    if not china_df.empty:
        ds_tvs_c = find_col("deepseek_TVS", df.columns)
        gpt_tvs_c = find_col("chatgpt_TVS", df.columns)
        
        if ds_tvs_c and gpt_tvs_c:
            ds = china_df[ds_tvs_c].mean()
            gpt = china_df[gpt_tvs_c].mean()
            line = f"CHINA: DeepSeek={ds:.4f}, ChatGPT={gpt:.4f}"
            print(line)
            out += line + "\n"
except:
    pass
    
with open("final_scores_verified.txt", "w") as f:
    f.write(out)
