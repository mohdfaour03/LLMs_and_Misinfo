
import pandas as pd
import os

FILE = "herbal_claims_mistral_updated_scored.csv"
if not os.path.exists(FILE):
    FILE = "herbal_claims_final_scored.csv" # Fallback
    print(f"Using fallback: {FILE}")

df = pd.read_csv(FILE)
models = ['gemini', 'llama', 'chatgpt', 'falcon', 'deepseek', 'mistral']

print("=== FINAL REPORT DATA ===")

data = []
for m in models:
    tvs_col = f"{m}_TVS"
    if tvs_col in df.columns:
        tvs = df[tvs_col].mean()
        
        # Safety (1-Misinfo)
        mis_col = [c for c in df.columns if m in c and 'Misinfo' in c]
        mis_val = df[mis_col[0]].mean() if mis_col else 0.0
        safety = 1.0 - mis_val
        
        # Clarity (TEC)
        tec_col = [c for c in df.columns if m in c and 'TEC' in c]
        tec_val = df[tec_col[0]].mean() if tec_col else 0.0
        
        print(f"{m.upper()}: TVS={tvs:.4f}, Safety={safety:.4f}, Clarity={tec_val:.4f}")
        data.append({'Model': m, 'TVS': tvs, 'Safety': safety, 'Clarity': tec_val})

# Regional
try:
    china_df = df[df['region_or_culture'].str.contains('China', case=False, na=False)]
    if not china_df.empty:
        ds_china = china_df['deepseek_TVS'].mean()
        gpt_china = china_df['chatgpt_TVS'].mean()
        print(f"\nCHINA INSIGHT: DeepSeek={ds_china:.4f} vs ChatGPT={gpt_china:.4f}")
        data.append({'Model': 'DeepSeek-China', 'TVS': ds_china})
        data.append({'Model': 'ChatGPT-China', 'TVS': gpt_china})
except:
    pass
    
pd.DataFrame(data).to_csv('final_stats_summary.csv', index=False)
print("Saved final_stats_summary.csv")
