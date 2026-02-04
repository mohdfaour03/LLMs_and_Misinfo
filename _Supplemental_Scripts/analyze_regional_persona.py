
import pandas as pd
import numpy as np

# Load the scored data
try:
    df = pd.read_csv('herbal_claims_final_scored.csv')
except FileNotFoundError:
    print("Error: Scored file not found.")
    exit()

# Identify model columns
models = ['gemini', 'llama', 'chatgpt', 'falcon', 'mistral', 'deepseek']

# Check available regions
if 'region_or_culture' not in df.columns:
    print("Error: 'region_or_culture' column missing.")
    exit()

with open("analysis_results.txt", "w") as f:
    def log(msg):
        print(msg)
        f.write(msg + "\n")

    log(f"Total Rows: {len(df)}")
    
    log("\n--- Overall Model Ranking (Mean TVS) ---")
    overall_scores = {}
    for m in models:
        col = f"{m}_TVS"
        if col in df.columns:
            overall_scores[m] = df[col].mean()

    for m, score in sorted(overall_scores.items(), key=lambda x: x[1], reverse=True):
        log(f"{m.capitalize()}: {score:.4f}")

    log("\n--- REGIONAL SHOWDOWNS ---")
    
    def compare_log(region_name, model_a, model_b):
        region_mask = df['region_or_culture'].str.contains(region_name, case=False, na=False)
        subset = df[region_mask]
        
        if len(subset) == 0:
            log(f"No data for {region_name}")
            return
            
        score_a = subset[f"{model_a}_TVS"].mean()
        score_b = subset[f"{model_b}_TVS"].mean()
        
        log(f"Comparing {model_a.upper()} vs {model_b.upper()} on '{region_name}' ({len(subset)} prompts):")
        log(f"  {model_a.upper()}: {score_a:.4f}")
        log(f"  {model_b.upper()}: {score_b:.4f}")
        diff = score_a - score_b
        if diff > 0:
            log(f"  WINNER: {model_a.upper()} (+{diff:.4f})")
        else:
            log(f"  WINNER: {model_b.upper()} (+{abs(diff):.4f})")
        log("-" * 30)

    # DeepSeek vs others on China
    compare_log('China', 'deepseek', 'chatgpt')
    compare_log('China', 'deepseek', 'gemini')

    # Falcon vs others on MENA/Middle East
    compare_log('MENA', 'falcon', 'llama')
    compare_log('MENA', 'falcon', 'chatgpt')
    compare_log('Middle East', 'falcon', 'llama')
