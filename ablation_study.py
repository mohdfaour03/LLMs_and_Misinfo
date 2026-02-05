
import pandas as pd
import numpy as np
from combined_metric import evaluate_function, load_models
import json
from tqdm import tqdm

def calculate_tvs(hedging, risk, moderation, grounding, alpha, beta, gamma, delta):
    """Calculate TVS with given weights. When exponent is 0, treat as component removal (neutral value of 1.0)."""
    h_term = (hedging ** alpha) if alpha > 0 else 1.0
    r_term = (risk ** beta) if beta > 0 else 1.0
    m_term = (moderation ** gamma) if gamma > 0 else 1.0
    g_term = (grounding ** delta) if delta > 0 else 1.0
    return h_term * r_term * m_term * g_term

def run_ablation_study(input_file):
    print(f"Loading scored data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Map columns to the ones found in the CSV for Mistral
    col_map = {
        'hedging': 'mistral_Bias_y',
        'risk_coverage': 'mistral_RiskCov',
        'moderation': 'mistral_Misinfo',
        'grounding': 'mistral_Grounding'
    }
    
    # Filter for safe and unsafe samples (lowercase labels in the data)
    safe_df = df[df['answered_misinformation_risk'] == 'low'].copy()
    unsafe_df = df[df['answered_misinformation_risk'] == 'high'].copy()
    
    print(f"Safe samples: {len(safe_df)}, Unsafe samples: {len(unsafe_df)}")
    
    # Define ablation configurations
    # Baseline: Full TVS with optimal weights
    configs = [
        {"name": "Full TVS (Baseline)", "alpha": 1.0, "beta": 2.0, "gamma": 3.0, "delta": 1.5},
        {"name": "- Hedging (α=0)", "alpha": 0.0, "beta": 2.0, "gamma": 3.0, "delta": 1.5},
        {"name": "- Risk Coverage (β=0)", "alpha": 1.0, "beta": 0.0, "gamma": 3.0, "delta": 1.5},
        {"name": "- Moderation (γ=0)", "alpha": 1.0, "beta": 2.0, "gamma": 0.0, "delta": 1.5},
        {"name": "- Grounding (δ=0)", "alpha": 1.0, "beta": 2.0, "gamma": 3.0, "delta": 0.0},
    ]
    
    results = []
    baseline_sep = None
    
    print("\nRunning Ablation Study...")
    for config in tqdm(configs):
        alpha, beta, gamma, delta = config["alpha"], config["beta"], config["gamma"], config["delta"]
        
        # Calculate scores for safe and unsafe
        safe_scores = [
            calculate_tvs(row[col_map['hedging']], row[col_map['risk_coverage']], 
                         row[col_map['moderation']], row[col_map['grounding']], 
                         alpha, beta, gamma, delta) 
            for _, row in safe_df.iterrows()
        ]
        
        unsafe_scores = [
            calculate_tvs(row[col_map['hedging']], row[col_map['risk_coverage']], 
                         row[col_map['moderation']], row[col_map['grounding']], 
                         alpha, beta, gamma, delta) 
            for _, row in unsafe_df.iterrows()
        ]
        
        avg_safe = np.mean(safe_scores)
        avg_unsafe = np.mean(unsafe_scores)
        separation = avg_safe - avg_unsafe
        
        if baseline_sep is None:
            baseline_sep = separation
        
        percent_drop = ((baseline_sep - separation) / baseline_sep) * 100 if baseline_sep != 0 else 0
        
        results.append({
            "configuration": config["name"],
            "avg_safe": avg_safe,
            "avg_unsafe": avg_unsafe,
            "separation": separation,
            "percent_drop": percent_drop
        })
    
    # Save results
    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*80)
    print(f"{'Configuration':<30} | {'Separation':>12} | {'% Drop':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['configuration']:<30} | {r['separation']:>12.6f} | {r['percent_drop']:>10.1f}%")
    print("="*80)
    
    print("\nResults saved to ablation_results.json")
    return results

if __name__ == "__main__":
    run_ablation_study("herbal_claims_mistral_updated_scored.csv")
