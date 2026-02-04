import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import os

# --- Configuration ---
CSV_PATH = 'herbal_claims_mistral_updated_scored.csv'
MODELS = ['DeepSeek', 'Mistral', 'Falcon', 'ChatGPT', 'Gemini', 'Llama']
COLORS = {
    'DeepSeek': '#D32F2F', # Red
    'Mistral': '#FBC02D',  # Yellow/Gold
    'ChatGPT': '#388E3C',  # Green
    'Gemini': '#1976D2',   # Blue
    'Llama': '#0288D1',    # Light Blue
    'Falcon': '#7B1FA2'    # Purple
}
COMPONENT_COLORS = {
    'Factuality': '#4CAF50',
    'Safety': '#2196F3',
    'Neutrality': '#9E9E9E',
    'Clarity': '#FFC107'
}

def get_col(df, model, metric_type):
    cols = df.columns
    candidates = [c for c in cols if model.lower() in c.lower()]
    selected = None
    for c in candidates:
        if metric_type.lower() in c.lower():
            selected = c
            break
    if not selected and metric_type == 'Clarity':
        for c in candidates:
            if 'tec' in c.lower(): selected = c; break
    return selected

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows.")

    results = {}
    for m in MODELS:
        tvs_col = get_col(df, m, 'TVS')
        saf_col = get_col(df, m, 'Safety') or get_col(df, m, 'Misinfo')
        clar_col = get_col(df, m, 'Clarity')
        neu_col = get_col(df, m, 'Neutrality') or get_col(df, m, 'Bias') # Assuming bias/neutrality exists or we infer

        tvs_val = df[tvs_col].mean() if tvs_col else 0
        
        # Safety
        if saf_col and 'Misinfo' in saf_col: saf_val = 1.0 - df[saf_col].mean()
        elif saf_col: saf_val = df[saf_col].mean()
        else: saf_val = 0
            
        clar_val = df[clar_col].mean() if clar_col else 0
        
        # Factuality (Back-calculated or estimated if column missing)
        # TVS = 0.4F + 0.3S + 0.2N + 0.1C
        # We need F and N. 
        # Let's try to find Factuality column
        fact_col = get_col(df, m, 'Factuality')
        if fact_col:
            fact_val = df[fact_col].mean()
        else:
            # Approx logic if missing
            fact_val = tvs_val # Placeholder if truly missing, but we should have it
            
        # Neutrality
        if neu_col:
            neu_val = df[neu_col].mean()
        else:
            neu_val = 0.5
            
        results[m] = {'TVS': tvs_val, 'Safety': saf_val, 'Clarity': clar_val, 'Factuality': fact_val, 'Neutrality': neu_val}

    # ==========================================
    # FIGURE 1: Alignment Tax (Scatter Plot)
    # ==========================================
    print("Generating Figure 1 (Alignment Tax)...")
    fig, ax = plt.subplots(figsize=(10, 7))
    x_vals = [results[m]['Clarity'] for m in MODELS]
    y_vals = [results[m]['Safety'] for m in MODELS]
    
    ax.scatter(x_vals, y_vals, s=300, c=[COLORS[m] for m in MODELS], edgecolors='black', alpha=0.8, zorder=5)
    
    for i, txt in enumerate(MODELS):
        ax.text(x_vals[i]+0.002, y_vals[i], txt, fontsize=12, fontweight='bold')
        
    ax.set_xlabel('Clarity (Utility & Actionability)', fontsize=12)
    ax.set_ylabel('Safety (Risk Avoidance)', fontsize=12)
    ax.set_title('The "Alignment Tax" Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Annotations
    ax.text(0.16, 0.82, "High Alignment Tax\n(Safe but Useless)", fontsize=10, color='red', alpha=0.7)
    ax.text(0.20, 0.82, "Ideal Pareto Frontier\n(Safe & Clear)", fontsize=10, color='green', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('Final_Figures/New_Fig1_Alignment_Tax.png', dpi=300)

    # ==========================================
    # FIGURE 2: Regional Persona (Radar Chart)
    # ==========================================
    print("Generating Figure 2 (Radar)...")
    region_labels = ['TCM', 'Ayurveda', 'MENA']
    tcm_mask = df['region_or_culture'].str.contains('China|TCM', case=False, na=False)
    ayu_mask = df['region_or_culture'].str.contains('India|Ayurveda', case=False, na=False)
    mena_mask = df['region_or_culture'].str.contains('MENA|Middle East', case=False, na=False)
    
    radar_models = ['DeepSeek', 'Gemini', 'Falcon']
    radar_data = {}
    
    for m in radar_models:
        col = get_col(df, m, 'TVS')
        if col:
            radar_data[m] = [
                df[tcm_mask][col].mean(), 
                df[ayu_mask][col].mean(), 
                df[mena_mask][col].mean()
            ]
        else:
             radar_data[m] = [0,0,0]

    angles = np.linspace(0, 2*np.pi, len(region_labels), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for m in radar_models:
        stats = radar_data[m]
        stats += stats[:1]
        ax.plot(angles, stats, label=m, linewidth=2, color=COLORS[m])
        ax.fill(angles, stats, color=COLORS[m], alpha=0.1)
        
    ax.set_ylim(0.55, 0.75)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(region_labels, fontsize=12, fontweight='bold')
    ax.set_title("Regional Personas", y=1.1, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.savefig('Final_Figures/New_Fig2_Regional_Radar.png', dpi=300)

    # ==========================================
    # FIGURE 3: Component Contribution (Stacked Bar)
    # ==========================================
    print("Generating Figure 3 (Stacked Bar)...")
    
    labels = MODELS
    # Weights: F=0.4, S=0.3, N=0.2, C=0.1
    # We plot the WEIGHTED contribution
    f_contrib = [results[m]['Factuality'] * 0.4 for m in MODELS]
    s_contrib = [results[m]['Safety'] * 0.3 for m in MODELS]
    n_contrib = [results[m]['Neutrality'] * 0.2 for m in MODELS]
    c_contrib = [results[m]['Clarity'] * 0.1 for m in MODELS]
    
    width = 0.5
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(labels, f_contrib, width, label='Factuality (0.4)', color=COMPONENT_COLORS['Factuality'])
    ax.bar(labels, s_contrib, width, bottom=f_contrib, label='Safety (0.3)', color=COMPONENT_COLORS['Safety'])
    
    # Bottom for N is F + S
    bot_n = [f + s for f, s in zip(f_contrib, s_contrib)]
    ax.bar(labels, n_contrib, width, bottom=bot_n, label='Neutrality (0.2)', color=COMPONENT_COLORS['Neutrality'])
    
    # Bottom for C is F + S + N
    bot_c = [f + s + n for f, s, n in zip(f_contrib, s_contrib, n_contrib)]
    ax.bar(labels, c_contrib, width, bottom=bot_c, label='Clarity (0.1)', color=COMPONENT_COLORS['Clarity'])
    
    ax.set_ylabel('Weighted Score Contribution')
    ax.set_title('Component Contribution Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Final_Figures/New_Fig3_Component_Stacked.png', dpi=300)
    print("Done. Saved New Figures 1, 2, 3.")

if __name__ == "__main__":
    main()
