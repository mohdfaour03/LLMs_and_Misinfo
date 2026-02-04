
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Configuration ---
CSV_PATH = 'herbal_claims_mistral_updated_scored.csv'
MODELS = ['DeepSeek', 'Mistral', 'ChatGPT', 'Gemini', 'Llama', 'Falcon']
COLORS = {
    'DeepSeek': '#E50010', # Red
    'Mistral': '#FCDC00',  # Yellow/Gold
    'ChatGPT': '#00A67E',  # OpenAI Green
    'Gemini': '#4285F4',   # Google Blue
    'Llama': '#0668E1',    # Meta Blue
    'Falcon': '#7F00FF'    # Purple
}

def get_col(df, base_name):
    candidates = [f"{base_name.lower()}_TVS", f"{base_name.lower()}_TVS_y", f"{base_name.lower()}_TVS_x"]
    for c in candidates:
        if c in df.columns: return c
    return None

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows.")

    # --- 1. Data Prep ---
    model_scores = {}
    for m in MODELS:
        col = get_col(df, m)
        if col:
            model_scores[m] = df[col].mean()
    
    # --- 2. INSIGHT: Efficiency Paradox (Mistral vs Falcon) ---
    print("\n--- INSIGHT 1: The Efficiency Paradox ---")
    mistral_score = model_scores.get('Mistral', 0)
    falcon_score = model_scores.get('Falcon', 0)
    print(f"Mistral (7B): {mistral_score:.4f}")
    print(f"Falcon (180B): {falcon_score:.4f}")
    if mistral_score > falcon_score:
        print("RESULT: Mistral (Small) > Falcon (Large). Paradox CONFIRMED.")
    
    # FIG 1: Efficiency Bar Chart
    plt.figure(figsize=(6, 5))
    bars = plt.bar(['Mistral-7B', 'Falcon-180B'], [mistral_score, falcon_score], 
                   color=[COLORS['Mistral'], COLORS['Falcon']], edgecolor='black')
    plt.title('The Efficiency Paradox: Size != Safety', fontsize=12, fontweight='bold')
    plt.ylabel('Trust Score (TVS)')
    plt.ylim(0.5, 0.7)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05, 
                 f"{bar.get_height():.3f}", ha='center', color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_efficiency_paradox.png', dpi=300)
    print("Saved fig_efficiency_paradox.png")

    # --- 3. INSIGHT: Regional Personas ---
    print("\n--- INSIGHT 2: Regional Personas ---")
    
    regions = ['TCM', 'Ayurveda', 'MENA']
    tcm_mask = df['region_or_culture'].str.contains('China|TCM', case=False, na=False)
    mena_mask = df['region_or_culture'].str.contains('MENA|Middle East', case=False, na=False)
    
    ds_col = get_col(df, 'DeepSeek')
    fal_col = get_col(df, 'Falcon')
    
    if ds_col and fal_col:
        ds_tcm = df[tcm_mask][ds_col].mean()
        ds_global = df[ds_col].mean()
        print(f"DeepSeek TCM: {ds_tcm:.4f} (Global: {ds_global:.4f}) -> Lift: {ds_tcm - ds_global:.4f}")
        
        fal_mena = df[mena_mask][fal_col].mean()
        fal_global = df[fal_col].mean()
        print(f"Falcon MENA: {fal_mena:.4f} (Global: {fal_global:.4f}) -> Lift: {fal_mena - fal_global:.4f}")

    # FIG 2: Regional Heatmap/Grouped Bar
    # Let's do a grouped bar for DeepSeek vs Falcon on TCM vs MENA
    if ds_col and fal_col:
        labels = ['TCM Prompts', 'MENA Prompts']
        ds_scores = [df[tcm_mask][ds_col].mean(), df[mena_mask][ds_col].mean()]
        fal_scores = [df[tcm_mask][fal_col].mean(), df[mena_mask][fal_col].mean()]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.figure(figsize=(8, 5))
        plt.bar(x - width/2, ds_scores, width, label='DeepSeek (China)', color=COLORS['DeepSeek'])
        plt.bar(x + width/2, fal_scores, width, label='Falcon (UAE)', color=COLORS['Falcon'])
        
        plt.ylabel('Trust Score (TVS)')
        plt.title('Regional Alignment: DeepSeek vs Falcon')
        plt.xticks(x, labels)
        plt.ylim(0.5, 0.75)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('fig_regional_persona_comparison.png', dpi=300)
        print("Saved fig_regional_persona_comparison.png")

    # --- 4. INSIGHT: Alignment Tax ---
    print("\n--- INSIGHT 3: Alignment Tax ---")
    saf_scores = []
    clar_scores = []
    names = []
    
    for m in MODELS:
        # Find Safety/Clarity cols
        saf_col = [c for c in df.columns if m.lower() in c.lower() and ('safety' in c.lower() or 'misinfo' in c.lower())][0]
        clar_col = [c for c in df.columns if m.lower() in c.lower() and ('clarity' in c.lower() or 'tec' in c.lower())][0]
        
        # Handle 1-Misinfo inversion if needed
        val_saf = df[saf_col].mean()
        if 'misinfo' in saf_col.lower(): val_saf = 1 - val_saf
            
        val_clar = df[clar_col].mean()
        
        saf_scores.append(val_saf)
        clar_scores.append(val_clar)
        names.append(m)
        print(f"{m}: Safety={val_saf:.2f}, Clarity={val_clar:.2f}")

    # FIG 3: Scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(saf_scores, clar_scores, c=[COLORS[n] for n in names], s=300, edgecolors='black', alpha=0.8)
    
    for i, txt in enumerate(names):
        plt.text(saf_scores[i]+0.002, clar_scores[i], txt, fontsize=11, fontweight='bold')
        
    plt.title('The Alignment Tax: Safety vs Utility Trade-off')
    plt.xlabel('Safety Score (Higher is Safer)')
    plt.ylabel('Clarity Score (Higher is More Useful)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('fig_alignment_tax_final.png', dpi=300)
    print("Saved fig_alignment_tax_final.png")

if __name__ == "__main__":
    main()
