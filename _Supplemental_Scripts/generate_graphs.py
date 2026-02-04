import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_all_graphs(csv_path='herbal_claims_mistral_updated_scored.csv'):
    # Load Data
    if not os.path.exists(csv_path):
        print(f"Graph Gen Error: '{csv_path}' not found.")
        return

    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded for graphs: {len(df)} rows.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    models = ['gemini', 'llama', 'chatgpt', 'falcon', 'mistral', 'deepseek']
    colors = ['#4285F4', '#4267B2', '#00A67E', '#7F00FF', '#FCDC00', '#E50010'] # Branding-ish colors

    # 1. Overall TVS Scores (Bar Chart)
    print("Generating Overall Score Chart...")
    overall_scores = {}
    for m in models:
        col_name = f"{m}_TVS"
        if col_name in df.columns:
            overall_scores[m] = df[col_name].mean()
    
    if overall_scores:
        sorted_scores = dict(sorted(overall_scores.items(), key=lambda item: item[1], reverse=True))

        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_scores.keys(), sorted_scores.values(), color=colors[:len(sorted_scores)], edgecolor='black', alpha=0.8)
        plt.ylabel('Trust Verification Score (TVS)')
        plt.title('Overall Model Performance')
        plt.ylim(0.4, 0.7) # Zoom in to show differences
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 3), ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('fig_overall_tvs.png', dpi=300)
        print("Saved fig_overall_tvs.png")

    # 2. Regional Persona: DeepSeek vs ChatGPT on "China"
    print("Generating China Comparison Chart...")
    if 'region_or_culture' in df.columns:
        china_mask = df['region_or_culture'].str.contains('China', case=False, na=False)
        china_df = df[china_mask]

        if not china_df.empty and 'deepseek_TVS' in df.columns and 'chatgpt_TVS' in df.columns:
            ds_score = china_df['deepseek_TVS'].mean()
            gpt_score = china_df['chatgpt_TVS'].mean()
            
            plt.figure(figsize=(6, 6))
            plt.bar(['DeepSeek (China)', 'ChatGPT (USA)'], [ds_score, gpt_score], color=['#E50010', '#00A67E'])
            plt.title('Regional Persona: TCM Prompts')
            plt.ylabel('TVS Score')
            plt.ylim(0.5, 0.8)
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig('fig_regional_china.png', dpi=300)
            print("Saved fig_regional_china.png")
    
    # 3. Alignment Tax: Safety vs Clarity (Scatter Plot)
    print("Generating Alignment Tax Chart...")

    saf_scores = []
    clar_scores = []
    plot_models = []

    for i, m in enumerate(models):
        # Find Safety Column
        saf_cols = [c for c in df.columns if m in c and ('Safety' in c or 'Misinfo' in c)]
        # Find Clarity/TEC Column
        clar_cols = [c for c in df.columns if m in c and ('Clarity' in c or 'TEC' in c)]
        
        if saf_cols and clar_cols:
            # Invert Misinfo if needed to get "Safety" (High is good)
            # If column is 'Misinfo', lower is better. Safety = 1 - Misinfo
            if 'Misinfo' in saf_cols[0]:
                saf_val = 1.0 - df[saf_cols[0]].mean()
            else:
                saf_val = df[saf_cols[0]].mean()
                
            clar_val = df[clar_cols[0]].mean()
            
            saf_scores.append(saf_val)
            clar_scores.append(clar_val)
            plot_models.append(m)

    if plot_models:
        plt.figure(figsize=(8, 6))
        plt.scatter(saf_scores, clar_scores, s=200, c=colors[:len(plot_models)], alpha=0.7, edgecolors='black')
        plt.title('The Alignment Tax: Safety vs. Clarity')
        plt.xlabel('Safety Score (1 - Misinfo)')
        plt.ylabel('Clarity / Specificity (TEC)')
        plt.grid(True, linestyle='--', alpha=0.5)

        # Annotate points
        for i, txt in enumerate(plot_models):
            plt.text(saf_scores[i]+0.001, clar_scores[i]+0.001, txt.capitalize(), fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('fig_alignment_tax.png', dpi=300)
        print("Saved fig_alignment_tax.png")

if __name__ == "__main__":
    generate_all_graphs()
