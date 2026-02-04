
import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv('herbal_claims_mistral_updated_scored.csv')

# Define Models and likely column names
models = {
    'DeepSeek': ['deepseek_TVS', 'deepseek_TVS_y', 'deepseek_TVS_x'],
    'Mistral': ['mistral_TVS', 'mistral_TVS_y', 'mistral_TVS_x'],
    'ChatGPT': ['chatgpt_TVS', 'chatgpt_TVS_y', 'chatgpt_TVS_x'],
    'Falcon': ['falcon_TVS', 'falcon_TVS_y', 'falcon_TVS_x'],
    'Llama': ['llama_TVS', 'llama_TVS_y', 'llama_TVS_x'],
    'Gemini': ['gemini_TVS', 'gemini_TVS_y', 'gemini_TVS_x']
}

# Helper to get valid column
def get_col(candidates, cols):
    for c in candidates:
        if c in cols: return c
    return None

# Define Regions
def get_region_mask(df, region_name):
    s = df['region_or_culture'].astype(str)
    if region_name == 'TCM':
        return s.str.contains('China', case=False) | s.str.contains('TCM', case=False)
    elif region_name == 'Ayurveda':
        return s.str.contains('Ayurveda', case=False) | s.str.contains('India', case=False)
    elif region_name == 'MENA':
        return s.str.contains('MENA', case=False) | s.str.contains('Middle East', case=False)
    return pd.Series([False]*len(df))

regions = ['TCM', 'Ayurveda', 'MENA']

print("=== REGIONAL PERFORMANCE MATRIX (Mean TVS) ===")
print(f"{'Model':<10} | {'TCM':<8} | {'Ayurveda':<8} | {'MENA':<8} | {'Avg Gap (Max-Min)':<6}")
print("-" * 60)

for m_name, candidates in models.items():
    col = get_col(candidates, df.columns)
    if not col:
        print(f"{m_name:<10} | N/A")
        continue
        
    row_vals = []
    for r in regions:
        mask = get_region_mask(df, r)
        score = df[mask][col].mean()
        row_vals.append(score)
    
    # Insights
    tcm, ayu, mena = row_vals
    gap = max(row_vals) - min(row_vals)
    
    print(f"{m_name:<10} | {tcm:.4f}   | {ayu:.4f}   | {mena:.4f}   | {gap:.4f}")
    
    # Save to CSV
    import csv
    with open('regional_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'TCM', 'Ayurveda', 'MENA', 'Gap'])
        for m_name, candidates in models.items():
            col = get_col(candidates, df.columns)
            if col:
                row_vals = []
                for r in regions:
                    mask = get_region_mask(df, r)
                    score = df[mask][col].mean()
                    row_vals.append(score)
                writer.writerow([m_name, *row_vals, max(row_vals) - min(row_vals)])
    print("Saved regional_matrix.csv")

