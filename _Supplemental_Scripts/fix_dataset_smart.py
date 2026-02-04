
import pandas as pd

INPUT_FILE = "herbal_claims_final_scored (1).csv"
OUTPUT_FILE = "herbal_claims_mistral_updated.csv"

try:
    df = pd.read_csv(INPUT_FILE, encoding='latin-1')
    print(f"Loaded {len(df)} rows.")
    
    # Find column fuzzily
    target_col = None
    for col in df.columns:
        if 'mistral' in col.lower() and ('answer' in col.lower() or 'reponse' in col.lower() or 'response' in col.lower()):
            # Avoid picking up 'mistral_model' if it exists, explicitly look for answer-like
            target_col = col
            # Prefer 'mistral answers' if multiple matches logic? No, just take first valid one
            break
            
    if target_col:
        print(f"Found target column: '{target_col}'")
        df.rename(columns={target_col: 'mistral_response'}, inplace=True)
        print(f"Renamed '{target_col}' to 'mistral_response'.")
        
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved to {OUTPUT_FILE}")
    else:
        print("ERROR: Could not find any column looking like 'mistral answers'.")
        # Print a few related
        mistral_cols = [c for c in df.columns if 'mistral' in c.lower()]
        print("Mistral-related columns found:", mistral_cols)

except Exception as e:
    print(f"Error: {e}")
