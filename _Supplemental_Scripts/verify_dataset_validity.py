
import pandas as pd
import numpy as np

FILE = "herbal_claims_mistral_updated.csv"
REQUIRED_COLS = ["prompt", "mistral_response"]
# Evidence is tricky, it might be 'evidence_snippet' OR 'evidence' OR 'ground_truth_reference'
EVIDENCE_CANDIDATES = ["evidence_snippet", "evidence", "ground_truth_reference"]

print(f"Checking {FILE}...")

try:
    df = pd.read_csv(FILE, encoding='latin-1')
    cols = df.columns.tolist()
    print("Columns found:", cols)
    
    # 1. Check Required Columns
    missing = [c for c in REQUIRED_COLS if c not in cols]
    
    # Check for at least one evidence column
    evidence_col = None
    for c in EVIDENCE_CANDIDATES:
        if c in cols:
            evidence_col = c
            break
            
    if not evidence_col:
        missing.append("ANY Evidence Column")
    else:
        print(f"Using Evidence Column: '{evidence_col}'")

    if missing:
        print(f"\n[FAIL] Missing columns: {missing}")
    else:
        print("\n[PASS] All required columns referencable.")

    # 2. Check for Empty Values in Critical Areas
    print("-" * 30)
    print("Checking for empty/missing values...")
    
    for c in REQUIRED_COLS + [evidence_col]:
        if not c: continue
        # Count NaNs
        n_nan = df[c].isna().sum()
        # Count Empty Strings (after strip)
        # Convert to str first to safely strip
        n_empty = df[c].astype(str).str.strip().eq("").sum()
        # Count explicit 'nan' strings
        n_nan_str = df[c].astype(str).str.lower().eq("nan").sum()
        
        total_bad = n_nan + n_empty + n_nan_str
        
        if total_bad > 0:
            print(f"WARNING: Column '{c}' has {total_bad} missing/empty values!")
        else:
            print(f"Column '{c}': OK (0 missing)")

    print("-" * 30)
    if len(df) > 0:
        print(f"Total Rows: {len(df)}")
        print("Dataset looks good for scoring.")
    else:
        print("[FAIL] Dataset is empty.")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
