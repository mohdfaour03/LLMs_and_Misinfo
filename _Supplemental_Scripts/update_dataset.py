
import pandas as pd

INPUT_FILE = "herbal_claims_final_scored (1).csv"
OUTPUT_FILE = "herbal_claims_mistral_updated.csv"

try:
    df = pd.read_csv(INPUT_FILE, encoding='latin-1')
    print(f"Loaded {len(df)} rows.")
    
    # Check for target column
    if 'mistral answers' in df.columns:
        df.rename(columns={'mistral answers': 'mistral_response'}, inplace=True)
        print("Renamed 'mistral answers' to 'mistral_response'.")
    elif 'mistral_answers' in df.columns:
        df.rename(columns={'mistral_answers': 'mistral_response'}, inplace=True)
        print("Renamed 'mistral_answers' to 'mistral_response'.")
    elif 'mistral response' in df.columns:
        df.rename(columns={'mistral response': 'mistral_response'}, inplace=True)
        print("Renamed 'mistral response' to 'mistral_response'.")
    # Handle user typo mentioned "mistral reponses" just in case
    elif 'mistral reponses' in df.columns:
         df.rename(columns={'mistral reponses': 'mistral_response'}, inplace=True)
         print("Renamed 'mistral reponses' to 'mistral_response'.")

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE} with columns: {df.columns.tolist()}")

except Exception as e:
    print(f"Error: {e}")
