
import pandas as pd

file = "herbal_claims_mistral_updated.csv"
try:
    df = pd.read_csv(file)
    print("Columns:", df.columns.tolist())
except:
    try:
        df = pd.read_csv(file, encoding='latin-1')
        print("Columns (latin-1):", df.columns.tolist())
    except Exception as e:
        print(f"Error: {e}")
