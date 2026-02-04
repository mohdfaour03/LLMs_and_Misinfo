
import pandas as pd
try:
    df = pd.read_csv('herbal_claims_mistral_updated_scored.csv')
    counts = df['region_or_culture'].value_counts()
    print("COUNTS START")
    print(counts)
    print("COUNTS END")
except Exception as e:
    print(e)
