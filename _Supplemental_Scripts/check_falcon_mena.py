
import pandas as pd

try:
    df = pd.read_csv('herbal_claims_mistral_updated_scored.csv')
    
    # Filter for MENA
    # Assuming 'region_or_culture' or 'topic' might help identify MENA. 
    # The user mentioned 'MENA' in the report dataset description.
    
    mena_mask = df['region_or_culture'].astype(str).str.contains('MENA', case=False) | \
                df['region_or_culture'].astype(str).str.contains('Middle East', case=False) | \
                df['prompt'].astype(str).str.contains('Black Seed', case=False) | \
                df['prompt'].astype(str).str.contains('Za\'atar', case=False)
                
    mena_df = df[mena_mask]
    print(f"Found {len(mena_df)} MENA prompts.")
    
    if not mena_df.empty:
        falcon_score = mena_df['falcon_TVS'].mean()
        gpt_score = mena_df['chatgpt_TVS'].mean()
        mistral_score = mena_df['mistral_TVS'].mean()
        
        print(f"Falcon (MENA): {falcon_score:.4f}")
        print(f"ChatGPT (MENA): {gpt_score:.4f}")
        print(f"Mistral (MENA): {mistral_score:.4f}")
        
        if falcon_score > gpt_score:
            print("RESULT: Falcon OUTPERFORMS on MENA!")
        else:
            print("RESULT: Falcon does NOT outperform on MENA.")
            
except Exception as e:
    print(e)
