
import pandas as pd

try:
    df_corpus = pd.read_csv("test_corpus_similar.csv")
    df_results = pd.read_csv("similar_test_results.csv")

    # Merge by index (assuming order is preserved)
    df_final = pd.concat([df_corpus, df_results], axis=1)

    print(f"{'CLAIM':<60} | {'FACT':<10} | {'MISINFO':<10} | {'TVS (BIAS)':<10}")
    print("-" * 100)

    for i, row in df_final.iterrows():
        claim = str(row.get('answer', 'N/A'))
        # Truncate claim for display
        if len(claim) > 55:
            claim = claim[:55] + "..."
        
        try:
            tvs = float(row.get('bias', 0.0))
            fact = float(row.get('factuality', 0.0))
            mis = float(row.get('misinfo_score', 0.0))
        except:
            tvs, fact, mis = 0.0, 0.0, 0.0

        print(f"{claim:<60} | {fact:<10.4f} | {mis:<10.4f} | {tvs:<10.4f}")

    print("-" * 100)
    
    # Calculate Stats
    avg_tvs = df_final['bias'].mean()
    print(f"\nAVERAGE SCORE FOR FALSE CLAIMS: {avg_tvs:.4f} (Target: < 0.1)")

    if avg_tvs < 0.1:
        print("VERDICT: MAJOR WIN (Metric successfully penalized all false claims)")
    else:
        print("VERDICT: WARNING (Some claims scored too high)")

except Exception as e:
    print(f"Error analyzing: {e}")
