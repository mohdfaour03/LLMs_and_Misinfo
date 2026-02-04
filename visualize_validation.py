
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def plot_stress_test(json_path, output_path):
    print("Generating Stress Test Chart...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    # Extract just the bias score (TVS)
    df['TVS Score'] = df['scores'].apply(lambda x: x['bias'])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, y='id', x='TVS Score', hue='id', palette='viridis', legend=False)
    plt.title('TVS Stress Test Results: Separation of Cases', fontsize=14)
    plt.xlabel('TVS Score (0-1)', fontsize=12)
    plt.ylabel('Scenario', fontsize=12)
    plt.axvline(x=0.1, color='r', linestyle='--', label='Danger Threshold (0.1)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

def plot_sweep_analysis(json_path, output_path):
    print("Generating Sweep Analysis Chart...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # We want to show the separation power.
    # Let's clean up the separation column if needed
    # (Assuming separation is negative if unsafe > safe? Wait, logic in tvs_sweep_analysis.py was separation = avg_safe - avg_unsafe)
    # Ideally we want positive separation (Safe > Unsafe) or meaningful magnitude.
    
    # Top 10 configurations by separation
    top_df = df.sort_values('separation', ascending=False).head(15)
    
    # Create a simpler string representation of params
    top_df['params_str'] = top_df['params'].apply(lambda x: f"H={x[0]}, R={x[1]}, M={x[2]}, G={x[3]}")
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_df, y='params_str', x='separation', hue='params_str', palette='coolwarm', legend=False)
    plt.title('Top 15 Parameter Configurations by Separation Power', fontsize=14)
    plt.xlabel('Separation (Avg Safe - Avg Unsafe)', fontsize=12)
    plt.ylabel('Weights (H=Hedging, R=Risk, M=Moderation, G=Grounding)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    if os.path.exists("stress_test_results.json"):
        plot_stress_test("stress_test_results.json", "validation_stress_test_chart.png")
    else:
        print("stress_test_results.json not found!")

    if os.path.exists("sweep_results.json"):
        plot_sweep_analysis("sweep_results.json", "validation_sweep_chart.png")
    else:
        print("sweep_results.json not found!")
