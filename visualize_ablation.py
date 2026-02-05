
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

def plot_ablation_results(json_path, output_path):
    print("Generating Ablation Study Chart...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Use absolute value of separation for visualization
    df['abs_separation'] = df['separation'].abs()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, y='configuration', x='abs_separation', hue='configuration', palette='viridis', legend=False)
    plt.title('Ablation Study: Impact of Removing TVS Components', fontsize=14)
    plt.xlabel('Absolute Separation Power', fontsize=12)
    plt.ylabel('Configuration', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_ablation_results("ablation_results.json", "ablation_chart.png")
