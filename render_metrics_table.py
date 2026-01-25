import pandas as pd
import matplotlib.pyplot as plt

def render_table():
    try:
        # Read the metrics
        df = pd.read_csv("research_metrics.csv")
        
        # Select key columns for cleaner display if too wide, 
        # but 9 columns usually fit. Let's keep all.
        # Rename columns for readability (optional, but good for "Paper-Ready")
        df_display = df.rename(columns={
            'Avg_Volt_Base': 'Avg V (Base)',
            'Avg_Volt_RL': 'Avg V (RL)',
            'Violations_Base': 'Viol. (Base)',
            'Violations_RL': 'Viol. (RL)',
            'Resilience_Improvement_%': 'Improvement (%)'
        })

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6)) # Adjust size
        ax.axis('tight')
        ax.axis('off')
        
        # Table
        # cellText: values
        # colLabels: headers
        # loc: center
        table = ax.table(cellText=df_display.values, colLabels=df_display.columns, loc='center', cellLoc='center')
        
        # Style
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5) # Scale width, height
        
        # Color headers
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#404040')
            elif row % 2 == 0:
                cell.set_facecolor('#f2f2f2') # Alternating rows
        
        plt.title("EcoGuard: Quantitative Research Metrics (Baseline vs RL)", fontweight="bold", y=1.05)
        
        output_file = "research_metrics_table.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Table saved to {output_file}")
        
    except Exception as e:
        print(f"Error rendering table: {e}")

if __name__ == "__main__":
    render_table()
