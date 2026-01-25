import pandapower as pp
import pandapower.plotting as plot
import matplotlib.pyplot as plt
import seaborn as sns

class GridVisualizer:
    """
    Handles advanced visualization of grid attacks and restorations.
    Uses pandapower's plotting capabilities.
    """
    
    @staticmethod
    def plot_voltage_time_series(results_df, save_path="phase2_voltage_time_series.png"):

        """
        Plots Voltage Magnitude vs. Simulation Steps (Phase 2 style).
        Matches user request: Orange line, Safety Limit, Red dots for violations.
        """
        try:
            plt.figure(figsize=(12, 6))
            steps = results_df['Step']
            voltages = results_df['Voltage']
            
            # 1. Plot Main Voltage Line
            plt.plot(steps, voltages, color='#d35400', linewidth=2.5, marker='o', label='Grid Voltage (Bus 671)')
            
            # 2. Safety Limit Line
            plt.axhline(y=0.95, color='gray', linestyle='--', linewidth=1.5, label='Safety Limit (0.95 p.u.)')
            plt.axhline(y=1.05, color='gray', linestyle='--', linewidth=1.5)

            # 3. Highlight Violations & Annotate Attacks
            violation_mask = (voltages < 0.95) | (voltages > 1.05)
            violations = results_df[violation_mask]
            
            if not violations.empty:
                plt.scatter(violations['Step'], violations['Voltage'], color='red', s=150, zorder=5, edgecolors='black', label='Violation Detected')
                
                # Annotate text for attacks
                for _, row in violations.iterrows():
                    plt.text(row['Step'], row['Voltage'] + 0.02, row['Attack'].upper(), 
                             color='red', fontweight='bold', ha='center')

            plt.title("EcoGuard Phase 2: Cascading Failure Analysis (Voltage Profile)", fontsize=14)
            plt.xlabel("Simulation Steps")
            plt.ylabel("Voltage Magnitude (p.u.)")
            plt.ylim(0.7, 1.1)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='lower left')
            
            filename = save_path
            plt.savefig(filename)
            plt.close()
            print(f"Saved Phase 2 time series: {filename}")
            
        except Exception as e:
            print(f"Visualization Error (Time Series): {e}")

    @staticmethod
    def plot_resilience_comparison(baseline_voltages, rl_voltages, attack_events=None, save_path="phase3_resilience_comparison.png"):

        """
        Plots comparison of Without RL vs With RL (Phase 3 style).
        Matches user request: Red Dashed (Without), Green Solid (With), Stability Limit.
        """
        try:
            plt.figure(figsize=(12, 6))
            steps = range(len(baseline_voltages))
            
            # 1. Plot Without RL (Cascading Failure)
            plt.plot(steps, baseline_voltages, color='#ff6b6b', linestyle='--', linewidth=3.5, label='Without RL (Cascading Failure)', alpha=0.9)
            
            # 2. Plot With RL (Self-Healing)
            # Use smaller width to sit "inside" the red one if they overlap, or distinct style
            plt.plot(steps, rl_voltages, color='green', linewidth=2.0, marker='o', markersize=6, label='With RL (Self-Healing Grid)')
            
            # Highlight the Improvement
            import numpy as np
            plt.fill_between(steps, baseline_voltages, rl_voltages, where=(np.array(rl_voltages) > np.array(baseline_voltages) + 0.01),
                             interpolate=True, color='green', alpha=0.15, label='Resilience Improvement')


            
            # 3. Stability Limit
            plt.axhline(y=0.95, color='black', linestyle=':', linewidth=1.5, label='Stability Limit')
            
            # 4. Annotations
            # Annotate "FIXED" where RL improved voltage significantly above baseline
            for i, (base, rl) in enumerate(zip(baseline_voltages, rl_voltages)):
                if rl > 0.9 and base < 0.8: # RL saved the grid
                    plt.text(i, rl + 0.02, "FIXED", color='green', fontweight='bold', ha='center')
                elif base < 0.8: # Baseline failed
                    # If we have attack names, use them, otherwise generic
                    name = attack_events[i].upper() if attack_events and i < len(attack_events) else "ATTACK"
                    plt.text(i, base - 0.05, name, color='red', fontsize=9, ha='center')

            plt.title("EcoGuard Phase 3: Resilience Analysis - AI vs. Cyber-Attacks", fontsize=14)
            plt.xlabel("Simulation Steps")
            plt.ylabel("Grid Voltage (p.u.)")
            plt.ylim(0.0, 1.1)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='lower left')
            
            filename = save_path
            plt.savefig(filename)
            plt.close()
            print(f"Saved Phase 3 comparison graph: {filename}")

            
        except Exception as e:
            print(f"Visualization Error (Comparison Plot): {e}")

