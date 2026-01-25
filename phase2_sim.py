import pandapower as pp
import pandas as pd
import numpy as np
import warnings
import csv
import os
from grid_models import GridBuilder
from attack_logic import apply_attack, get_critical_assets
from visualization import GridVisualizer

# Suppress warnings
warnings.filterwarnings('ignore')

def generate_dummy_scenarios(steps=20):
    """Generates a dummy attack scenario CSV if none exists."""
    scenarios = []
    attacks = ['normal', 'dos', 'fuzzers', 'worms', 'exploits', 'reconnaissance', 'data_injection', 'ransomware']
    
    # Create a sequence with some stability and some attacks
    for i in range(1, steps + 1):
        if i % 5 == 0: # Attack every 5th step
             attack_type = np.random.choice(attacks[1:]) # Random attack
        else:
             attack_type = 'normal'
             
        scenarios.append({
            'AttackID': i,
            'DetectedEvent': attack_type,
            'Description': f"Step {i} Event"
        })
        
    filename = "Attack_Scenarios.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['AttackID', 'DetectedEvent', 'Description'])
        writer.writeheader()
        writer.writerows(scenarios)
    print(f"Generated dummy scenarios in {filename}")
    return filename

def run_phase2_simulation(grid_type="ieee13", steps=20):
    """
    Runs the Phase 2 Cyber-Physical Grid Simulation.
    """
    print(f"--- PHASE 2: CYBER-PHYSICAL SIMULATION ({grid_type.upper()}) ---")
    
    # 1. Load or Generate Scenarios
    if not os.path.exists("Attack_Scenarios.csv"):
        generate_dummy_scenarios(steps)
    
    scenarios = pd.read_csv("Attack_Scenarios.csv")
    # Limit steps if csv is longer than requested or pad if shorter (simple truncation for now)
    if len(scenarios) > steps:
        scenarios = scenarios.head(steps)
    
    print(f"Loaded {len(scenarios)} simulation steps.")
    
    results = []
    
    for i, row in scenarios.iterrows():
        # Force deterministic randomness to align with Phase 3
        import random
        random.seed(i)
        np.random.seed(i)

        
        # A. Reset Grid for each step (Static Simulation per step) or keep state?

        # The original notebook reset grid every step to simulate "What happens if X attacks NOW on normal grid"
        # We will follow that logic for consistency, but modularity allows changing this later.
        net = GridBuilder.get_grid(grid_type)
        
        # Determine Critical Bus for Observation
        critical_load_idx, _, _ = get_critical_assets(net)
        if critical_load_idx is not None:
             critical_bus_idx = net.load.at[critical_load_idx, 'bus']
             critical_bus_name = net.bus.at[critical_bus_idx, 'name']
        else:
             # Fallback to first bus if no loads
             critical_bus_idx = net.bus.index[0]
             critical_bus_name = net.bus.at[critical_bus_idx, 'name']

        # B. Apply Attack
        attack_type = row['DetectedEvent']
        desc, target = apply_attack(net, attack_type)
        
        # C. Solve Power Flow
        try:
            pp.runpp(net)
            # Record Voltage at Critical Bus
            v_pu = net.res_bus.at[critical_bus_idx, 'vm_pu']
            status = "Converged"
            
            # --- Visualization Hook ---
            # Identify violated buses
            # GridVisualizer.plot_cascading_failure(net, i+1, failed_lines=[], overloaded_lines=[]) 
            # Note: We need to track actual failed lines from attack logic to pass here if we want black lines.
            # Currently apply_attack modifies the net directly.

            
        except Exception:
            # Grid collapse
            v_pu = 0.0
            status = "Diverged (Blackout)"
            
        print(f"Step {i+1}: Type={attack_type} | Target={target} | V_bus{critical_bus_name}={v_pu:.4f} p.u. | {status}")
        
        results.append({
            'Step': i + 1,
            'Attack': attack_type,
            'Target': target,
            'Voltage': v_pu,
            'Status': status
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("simulation_results.csv", index=False)
    
    # Generate Time-Series Plot
    GridVisualizer.plot_voltage_time_series(results_df)
    
    print("\nSimulation Complete. Results saved to 'simulation_results.csv'.")
    
    return results_df, net # Return last network state for visualization if needed


if __name__ == "__main__":
    run_phase2_simulation()
