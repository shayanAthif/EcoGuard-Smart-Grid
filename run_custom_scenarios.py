import os
import pandas as pd
import random
import warnings
from phase2_sim import run_phase2_simulation, generate_dummy_scenarios
from phase3_defense import run_comparative_simulation
from visualization import GridVisualizer
import pandapower as pp
from grid_models import GridBuilder

# Suppress all warnings for clean output
warnings.filterwarnings('ignore')

def generate_fixed_scenario(steps, scenario_id):
    """
    Generates a unique but fixed attack sequence for reproducibility.
    """
    attacks = ['dos', 'fuzzers', 'worms', 'exploits', 'data_injection', 'ransomware']
    scenarios = []
    
    # Simple deterministic logic based on ID
    # Scenario 1 (IEEE 13): Mostly DoS
    # Scenario 2 (IEEE 33): Mostly Fuzzers
    # ...
    
    start_offset = scenario_id % len(attacks)
    
    for i in range(1, steps + 1):
        if i % 3 == 0: # Attack every 3rd step
            idx = (start_offset + i) % len(attacks)
            attack_type = attacks[idx]
        else:
            attack_type = 'normal'
            
        scenarios.append({
            'AttackID': i,
            'DetectedEvent': attack_type,
            'Description': f"Scenario {scenario_id} Step {i}"
        })
        
    filename = f"Attack_Scenarios.csv" # Overwrite default for simplicity
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['AttackID', 'DetectedEvent', 'Description'])
        writer.writeheader()
        writer.writerows(scenarios)
    return filename

def run_batch_study():
    # 10 Scenarios across different grids (repeating with different attack patterns)
    grids = [
        'ieee13', 'ieee33', 'ieee300', 'ieee118', 'cigre_mv',  # 1-5
        'gbnetwork', 'ieee13', 'ieee33', 'ieee118', 'cigre_mv' # 6-10 (Variation)
    ]

    steps = 15
    
    if not os.path.exists("graphs"):
        os.makedirs("graphs")
        
    print(f"--- STARTING BATCH STUDY (5 Scenarios) ---")
    
    for i, grid_name in enumerate(grids):
        scenario_id = i + 1
        print(f"\n[Scenario {scenario_id}] Running for Grid: {grid_name.upper()}")
        
        # 1. Generate Scenario File
        generate_fixed_scenario(steps, scenario_id)
        
        # 2. Run Phase 2 (Time Series)
        try:
            print(f"  > Phase 2: Simulating Attacks...")
            # We call run_phase2_simulation but catch the result and plot manually to redirect file
            # Actually, modifying run_phase2 to accept save_path is cleaner, but to adhere to "don't touch my code unless needed"
            # I will invoke the function, let it generate 'simulation_results.csv', then load it and replot here.
            run_phase2_simulation(grid_type=grid_name, steps=steps)
            
            # Re-read and Plot to custom path
            results_df = pd.read_csv("simulation_results.csv")
            save_path = f"graphs/Scenario{scenario_id}_{grid_name}_Phase2_Voltage.png"
            GridVisualizer.plot_voltage_time_series(results_df, save_path=save_path)
            
        except Exception as e:
            print(f"  ! Error in Phase 2 for {grid_name}: {e}")
            
        # 3. Run Phase 3 (Comparative Defense)
        try:
            print(f"  > Phase 3: Testing Defense...")
            # I updated phase3_defense.py to take save_path argument.
            
            target_out = f"graphs/Scenario{scenario_id}_{grid_name}_Phase3_Defense.png"
            run_comparative_simulation(steps=steps, grid_type=grid_name, save_path=target_out)
            print(f"  > Saved Graph: {target_out}")
                
        except Exception as e:

             print(f"  ! Error in Phase 3 for {grid_name}: {e}")

    print("\n--- BATCH STUDY COMPLETE ---")
    print("Files available in 'graphs/' folder.")

if __name__ == "__main__":
    run_batch_study()
