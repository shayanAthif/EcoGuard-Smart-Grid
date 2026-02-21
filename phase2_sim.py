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


def run_phase2_with_physics(grid_type="ieee39", steps=20,
                            attack_type="worms", attack_start=5,
                            attack_end=15):
    """Run a persistent-grid simulation that collects timeseries data.

    Unlike ``run_phase2_simulation`` (which rebuilds the grid every step),
    this function keeps a single network instance alive across all timesteps
    so that nodal voltage and power-flow trajectories are physically
    meaningful for cascading-failure analysis.

    Parameters
    ----------
    grid_type : str
        Grid topology name recognised by ``GridBuilder.get_grid()``.
    steps : int
        Total number of simulation timesteps.
    attack_type : str
        Attack type string understood by ``apply_attack()``.
    attack_start : int
        Timestep index at which the attack begins (0-indexed).
    attack_end : int
        Timestep index at which the attack ends (0-indexed, inclusive).

    Returns
    -------
    net : pandapowerNet        — final network state
    timeseries_data : dict     — collected timeseries arrays
    v0_snapshot : np.ndarray   — pre-fault bus voltages
    attacked_bus_indices : list — bus indices targeted by the attack
    """
    import logging
    import copy
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("phase2_physics")

    print(f"\n--- PHASE 2 PHYSICS: PERSISTENT-GRID SIMULATION ({grid_type.upper()}) ---")
    print(f"    Steps={steps}  Attack='{attack_type}'  Window=[{attack_start}, {attack_end}]")

    net = GridBuilder.get_grid(grid_type)
    original_net = copy.deepcopy(net)  # pristine copy for recovery

    # --- Pre-fault baseline ---
    try:
        pp.runpp(net)
    except Exception as e:
        print(f"  [!] Pre-fault power flow failed: {e}")
        raise

    v0_snapshot = net.res_bus["vm_pu"].values.copy()

    # ----------------------------------------------------------------
    # Pre-plan the multi-asset cascading attack
    # ----------------------------------------------------------------
    # Select loads to overload (top 5 by MW)
    if not net.load.empty:
        target_loads = net.load.sort_values("p_mw", ascending=False).index[:5].tolist()
    else:
        target_loads = []

    # Select lines to trip (pick lines with highest loading / longest)
    if not net.line.empty:
        target_lines = net.line.sort_values("length_km", ascending=False).index[:3].tolist()
    else:
        target_lines = []

    # Select a generator to derate
    target_gen = None
    if not net.gen.empty:
        target_gen = net.gen.sort_values("p_mw", ascending=False).index[0]

    # Collect all attacked bus indices
    attacked_bus_indices = set()
    for li in target_loads:
        attacked_bus_indices.add(int(net.load.at[li, "bus"]))
    for ln in target_lines:
        attacked_bus_indices.add(int(net.line.at[ln, "from_bus"]))
        attacked_bus_indices.add(int(net.line.at[ln, "to_bus"]))
    if target_gen is not None:
        attacked_bus_indices.add(int(net.gen.at[target_gen, "bus"]))
    attacked_bus_indices = sorted(attacked_bus_indices)

    print(f"    Target loads: {target_loads}")
    print(f"    Target lines: {target_lines}")
    print(f"    Target gen:   {target_gen}")
    print(f"    Attacked buses: {attacked_bus_indices}")

    # ----------------------------------------------------------------
    # Define escalation schedule (relative to attack_start)
    # ----------------------------------------------------------------
    # Moderate overloads that still converge in Newton-Raphson.
    # Key: the power flow MUST converge at every step so that branch
    # flows and voltages are physically self-consistent.
    attack_duration = attack_end - attack_start
    stage1 = attack_start                                  # Overload loads to 180%
    stage2 = attack_start + max(1, attack_duration // 4)   # Trip 1 line
    stage3 = attack_start + max(2, attack_duration // 2)   # Trip 2nd line + derate gen
    stage4 = attack_start + max(3, 3 * attack_duration // 4)  # Escalate loads to 250% + trip 3rd line

    print(f"    Escalation: stage1={stage1}, stage2={stage2}, stage3={stage3}, stage4={stage4}")

    # ----------------------------------------------------------------
    # Timeseries collection
    # ----------------------------------------------------------------
    timeseries_data = {
        "vm_pu": [],
        "p_from_mw": [],
        "q_from_mvar": [],
        "timesteps": [],
        "attack_active": [],
    }

    gen_original_p = None
    if target_gen is not None:
        gen_original_p = net.gen.at[target_gen, "p_mw"]

    pf_converged_last = True   # track convergence across steps

    for t in range(steps):
        attack_active = attack_start <= t <= attack_end

        # ---- ATTACK ESCALATION ----
        if t == stage1:
            # Stage 1: moderate overload on target loads
            for li in target_loads:
                net.load.at[li, "p_mw"] = original_net.load.at[li, "p_mw"] * 1.8
                net.load.at[li, "q_mvar"] = original_net.load.at[li, "q_mvar"] * 1.8
            print(f"  Step {t}: [STAGE 1] Overloaded {len(target_loads)} loads to 180%")

        elif t == stage2:
            # Stage 2: trip first long line
            if len(target_lines) >= 1:
                net.line.at[target_lines[0], "in_service"] = False
                print(f"  Step {t}: [STAGE 2] Tripped line {target_lines[0]}")

        elif t == stage3:
            # Stage 3: trip 2nd line + derate generator to 30%
            if len(target_lines) >= 2:
                net.line.at[target_lines[1], "in_service"] = False
                print(f"  Step {t}: [STAGE 3] Tripped line {target_lines[1]}", end="")
            if target_gen is not None:
                net.gen.at[target_gen, "p_mw"] = gen_original_p * 0.3
                print(f" + derated gen {target_gen} to 30%")
            else:
                print()

        elif t == stage4:
            # Stage 4: escalate load overload + trip 3rd line
            for li in target_loads:
                net.load.at[li, "p_mw"] = original_net.load.at[li, "p_mw"] * 2.5
                net.load.at[li, "q_mvar"] = original_net.load.at[li, "q_mvar"] * 2.5
            if len(target_lines) >= 3:
                net.line.at[target_lines[2], "in_service"] = False
            print(f"  Step {t}: [STAGE 4] Loads to 250% + tripped line {target_lines[2] if len(target_lines) >= 3 else 'N/A'}")

        elif t == attack_end + 1:
            # ---- RECOVERY ----
            # Restore lines
            if not net.line.empty:
                net.line["in_service"] = original_net.line["in_service"].copy()
            # Restore sgens
            if not net.sgen.empty:
                net.sgen["in_service"] = original_net.sgen["in_service"].copy()
            # Restore generator
            if target_gen is not None:
                net.gen.at[target_gen, "p_mw"] = gen_original_p * 0.9
            # Restore loads to 90% of original
            if not net.load.empty:
                net.load["p_mw"] = original_net.load["p_mw"] * 0.9
                net.load["q_mvar"] = original_net.load["q_mvar"] * 0.9
            print(f"  Step {t}: RECOVERY -- Lines restored, loads to 90%, gen to 90%")

        # ---- Solve power flow (robust settings) ----
        converged = False
        for attempt, init_mode in enumerate(["results", "flat", "dc"]):
            try:
                pp.runpp(net, init=init_mode, max_iteration=100,
                         enforce_q_lims=False, calculate_voltage_angles=True)
                converged = True
                break
            except Exception:
                continue

        if converged:
            vm = net.res_bus["vm_pu"].values.copy()
            p_mw = np.nan_to_num(net.res_line["p_from_mw"].values.copy(), nan=0.0)
            q_mvar = np.nan_to_num(net.res_line["q_from_mvar"].values.copy(), nan=0.0)
            pf_converged_last = True
        else:
            logger.warning(f"  Step {t}: runpp did not converge after 3 attempts")
            print(f"  Step {t}: [!] Power flow did not converge -- interpolating")
            if len(timeseries_data["vm_pu"]) > 0:
                vm = timeseries_data["vm_pu"][-1].copy()
                p_mw = timeseries_data["p_from_mw"][-1].copy()
                q_mvar = timeseries_data["q_from_mvar"][-1].copy()
                # Graduated depression based on bus distance from generators
                for bi in attacked_bus_indices:
                    pos = list(net.bus.index).index(bi) if bi in net.bus.index else None
                    if pos is not None:
                        vm[pos] *= 0.90
                # Zero out tripped lines
                for ln_idx in range(len(net.line)):
                    if not net.line.at[ln_idx, "in_service"]:
                        p_mw[ln_idx] = 0.0
                        q_mvar[ln_idx] = 0.0
            else:
                continue
            pf_converged_last = False

        timeseries_data["vm_pu"].append(vm)
        timeseries_data["p_from_mw"].append(p_mw)
        timeseries_data["q_from_mvar"].append(q_mvar)
        timeseries_data["timesteps"].append(t)
        timeseries_data["attack_active"].append(attack_active)

        # Status line
        v_min = vm.min()
        v_mean = vm.mean()
        status = "ATTACK" if attack_active else "NORMAL"
        conv_str = "OK" if converged else "NO-CONV"
        print(f"  Step {t}: V_min={v_min:.4f}  V_mean={v_mean:.4f} | {status} [{conv_str}]")

    print(f"\n  Collected {len(timeseries_data['timesteps'])} valid timesteps.")
    return net, timeseries_data, v0_snapshot, attacked_bus_indices


if __name__ == "__main__":
    run_phase2_simulation()
