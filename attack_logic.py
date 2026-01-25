import pandapower as pp
import numpy as np

# Mapping "Cyber Attacks" to "Physical Grid Failures"
ATTACK_PHYSICS = {
    'dos':        {'action': 'trip_capacitor', 'desc': 'Capacitor Blocked (Voltage Drop)'},
    'fuzzers':    {'action': 'trip_line',      'desc': 'Line Protection Trip'},
    'worms':      {'action': 'overload_load',  'desc': 'Massive Overload (300%)'},
    'exploits':   {'action': 'overload_minor', 'desc': 'Minor Overload (120%)'},
    'shellcode':  {'action': 'trip_line',      'desc': 'Breaker Malfunction'},
    'data_injection': {'action': 'hidden_overload', 'desc': 'FDI: Masked Sensor Overload'},
    'ransomware':     {'action': 'multi_trip',      'desc': 'Ransomware: Multi-Asset Lockout'},
    'reconnaissance': {'action': 'none',       'desc': 'Scanning (No Impact)'},
    'normal':     {'action': 'none',           'desc': 'Stable Operation'},
    'generic':    {'action': 'none',           'desc': 'Stable Operation'}
}


def get_critical_assets(net):
    """
    Dynamically finds critical components in ANY pandapower network
    to avoid hardcoded naming errors.
    """
    # 1. Find Critical Load (Highest P_mw)
    if not net.load.empty:
        # Sort by active power consumption (descending)
        critical_load_idx = net.load.sort_values("p_mw", ascending=False).index[0]
    else:
        critical_load_idx = None

    # 2. Find Critical Line (Longest Line often represents backbone or high impedance)
    if not net.line.empty:
        # Sort by length (descending)
        critical_line_idx = net.line.sort_values("length_km", ascending=False).index[0]
    else:
        critical_line_idx = None

    # 3. Find Capacitor/Sgen
    if not net.sgen.empty:
        # Just pick the first available static generator/capacitor
        capacitor_idx = net.sgen.index[0]
    else:
        # Try to find a capacitor in the shunt table if sgen is empty
        if not net.shunt.empty:
             capacitor_idx = net.shunt.index[0]
        else:
            capacitor_idx = None
        
    return critical_load_idx, critical_line_idx, capacitor_idx

def apply_attack(net, attack_type):
    """
    Applies the physical consequences of a cyber attack to the grid.
    Returns impact description and target information for visualization.
    """
    
    # Get attack logic or default to generic/none
    logic = ATTACK_PHYSICS.get(attack_type, ATTACK_PHYSICS['generic'])
    action = logic['action']
    desc = logic['desc']
    
    load_idx, line_idx, cap_idx = get_critical_assets(net)
    target_info = None

    if action == 'trip_capacitor':
        if cap_idx is not None:
            # Check if it's an sgen or shunt
            if cap_idx in net.sgen.index and not net.sgen.empty: #Re-verify existence in sgen
                 net.sgen.at[cap_idx, 'in_service'] = False
                 target_info = f"Sgen {cap_idx}"
            elif not net.shunt.empty and cap_idx in net.shunt.index:
                 net.shunt.at[cap_idx, 'in_service'] = False
                 target_info = f"Shunt {cap_idx}"
        else:
            print(f"Warning: No capacitor/sgen found to trip for attack '{attack_type}'.")

    elif action == 'trip_line':
        if line_idx is not None:
            net.line.at[line_idx, 'in_service'] = False
            target_info = f"Line {line_idx}"
        else:
             print(f"Warning: No line found to trip for attack '{attack_type}'.")

    elif action == 'overload_load':
        if load_idx is not None:
            # Massive overload to force voltage sag
            net.load.at[load_idx, 'p_mw'] *= 7.0 
            target_info = f"Load {load_idx} (700%)"

        else:
             print(f"Warning: No load found to overload for attack '{attack_type}'.")

    elif action == 'overload_minor':
        if load_idx is not None:
            net.load.at[load_idx, 'p_mw'] *= 1.5
            target_info = f"Load {load_idx} (150%)"
        else:
             print(f"Warning: No load found to overload for attack '{attack_type}'.")

    elif action == 'hidden_overload':
        # False Data Injection logic - Increased severity to force reaction
        if load_idx is not None:
            net.load.at[load_idx, 'p_mw'] *= 4.0 
            desc = f"FDI Attack: Hidden Overload on Load {load_idx}"
            target_info = f"Load {load_idx} (Masked)"


    elif action == 'multi_trip':
        # Ransomware logic: Trip multiple lines
        lines_to_trip = net.line.sample(n=min(3, len(net.line))).index
        net.line.loc[lines_to_trip, 'in_service'] = False
        desc = f"Ransomware: Locked/Tripped {len(lines_to_trip)} Lines"
        target_info = f"Lines {list(lines_to_trip)}"


    
    return desc, target_info
