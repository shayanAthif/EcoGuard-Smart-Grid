"""
physics_analysis.py — Physics-Level Nodal Analysis for EcoGuard Smart Grid

Implements epidemic-model-based (SIR) cascading failure detection and
bus-level vulnerability indices for cyber-physical power grid analysis.

Physical Interpretation:
    In a power system, cascading failures propagate analogously to epidemics.
    A fault at one bus "infects" neighbouring buses through power-flow coupling,
    voltage depression, and frequency deviation.  This module quantifies that
    propagation using three nodal indices (Frequency Response D, Power Flow
    Coupling L, Voltage Offset I) combined into a composite Fault Severity
    Index R, then fits an SIR ODE model to the time-evolution of R to
    estimate contagion (lambda) and recovery (mu) rates.

Dependencies: numpy, scipy, matplotlib, networkx, pandapower
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import odeint

# ---------------------------------------------------------------------------
#  Ensure output directory exists
# ---------------------------------------------------------------------------
GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

# Try to use the seaborn style; fall back gracefully
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass  # use default


# ===================================================================
#  Function 1 — Voltage Timeseries
# ===================================================================
def plot_voltage_timeseries(timeseries_data, net, attack_start_step, attack_end_step):
    """Plot per-bus voltage magnitude over all simulation timesteps.

    Physical meaning
    ----------------
    Voltage magnitude (vm_pu) reflects the balance between generation, load,
    and reactive-power support at each bus.  A sustained drop indicates the
    bus is approaching voltage instability — the mechanism by which cascading
    blackouts propagate through the network.

    Parameters
    ----------
    timeseries_data : dict
        Keys: 'vm_pu' (list of arrays), 'timesteps' (list of int),
              'attack_active' (list of bool).
    net : pandapowerNet
        The pandapower network (for bus names).
    attack_start_step, attack_end_step : int
        Timestep indices marking the attack window.
    """
    vm_matrix = np.array(timeseries_data["vm_pu"])  # (T, n_bus)
    timesteps = np.array(timeseries_data["timesteps"])
    n_bus = vm_matrix.shape[1]

    # Identify 3 most-affected buses (largest max voltage drop from t=0)
    v0 = vm_matrix[0]
    max_drop = v0 - vm_matrix.min(axis=0)
    top3 = np.argsort(max_drop)[-3:][::-1]

    bus_names = net.bus["name"].values if "name" in net.bus.columns else np.arange(n_bus)

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.tab20(np.linspace(0, 1, n_bus))

    for b in range(n_bus):
        lw, alpha_val = (0.7, 0.45) if b not in top3 else (2.5, 1.0)
        label = str(bus_names[b]) if b in top3 else None
        ax.plot(timesteps, vm_matrix[:, b], color=cmap[b % len(cmap)],
                linewidth=lw, alpha=alpha_val, label=label)

    ax.axvline(attack_start_step, color="red", linestyle="--", linewidth=1.2, label="Attack Start")
    ax.axvline(attack_end_step, color="green", linestyle="--", linewidth=1.2, label="Attack End")

    ax.set_title("Bus Voltage Profiles During Attack Event", fontsize=14)
    ax.set_xlabel("Timestep (s)")
    ax.set_ylabel("Voltage Magnitude (p.u.)")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(GRAPH_DIR, "voltage_timeseries.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ===================================================================
#  Function 2 — Branch Power Flow
# ===================================================================
def plot_branch_power(timeseries_data, net, attacked_bus_indices,
                      attack_start_step, attack_end_step):
    """Plot active & reactive power on lines adjacent to attacked buses.

    Physical meaning
    ----------------
    Branch power flow reveals how energy is redistributed after a fault.
    Lines adjacent to attacked buses experience sudden flow changes that can
    trigger protective-relay operations and cascade to further outages.

    Parameters
    ----------
    timeseries_data : dict
        Keys include 'p_from_mw' and 'q_from_mvar' (lists of arrays).
    net : pandapowerNet
    attacked_bus_indices : list[int]
    attack_start_step, attack_end_step : int
    """
    p_matrix = np.array(timeseries_data["p_from_mw"])   # (T, n_line)
    q_matrix = np.array(timeseries_data["q_from_mvar"])  # (T, n_line)
    timesteps = np.array(timeseries_data["timesteps"])

    # Find lines connected to attacked buses
    adj_lines = set()
    for idx in attacked_bus_indices:
        mask_from = net.line["from_bus"] == idx
        mask_to = net.line["to_bus"] == idx
        adj_lines.update(net.line.index[mask_from | mask_to].tolist())
    adj_lines = sorted(adj_lines)

    if len(adj_lines) == 0:
        print("  [!] No lines adjacent to attacked buses -- skipping branch power plot.")
        return

    line_names = []
    for li in adj_lines:
        name = net.line.at[li, "name"] if "name" in net.line.columns and net.line.at[li, "name"] else f"Line {li}"
        line_names.append(name)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    cmap = plt.cm.Set1(np.linspace(0, 1, max(len(adj_lines), 1)))

    for i, li in enumerate(adj_lines):
        ax1.plot(timesteps, p_matrix[:, li], color=cmap[i % len(cmap)],
                 linewidth=1.4, label=line_names[i])
        ax2.plot(timesteps, q_matrix[:, li], color=cmap[i % len(cmap)],
                 linewidth=1.4, label=line_names[i])

    for ax in (ax1, ax2):
        ax.axvline(attack_start_step, color="red", linestyle="--", linewidth=1)
        ax.axvline(attack_end_step, color="green", linestyle="--", linewidth=1)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("Active Power (MW)")
    ax1.set_title("Branch Power Flow - Lines Adjacent to Attacked Nodes", fontsize=13)
    ax2.set_ylabel("Reactive Power (MVAr)")
    ax2.set_xlabel("Timestep (s)")
    fig.tight_layout()
    path = os.path.join(GRAPH_DIR, "branch_power.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ===================================================================
#  Function 3 — Voltage Offset Index (I_v)
# ===================================================================
def compute_voltage_offset_index(net, v0_snapshot, vcr=0.9):
    """Compute the Voltage Offset Index for every bus.

    Physical meaning
    ----------------
    I_v measures how close a bus's current voltage is to the critical voltage
    threshold (vcr) relative to its healthy pre-fault baseline.  A value near
    0 means the bus is at the edge of voltage collapse; a value near 1 means
    it is comfortably within safe margins.

    Parameters
    ----------
    net : pandapowerNet   (must have res_bus populated via runpp)
    v0_snapshot : np.ndarray  — pre-fault voltage vector
    vcr : float — critical voltage threshold (p.u.)

    Returns
    -------
    I_v : np.ndarray of shape (n_bus,) clipped to [0, 1]
    """
    vm = net.res_bus["vm_pu"].values.copy()
    # I = 0 when voltage is at or below critical threshold (vulnerable)
    # I = 1 when voltage is at pre-fault level (healthy)
    # For voltages below Vcr, I = 0 (fully vulnerable)
    numer = np.maximum(vm - vcr, 0.0)
    denom = np.abs(v0_snapshot - vcr)
    denom[denom < 1e-12] = 1e-12
    I_v = numer / denom
    return np.clip(I_v, 0.0, 1.0)


# ===================================================================
#  Function 4 — Power Flow Coupling Index (L_v)
# ===================================================================
def compute_power_flow_coupling_index(net):
    """Compute entropy-based power-flow coupling index per bus.

    Physical meaning
    ----------------
    At each bus the outgoing and incoming power flows create a distribution.
    High entropy means the bus distributes power evenly across many branches
    (tightly coupled, harder to isolate during a fault).  Low entropy means
    flow is concentrated on one branch (easier to isolate).  We weight by
    apparent power to emphasise heavily-loaded connections.

    Returns
    -------
    L_v : np.ndarray of shape (n_bus,)
    """
    n_bus = len(net.bus)
    L_out = np.zeros(n_bus)
    L_in = np.zeros(n_bus)

    if net.res_line.empty:
        return np.zeros(n_bus)

    p_from = net.res_line["p_from_mw"].values
    q_from = net.res_line["q_from_mvar"].values
    s_from = np.sqrt(p_from ** 2 + q_from ** 2)

    from_buses = net.line["from_bus"].values
    to_buses = net.line["to_bus"].values

    for bus_i in range(n_bus):
        # Outgoing: lines where bus_i is the from_bus
        out_mask = from_buses == net.bus.index[bus_i]
        out_s = s_from[out_mask]
        if len(out_s) > 0 and out_s.sum() > 1e-12:
            ratios = out_s / out_s.sum()
            ratios = ratios[ratios > 1e-12]
            L_out[bus_i] = -np.sum(ratios * np.log(ratios)) * out_s.sum()

        # Incoming: lines where bus_i is the to_bus
        in_mask = to_buses == net.bus.index[bus_i]
        in_s = s_from[in_mask]
        if len(in_s) > 0 and in_s.sum() > 1e-12:
            ratios = in_s / in_s.sum()
            ratios = ratios[ratios > 1e-12]
            L_in[bus_i] = -np.sum(ratios * np.log(ratios)) * in_s.sum()

    L_v = 0.5 * L_out + 0.5 * L_in
    return L_v


# ===================================================================
#  Function 5 — Frequency Response Index (D_v)
# ===================================================================
def compute_frequency_response_index(net):
    """Compute frequency-response vulnerability index per bus.

    Physical meaning
    ----------------
    After a generation–load imbalance, system frequency deviates.  Buses
    electrically close to low-inertia generators (e.g. wind/solar inverters)
    experience faster and deeper frequency excursions.  D_v quantifies this
    vulnerability by weighting generator inertia constants by electrical
    distance (from the susceptance matrix) to each bus.

    Returns
    -------
    D_v : np.ndarray of shape (n_bus,)
    """
    n_bus = len(net.bus)
    bus_indices = net.bus.index.values

    # --- Collect generator info (bus, inertia H) ---
    gen_info = []  # (bus_idx, H)

    # Synchronous generators -- scale H by MW rating for realistic variation
    if not net.gen.empty:
        p_max = net.gen["p_mw"].max()
        p_max = max(p_max, 1.0)  # avoid division by zero
        for _, g in net.gen.iterrows():
            # H ranges from 2.5 (smallest unit) to 8.0 (largest unit)
            H = 2.5 + 5.5 * (g["p_mw"] / p_max)
            gen_info.append((g["bus"], H))

    # External grid (slack) -- high but not infinite inertia
    if not net.ext_grid.empty:
        for _, eg in net.ext_grid.iterrows():
            gen_info.append((eg["bus"], 20.0))

    # Static generators (sgen) -- renewables / inverter-based -- low inertia
    if not net.sgen.empty:
        for _, sg in net.sgen.iterrows():
            gen_info.append((sg["bus"], 0.5))

    if len(gen_info) == 0:
        return np.zeros(n_bus)

    # --- Build distance matrix from Ybus ---
    use_topological = True
    try:
        if hasattr(net, "_ppc") and net._ppc is not None:
            Ybus = net._ppc["internal"]["Ybus"]
            B = Ybus.toarray().imag  # susceptance matrix
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Pseudo-inverse to get electrical distances
                B_abs = np.abs(B)
                np.fill_diagonal(B_abs, 0)
                # Distance ≈ 1 / |B_ij|  (off-diagonal)
                dist_matrix = np.zeros_like(B_abs)
                nonzero = B_abs > 1e-12
                dist_matrix[nonzero] = 1.0 / B_abs[nonzero]
                dist_matrix[~nonzero] = 1e6  # large distance for unconnected
                np.fill_diagonal(dist_matrix, 0)
            use_topological = False
    except Exception:
        pass

    if use_topological:
        # Fallback: topological hop distance via networkx
        G = nx.Graph()
        for _, row in net.line.iterrows():
            if row.get("in_service", True):
                G.add_edge(row["from_bus"], row["to_bus"])
        for _, row in net.trafo.iterrows():
            if row.get("in_service", True):
                G.add_edge(row["hv_bus"], row["lv_bus"])

        dist_matrix = np.full((n_bus, n_bus), 1e6)
        bus_to_pos = {b: i for i, b in enumerate(bus_indices)}
        for src in bus_indices:
            if src not in G:
                continue
            lengths = nx.single_source_shortest_path_length(G, src)
            for tgt, d in lengths.items():
                if tgt in bus_to_pos:
                    dist_matrix[bus_to_pos[src], bus_to_pos[tgt]] = max(d, 1e-6)

    # --- Compute D_v for each bus ---
    D_v = np.zeros(n_bus)
    bus_to_pos = {b: i for i, b in enumerate(bus_indices)}

    for i, bus_i in enumerate(bus_indices):
        numer = 0.0
        denom = 0.0
        for (gen_bus, H) in gen_info:
            j = bus_to_pos.get(gen_bus)
            if j is None:
                continue
            d_ij = dist_matrix[i, j] if not use_topological else dist_matrix[i, j]
            if d_ij < 1e-12:
                d_ij = 1e-6
            inv_d = 1.0 / d_ij
            # Use |B_ij| as additional weight when available
            if not use_topological:
                B_ji = np.abs(B[j, i]) if j < B.shape[0] and i < B.shape[1] else 1.0
            else:
                B_ji = 1.0
            w = inv_d * B_ji
            numer += (1.0 / H) * w
            denom += w
        D_v[i] = numer / denom if denom > 1e-12 else 0.0

    return D_v


# ===================================================================
#  Function 6 — Fault Severity Index (R)
# ===================================================================
def compute_fault_severity_index(D, L, I, method="equal"):
    """Compute composite Fault Severity Index from D, L, I.

    Physical meaning
    ----------------
    R aggregates three complementary vulnerability perspectives:
      • D — how exposed the bus is to frequency disturbances,
      • L — how tightly coupled the bus's power flows are,
      • I — how close the bus is to voltage collapse.
    A higher R indicates a bus that is simultaneously vulnerable across
    multiple failure modes (frequency, coupling, voltage) and is therefore
    the most likely to experience or propagate a cascading failure.

    Parameters
    ----------
    D, L, I : np.ndarray of shape (n_bus,)
    method : str, 'equal' or 'entropy'

    Returns
    -------
    R : np.ndarray of shape (n_bus,)
    weights : tuple of 3 floats
    """
    def _minmax(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    D_n = _minmax(D)
    L_n = _minmax(L)
    I_n = _minmax(I)

    if method == "entropy":
        stds = np.array([D_n.std(), L_n.std(), I_n.std()])
        total = stds.sum()
        if total < 1e-12:
            weights = (1 / 3, 1 / 3, 1 / 3)
        else:
            weights = tuple((stds / total).tolist())
    else:
        weights = (1 / 3, 1 / 3, 1 / 3)

    R = weights[0] * D_n + weights[1] * L_n + weights[2] * I_n
    return R, weights


# ===================================================================
#  Function 7 — Nodal Indices Dashboard
# ===================================================================
def plot_node_indices(net, D, L, I, R, timestep_label):
    """Plot a 2×2 dashboard of per-bus vulnerability indices.

    Physical meaning
    ----------------
    A quick visual summary of which buses are most vulnerable and why.
    The Fault Severity subplot uses a traffic-light colour scheme so that
    operators can immediately identify buses requiring intervention.

    Parameters
    ----------
    net : pandapowerNet
    D, L, I, R : np.ndarray of shape (n_bus,)
    timestep_label : str or int
    """
    n_bus = len(net.bus)
    bus_labels = [str(net.bus.at[b, "name"]) if "name" in net.bus.columns else str(b)
                  for b in net.bus.index]
    x = np.arange(n_bus)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (1,1) Frequency Response Index
    axes[0, 0].bar(x, D, color="steelblue", alpha=0.85)
    axes[0, 0].set_title("Frequency Response Index (D)", fontsize=11)
    axes[0, 0].set_ylabel("D value")

    # (1,2) Power Flow Coupling Index
    axes[0, 1].bar(x, L, color="darkorange", alpha=0.85)
    axes[0, 1].set_title("Power Flow Coupling Index (L)", fontsize=11)
    axes[0, 1].set_ylabel("L value")

    # (2,1) Voltage Vulnerability (1 - I): 1 = collapsed, 0 = healthy
    vulnerability = 1.0 - np.array(I)
    v_colors = []
    for vv in vulnerability:
        if vv > 0.7:
            v_colors.append("red")          # collapsed
        elif vv > 0.3:
            v_colors.append("orange")       # stressed
        else:
            v_colors.append("mediumseagreen")  # healthy
    axes[1, 0].bar(x, vulnerability, color=v_colors, alpha=0.85)
    axes[1, 0].axhline(y=0.7, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1, 0].set_title("Voltage Vulnerability (1 - I) at Attack Peak", fontsize=10)
    axes[1, 0].set_ylabel("1 - I (0=healthy, 1=collapsed)")
    axes[1, 0].set_ylim(-0.05, 1.15)

    # (2,2) Fault Severity Index — colour-coded
    colors = []
    for r in R:
        if r > 0.6:
            colors.append("red")
        elif r > 0.4:
            colors.append("orange")
        else:
            colors.append("green")
    axes[1, 1].bar(x, R, color=colors, alpha=0.85)
    axes[1, 1].set_title("Fault Severity Index (R)", fontsize=11)
    axes[1, 1].set_ylabel("R value")

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(bus_labels, rotation=60, fontsize=6, ha="right")
        ax.set_xlabel("Bus")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Nodal Analysis Indices at Timestep {timestep_label}", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(GRAPH_DIR, f"nodal_indices_{timestep_label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ===================================================================
#  Function 8 — SIR Propagation Analysis
# ===================================================================
def run_sir_analysis(timeseries_data, net, v0_snapshot, threshold_rate=0.15):
    """Run SIR-model analysis on cascading failure propagation.

    Physical meaning
    ----------------
    Each bus is treated as a node in an "epidemic" where:
      • S (Susceptible) — operating normally, not yet affected.
      • I (Infected) — experiencing a rapid increase in fault severity.
      • R (Recovered) — was infected but voltage is recovering and R is
        declining; remains in R permanently (like inoculation — the
        system operator has taken corrective action).
    The infection rate λ and recovery rate μ are estimated from the actual
    trajectories, then an SIR ODE model is fitted to compare predicted
    versus actual propagation dynamics.  The Maximum Infected Density Peak
    (MIDP) indicates the worst moment of the cascade.

    Parameters
    ----------
    timeseries_data : dict
    net : pandapowerNet
    v0_snapshot : np.ndarray
    threshold_rate : float — R-increase rate to trigger S→I transition

    Returns
    -------
    sir_results : dict with keys 'lambda', 'mu', 'midp_actual',
                  'midp_predicted', 'midp_time_actual', 'midp_time_predicted',
                  'final_infected_density', 'n_infected', 'n_recovered',
                  'bus_states'
    """
    vm_series = np.array(timeseries_data["vm_pu"])  # (T, n_bus)
    T, n_bus = vm_series.shape

    # Recompute R for every timestep
    R_series = np.zeros((T, n_bus))
    for t in range(T):
        # We need a "net" with res_bus populated. We'll reconstruct from stored data.
        vm_t = vm_series[t]
        # Voltage offset index — I=0 when at/below collapse, I=1 when healthy
        vcr = 0.9
        numer = np.maximum(vm_t - vcr, 0.0)
        denom = np.abs(v0_snapshot - vcr)
        denom[denom < 1e-12] = 1e-12
        I_v = np.clip(numer / denom, 0, 1)

        # For L and D we use the final network state (approximation — these
        # structural indices change slowly relative to voltage dynamics)
        if t == 0:
            D_v = compute_frequency_response_index(net)
            L_v = compute_power_flow_coupling_index(net)
        # Keep D_v, L_v constant across timesteps (structural property)

        R_t, _ = compute_fault_severity_index(D_v, L_v, I_v, method="equal")
        R_series[t] = R_t

    # --- Auto-tune SIR threshold from actual dynamics ---
    # Compute delta_R across all buses and timesteps
    dR = np.diff(R_series, axis=0)  # (T-1, n_bus)
    dR_std = dR.std()
    adaptive_threshold = max(0.02, min(0.5, 0.3 * dR_std))
    print(f"  SIR adaptive threshold: {adaptive_threshold:.4f} (dR_std={dR_std:.4f})")

    # --- SIR State Machine (cumulative threshold for staircase) ---
    # Instead of delta_R, use the cumulative R value itself with a
    # time-varying threshold.  At each timestep, a bus transitions S→I
    # when its R exceeds a threshold computed from the distribution of
    # R across all buses.  This produces a staircase because each
    # escalation stage pushes more buses above the threshold.
    states = np.full((T, n_bus), "S", dtype="U1")

    # R baseline (pre-attack): use median + 1 std as the infection threshold
    R_pre = R_series[:max(1, int(T * 0.2))].mean(axis=0)  # first 20% of steps
    R_baseline_median = np.median(R_pre)
    R_baseline_std = R_pre.std() if R_pre.std() > 0.001 else 0.05

    # Per-bus threshold: bus crosses into I when R_t exceeds its own
    # pre-attack R by more than a graduated amount
    for b in range(n_bus):
        r_thresh = R_pre[b] + R_baseline_std * 2.0  # 2-sigma above pre-attack
        for t in range(1, T):
            prev = states[t - 1, b]
            if prev == "R":
                states[t, b] = "R"
                continue
            r_now = R_series[t, b]
            if prev == "S":
                if r_now > r_thresh:
                    states[t, b] = "I"
                else:
                    states[t, b] = "S"
            elif prev == "I":
                # Recovery: R declining significantly OR voltage recovered
                delta_R = R_series[t, b] - R_series[t - 1, b]
                if (delta_R < -R_baseline_std * 0.5) or (vm_series[t, b] > 0.95 and delta_R < 0):
                    states[t, b] = "R"
                else:
                    states[t, b] = "I"

    # Density curves
    s_density = np.array([(states[t] == "S").sum() / n_bus for t in range(T)])
    i_density = np.array([(states[t] == "I").sum() / n_bus for t in range(T)])
    r_density = np.array([(states[t] == "R").sum() / n_bus for t in range(T)])

    # Estimate lambda and mu
    # lambda: mean rate of S→I transitions during spreading phase
    si_transitions = []
    ir_transitions = []
    for t in range(1, T):
        n_si = sum(1 for b in range(n_bus) if states[t - 1, b] == "S" and states[t, b] == "I")
        n_s = max((states[t - 1] == "S").sum(), 1)
        n_i = max((states[t - 1] == "I").sum(), 1)
        if n_si > 0:
            si_transitions.append(n_si / (n_s * i_density[t - 1] if i_density[t - 1] > 0 else 1))
        n_ir = sum(1 for b in range(n_bus) if states[t - 1, b] == "I" and states[t, b] == "R")
        if n_ir > 0:
            ir_transitions.append(n_ir / n_i)

    lam = np.mean(si_transitions) if si_transitions else 0.1
    mu = np.mean(ir_transitions) if ir_transitions else 0.05

    # Clamp to reasonable range
    lam = np.clip(lam, 0.01, 5.0)
    mu = np.clip(mu, 0.01, 5.0)

    # --- Fit SIR ODE ---
    def sir_ode(y, t_val, lam_, mu_):
        s, i, r = y
        dsdt = -lam_ * s * i
        didt = lam_ * s * i - mu_ * i
        drdt = mu_ * i
        return [dsdt, didt, drdt]

    # Find first timestep where infection appears
    first_inf_t = 0
    for tt in range(T):
        if i_density[tt] > 0:
            first_inf_t = max(0, tt - 1)
            break

    # Seed the ODE from just before first infection
    s0 = s_density[first_inf_t]
    i0 = max(i_density[first_inf_t], 1 / n_bus)
    r0 = r_density[first_inf_t]
    y0 = [s0, i0, r0]

    # Run ODE from first_inf_t onward
    n_ode_steps = T - first_inf_t
    t_eval = np.linspace(0, n_ode_steps - 1, n_ode_steps)
    sol = odeint(sir_ode, y0, t_eval, args=(lam, mu))

    # Build full prediction array (zeros before first infection)
    i_pred = np.zeros(T)
    i_pred[first_inf_t:] = sol[:, 1]

    # MIDP
    midp_actual = i_density.max()
    midp_pred = i_pred.max()
    midp_time_actual = int(np.argmax(i_density))
    midp_time_pred = int(np.argmax(i_pred))

    # Final state counts
    final_states = states[-1]
    n_infected = int((final_states == "I").sum())
    n_recovered = int((final_states == "R").sum())
    final_i_density = i_density[-1]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    timesteps = np.array(timeseries_data["timesteps"])

    ax.plot(timesteps, i_density, "o-", color="crimson", linewidth=2, label="Actual Infected Density")
    ax.plot(timesteps, i_pred, "s--", color="royalblue", linewidth=2, label="SIR Model Prediction")
    ax.plot(timesteps[midp_time_actual], midp_actual, "*", color="crimson", markersize=18, label=f"Actual MIDP = {midp_actual:.3f}")
    ax.plot(timesteps[midp_time_pred], midp_pred, "*", color="royalblue", markersize=18, label=f"Predicted MIDP = {midp_pred:.3f}")

    ax.set_title("SIR Cascading Failure Propagation", fontsize=14)
    ax.set_xlabel("Timestep (s)")
    ax.set_ylabel("Infected Density i(t)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(GRAPH_DIR, "sir_propagation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved {path}")

    # --- Console summary ---
    print(f"\n  SIR Model Estimates:")
    print(f"    lambda (infection rate) = {lam:.4f}")
    print(f"    mu (recovery rate)  = {mu:.4f}")
    print(f"    Actual  MIDP = {midp_actual:.4f}  at timestep {midp_time_actual}")
    print(f"    Predicted MIDP = {midp_pred:.4f}  at timestep {midp_time_pred}")

    return {
        "lambda": lam,
        "mu": mu,
        "midp_actual": midp_actual,
        "midp_predicted": midp_pred,
        "midp_time_actual": midp_time_actual,
        "midp_time_predicted": midp_time_pred,
        "final_infected_density": final_i_density,
        "n_infected": n_infected,
        "n_recovered": n_recovered,
        "bus_states": states,
    }


# ===================================================================
#  Master Orchestrator
# ===================================================================
def run_full_physics_analysis(net, timeseries_data, v0_snapshot,
                              attack_start_step, attack_end_step,
                              attacked_bus_indices):
    """Run the complete physics-level nodal analysis pipeline.

    This is the single entry-point for external callers.  It executes all
    eight analysis functions in order, then prints a human-readable summary
    report of the cascade propagation characteristics.

    Parameters
    ----------
    net : pandapowerNet — final network state (post-simulation)
    timeseries_data : dict — collected during simulation
    v0_snapshot : np.ndarray — pre-fault bus voltages
    attack_start_step : int
    attack_end_step : int
    attacked_bus_indices : list[int]
    """
    print("\n" + "=" * 65)
    print("  PHYSICS-LEVEL NODAL ANALYSIS -- CASCADING FAILURE REPORT")
    print("=" * 65)

    # 1. Voltage Timeseries
    print("\n[1/8] Plotting voltage timeseries...")
    plot_voltage_timeseries(timeseries_data, net, attack_start_step, attack_end_step)

    # 2. Branch Power
    print("[2/8] Plotting branch power flows...")
    plot_branch_power(timeseries_data, net, attacked_bus_indices,
                      attack_start_step, attack_end_step)

    # 3-6. Compute indices using attack-peak state (not post-recovery)
    # Find the timestep with lowest mean voltage (worst moment of attack)
    vm_array = np.array(timeseries_data["vm_pu"])
    attack_mask = np.array(timeseries_data["attack_active"])
    if attack_mask.any():
        attack_vm = vm_array[attack_mask]
        worst_idx = np.argmin(attack_vm.mean(axis=1))
        vm_at_peak = attack_vm[worst_idx]
    else:
        vm_at_peak = vm_array[-1]

    print("[3/8] Computing Voltage Offset Index (at attack peak)...")
    # I = max(0, V - Vcr) / (V0 - Vcr): 0 = at/below collapse, 1 = healthy
    vcr = 0.9
    numer_i = np.maximum(vm_at_peak - vcr, 0.0)
    denom_i = np.abs(v0_snapshot - vcr)
    denom_i[denom_i < 1e-12] = 1e-12
    I_v = np.clip(numer_i / denom_i, 0.0, 1.0)
    print(f"  I_v range: [{I_v.min():.3f}, {I_v.max():.3f}] | {(I_v < 0.01).sum()} buses collapsed")

    print("[4/8] Computing Power Flow Coupling Index...")
    L_v = compute_power_flow_coupling_index(net)

    print("[5/8] Computing Frequency Response Index...")
    D_v = compute_frequency_response_index(net)

    weight_method = "entropy"
    print(f"[6/8] Computing Fault Severity Index (method={weight_method})...")
    R_v, weights = compute_fault_severity_index(D_v, L_v, I_v, method=weight_method)

    # 7. Plot indices dashboard
    print("[7/8] Plotting nodal indices dashboard...")
    plot_node_indices(net, D_v, L_v, I_v, R_v, "post_attack")

    # 8. SIR Analysis
    print("[8/8] Running SIR propagation analysis...")
    sir = run_sir_analysis(timeseries_data, net, v0_snapshot)

    # ---- Summary Report ----
    top5_idx = np.argsort(R_v)[-5:][::-1]
    bus_names = net.bus["name"].values if "name" in net.bus.columns else net.bus.index.values

    print("\n" + "-" * 65)
    print("  SUMMARY REPORT")
    print("-" * 65)
    print(f"  Buses that reached INFECTED state : {sir['n_infected'] + sir['n_recovered']}")
    print(f"  Buses that RECOVERED              : {sir['n_recovered']}")
    print(f"  Final Infected Density             : {sir['final_infected_density']:.4f}")
    print(f"  Maximum Infected Density Peak      : {sir['midp_actual']:.4f}  (at t={sir['midp_time_actual']})")
    print(f"  Estimated lambda (infection rate)   : {sir['lambda']:.4f}")
    print(f"  Estimated mu (recovery rate)        : {sir['mu']:.4f}")
    print(f"  Weight method                      : {weight_method}")
    print(f"  Weights (D, L, I)                  : ({weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f})")
    print(f"\n  Top 5 Highest Fault Severity Nodes:")
    for rank, idx in enumerate(top5_idx, 1):
        print(f"    {rank}. Bus '{bus_names[idx]}' -- R = {R_v[idx]:.4f}")

    print("-" * 65)
    print("  Analysis complete. Graphs saved to graphs/ folder.")
    print("=" * 65)
