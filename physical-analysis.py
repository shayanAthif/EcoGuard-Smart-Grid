"""
IEEE 39-Bus New England System — Scenario 8 Cyber-Physical Attack
Full Newton-Raphson AC Power Flow (no external library)
Xu et al. (2024) CSEE Journal — Epidemic Model for Cascading Failure

Power flow:  Full AC Newton-Raphson, vectorized Jacobian (converges in ~12 iters)
Data:        IEEE 39-bus standard (Pai 1989 / MATPOWER case39)
Attack:      Scenario 8: ransomware(3,9,15) + worms(6,12)
Outputs:     4 publication-quality graphs
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.integrate import odeint
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — IEEE 39-BUS DATA  (Pai 1989 / MATPOWER case39)
# ═══════════════════════════════════════════════════════════════════════════

SBASE = 100.0   # MVA base
N_BUS = 39

# Bus data: [id, type(1=PQ,2=PV,3=slack), Pd_MW, Qd_MVAr, Vm_init, Va_init_deg]
BUS_DATA = np.array([
    [ 1, 1,    0.0,    0.0, 1.0393, -13.536],
    [ 2, 1,    0.0,    0.0, 1.0484, -12.286],
    [ 3, 1,  322.0,    2.4, 1.0307, -12.356],
    [ 4, 1,  500.0,  184.0, 1.0039, -12.988],
    [ 5, 1,    0.0,    0.0, 1.0054, -13.359],
    [ 6, 1,    0.0,    0.0, 1.0082, -14.038],
    [ 7, 1,  233.8,   84.0, 0.9970, -13.855],
    [ 8, 1,  522.0,  176.6, 0.9960, -14.048],
    [ 9, 1,    0.0,    0.0, 1.0282, -14.649],
    [10, 1,    0.0,    0.0, 1.0178, -14.958],
    [11, 1,    0.0,    0.0, 1.0133, -14.793],
    [12, 1,    7.5,   88.0, 1.0003, -14.886],
    [13, 1,    0.0,    0.0, 1.0143, -14.362],
    [14, 1,    0.0,    0.0, 1.0116, -14.221],
    [15, 1,  320.0,  153.0, 1.0155, -14.916],
    [16, 1,  329.0,   32.3, 1.0323, -14.600],
    [17, 1,    0.0,    0.0, 1.0343, -15.546],
    [18, 1,  158.0,   30.0, 1.0318, -16.456],
    [19, 1,    0.0,    0.0, 1.0505, -13.508],
    [20, 1,  628.0,  103.0, 0.9912, -14.104],
    [21, 1,  274.0,  115.0, 1.0323, -14.100],
    [22, 1,    0.0,    0.0, 1.0501, -12.685],
    [23, 1,  247.5,   84.6, 1.0451, -13.440],
    [24, 1,  308.6,  -92.2, 1.0373, -14.064],
    [25, 1,  224.0,   47.2, 1.0576, -13.406],
    [26, 1,  139.0,   17.0, 1.0521, -14.459],
    [27, 1,  281.0,   75.5, 1.0377, -15.241],
    [28, 1,  206.0,   27.6, 1.0501, -13.172],
    [29, 1,  283.5,   26.9, 1.0497, -13.322],
    [30, 2,    0.0,    0.0, 1.0475,  -7.366],
    [31, 3,    9.2,    4.6, 0.9820,   0.000],   # slack bus
    [32, 2,    0.0,    0.0, 0.9831,  -1.474],
    [33, 2,    0.0,    0.0, 0.9972,  -0.720],
    [34, 2,    0.0,    0.0, 1.0123,  -1.826],
    [35, 2,    0.0,    0.0, 1.0493,   1.247],
    [36, 2,    0.0,    0.0, 1.0635,   6.385],
    [37, 2,    0.0,    0.0, 1.0278,  -6.693],
    [38, 2,    0.0,    0.0, 1.0265,  -8.015],
    [39, 2, 1104.0,  250.0, 1.0300,  -9.424],
])

# Generator active power dispatch and voltage setpoints
GEN_PG = {30:250.0, 31:677.87, 32:650.0, 33:632.0, 34:508.0,
          35:650.0, 36:560.0,  37:540.0, 38:830.0, 39:1000.0}
GEN_VS = {30:1.0475, 31:0.9820, 32:0.9831, 33:0.9972, 34:1.0123,
          35:1.0493, 36:1.0635, 37:1.0278, 38:1.0265, 39:1.0300}
GEN_H  = {30:0.0,   31:5.00,  32:4.33,  33:4.47,  34:3.57,
          35:4.33,  36:3.50,  37:5.12,  38:7.06,  39:0.0}

SLACK_BUS = 31   # bus 31 (index 30)

# Full IEEE 39 branch set (45 branches):
# [from, to, R_pu, X_pu, B_pu, tap (0=line, >0=transformer)]
BRANCH_DATA = np.array([
    [ 1,  2, 0.0035, 0.0411, 0.6987, 0.000],
    [ 1, 39, 0.0010, 0.0250, 0.7500, 0.000],
    [ 2,  3, 0.0013, 0.0151, 0.2572, 0.000],
    [ 2, 25, 0.0070, 0.0086, 0.1460, 0.000],
    [ 2, 30, 0.0000, 0.0181, 0.0000, 1.025],   # transformer
    [ 3,  4, 0.0013, 0.0213, 0.2214, 0.000],
    [ 3, 18, 0.0011, 0.0133, 0.2138, 0.000],
    [ 4,  5, 0.0008, 0.0128, 0.1342, 0.000],
    [ 4, 14, 0.0008, 0.0129, 0.1382, 0.000],
    [ 5,  6, 0.0002, 0.0026, 0.0434, 0.000],
    [ 5,  8, 0.0008, 0.0112, 0.1476, 0.000],
    [ 6,  7, 0.0006, 0.0092, 0.1130, 0.000],
    [ 6, 11, 0.0007, 0.0082, 0.1389, 0.000],
    [ 6, 31, 0.0000, 0.0250, 0.0000, 1.070],   # transformer
    [ 7,  8, 0.0004, 0.0046, 0.0780, 0.000],
    [ 8,  9, 0.0023, 0.0363, 0.3804, 0.000],
    [ 9, 39, 0.0010, 0.0250, 1.2000, 0.000],
    [10, 11, 0.0004, 0.0043, 0.0729, 0.000],
    [10, 13, 0.0004, 0.0043, 0.0729, 0.000],
    [10, 32, 0.0000, 0.0200, 0.0000, 1.070],   # transformer
    [11, 12, 0.0016, 0.0435, 0.0000, 1.006],   # transformer
    [12, 13, 0.0016, 0.0435, 0.0000, 1.006],   # transformer
    [13, 14, 0.0009, 0.0101, 0.1723, 0.000],
    [14, 15, 0.0018, 0.0217, 0.3660, 0.000],   # ← tripped step 3
    [15, 16, 0.0009, 0.0094, 0.1710, 0.000],
    [16, 17, 0.0007, 0.0089, 0.1342, 0.000],
    [16, 19, 0.0016, 0.0195, 0.3040, 0.000],   # ← tripped step 9
    [16, 21, 0.0008, 0.0135, 0.2548, 0.000],
    [17, 18, 0.0007, 0.0082, 0.1319, 0.000],   # ← tripped step 15
    [17, 27, 0.0013, 0.0173, 0.3216, 0.000],
    [19, 20, 0.0007, 0.0138, 0.0000, 0.000],
    [19, 33, 0.0007, 0.0142, 0.0000, 1.070],   # transformer
    [20, 34, 0.0009, 0.0180, 0.0000, 1.009],   # transformer
    [21, 22, 0.0008, 0.0140, 0.2565, 0.000],
    [22, 23, 0.0006, 0.0096, 0.1846, 0.000],
    [22, 35, 0.0000, 0.0143, 0.0000, 1.025],   # transformer
    [23, 24, 0.0022, 0.0350, 0.3610, 0.000],
    [23, 36, 0.0005, 0.0272, 0.0000, 1.000],   # transformer
    [25, 26, 0.0032, 0.0323, 0.5310, 0.000],
    [25, 37, 0.0006, 0.0232, 0.0000, 1.025],   # transformer
    [26, 27, 0.0014, 0.0147, 0.2396, 0.000],
    [26, 28, 0.0043, 0.0474, 0.7802, 0.000],
    [26, 29, 0.0057, 0.0625, 1.0290, 0.000],
    [28, 29, 0.0014, 0.0151, 0.2490, 0.000],
    [29, 38, 0.0008, 0.0156, 0.0000, 1.025],   # transformer
])

N_BRANCH = len(BRANCH_DATA)

# Bus type sets (0-indexed)
SLACK_IDX = SLACK_BUS - 1
PV_IDX    = [b-1 for b in GEN_VS if b != SLACK_BUS]
PQ_IDX    = [b for b in range(N_BUS) if b != SLACK_IDX and b not in PV_IDX]

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — NEWTON-RAPHSON AC POWER FLOW
# ═══════════════════════════════════════════════════════════════════════════

def build_ybus(active_set):
    """Build complex admittance matrix for the active branch set."""
    Y = np.zeros((N_BUS, N_BUS), dtype=complex)
    for k in active_set:
        f  = int(BRANCH_DATA[k, 0]) - 1
        t  = int(BRANCH_DATA[k, 1]) - 1
        R  = BRANCH_DATA[k, 2]; X = BRANCH_DATA[k, 3]
        B  = BRANCH_DATA[k, 4]; tap = BRANCH_DATA[k, 5]

        ys = 1/complex(R, X) if abs(complex(R, X)) > 1e-12 else complex(0, 1e4)
        tk = complex(tap, 0) if tap > 1e-6 else complex(1, 0)
        tm2 = abs(tk) ** 2
        ysh = complex(0, B/2)

        Y[f, f] += ys/tm2 + ysh
        Y[t, t] += ys     + ysh
        Y[f, t] -= ys / np.conj(tk)
        Y[t, f] -= ys / tk
    return Y


def run_nr(Pd_pu, Qd_pu, active_set, tol=1e-8, max_iter=50):
    """
    Newton-Raphson AC power flow.

    State: x = [Δθ_{non-slack}, Δ|V|/|V|_{PQ}]
    Mismatch: ΔP for all non-slack, ΔQ for PQ only.

    Vectorized Jacobian:
      J11 = Re(dS/dθ),  J12 = Re(dS/d|V|·|V|)
      J21 = Im(dS/dθ),  J22 = Im(dS/d|V|·|V|)

    Returns V_mag (p.u.), P_flow (MW), Q_flow (MVAr), converged, n_iter
    """
    ns  = sorted(PV_IDX + PQ_IDX)   # non-slack
    n_th = len(ns); n_pq = len(PQ_IDX); n_st = n_th + n_pq

    Y = build_ybus(active_set)

    # Scheduled injections (p.u.)
    Pg_pu = np.zeros(N_BUS)
    for bus, pg in GEN_PG.items():
        Pg_pu[bus-1] += pg / SBASE

    Psch = Pg_pu - Pd_pu   # for all non-slack buses
    Qsch = -Qd_pu           # for PQ buses only (generators hold V, Q is free)

    # Initial voltages
    Vm = BUS_DATA[:, 4].copy()
    Va = BUS_DATA[:, 5] * np.pi / 180.0
    for bus, vs in GEN_VS.items():
        Vm[bus-1] = vs

    converged = False
    for it in range(max_iter):
        Vc   = Vm * np.exp(1j * Va)
        Ibus = Y @ Vc
        Sbus = Vc * np.conj(Ibus)

        # Mismatches
        dP   = np.array([Psch[i] - Sbus[i].real for i in ns])
        dQ   = np.array([Qsch[i] - Sbus[i].imag for i in PQ_IDX])
        mis  = np.concatenate([dP, dQ])
        norm = np.max(np.abs(mis))

        if norm < tol:
            converged = True
            break

        # Vectorized Jacobian blocks
        Vd   = np.diag(Vc)
        Vdn  = np.diag(Vc / (np.abs(Vc) + 1e-15))
        Idc  = np.diag(np.conj(Ibus))

        dSth = 1j * Vd @ (Idc - np.conj(Y @ Vd))
        dSVm = Vdn @ np.conj(Idc + Y @ Vd)

        J = np.zeros((n_st, n_st))
        for ii, i in enumerate(ns):
            for jj, j in enumerate(ns):         J[ii,        jj       ] = dSth[i,j].real
            for jj, j in enumerate(PQ_IDX):     J[ii,        n_th+jj  ] = dSVm[i,j].real
        for ii, i in enumerate(PQ_IDX):
            for jj, j in enumerate(ns):         J[n_th+ii,   jj       ] = dSth[i,j].imag
            for jj, j in enumerate(PQ_IDX):     J[n_th+ii,   n_th+jj  ] = dSVm[i,j].imag

        try:
            dx = np.linalg.solve(J, mis)
        except np.linalg.LinAlgError:
            break

        for ii, i in enumerate(ns):       Va[i] += dx[ii]
        for ii, i in enumerate(PQ_IDX):   Vm[i] += dx[n_th+ii] * Vm[i]
        for bus, vs in GEN_VS.items():     Vm[bus-1] = vs
        Vm = np.clip(Vm, 0.75, 1.25)

    # Branch flows
    Vc = Vm * np.exp(1j * Va)
    Pf = np.zeros(N_BRANCH); Qf = np.zeros(N_BRANCH)
    for k in active_set:
        f  = int(BRANCH_DATA[k,0])-1; t = int(BRANCH_DATA[k,1])-1
        R  = BRANCH_DATA[k,2]; X = BRANCH_DATA[k,3]
        B  = BRANCH_DATA[k,4]; tap = BRANCH_DATA[k,5]
        ys = 1/complex(R,X) if abs(complex(R,X))>1e-12 else complex(0,1e4)
        tk = complex(tap,0) if tap>1e-6 else complex(1,0)
        tm2 = abs(tk)**2
        Vf_t = Vc[f]/tk; Vt_v = Vc[t]
        I_ft = (Vf_t - Vt_v)*ys + Vf_t*complex(0,B/2)
        S_ft = Vf_t * np.conj(I_ft) * SBASE
        Pf[k] = S_ft.real; Qf[k] = S_ft.imag

    return Vm, Pf, Qf, converged, it+1


# ── Validate base case ────────────────────────────────────────────────────
print("=" * 60)
print("IEEE 39-Bus — Newton-Raphson AC Power Flow Validation")
print("=" * 60)

Pd0 = BUS_DATA[:,2] / SBASE
Qd0 = BUS_DATA[:,3] / SBASE
Vm0, Pf0, Qf0, conv0, itr0 = run_nr(Pd0, Qd0, set(range(N_BRANCH)))

print(f"Base case: converged={conv0}, iterations={itr0}")
print(f"V range: [{Vm0.min():.4f}, {Vm0.max():.4f}] p.u.")
print(f"Bus 20: {Vm0[19]:.4f}  (MATPOWER: 0.9912, Δ={abs(Vm0[19]-0.9912):.4f})")
print(f"Bus  4: {Vm0[3]:.4f}  (MATPOWER: 1.0039, Δ={abs(Vm0[3]-1.0039):.4f})")
print(f"Bus  7: {Vm0[6]:.4f}  (MATPOWER: 0.9970, Δ={abs(Vm0[6]-0.9970):.4f})")
print()

V_BASE = Vm0.copy()   # TRUE power flow base case

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — SCENARIO 8
# ═══════════════════════════════════════════════════════════════════════════

ATTACKS = ['dos','fuzzers','worms','exploits','data_injection','ransomware']
N_STEPS = 15

SEQUENCE = []
for i in range(1, N_STEPS+1):
    if i % 3 == 0:
        SEQUENCE.append(ATTACKS[(2 + i) % 6])
    else:
        SEQUENCE.append('normal')

print("Scenario 8 sequence:")
for i, a in enumerate(SEQUENCE):
    print(f"  Step {i+1:2d}: {a}" + ("  ◄" if a != 'normal' else ""))
print()

# Ransomware targets (branch index, shed bus 1-indexed)
RW = {3: (23, 20), 9: (26, 15), 15: (28, 23)}  # step: (branch_idx, shed_bus)
# Worms targets (bus 1-indexed, overload factor)
WM = {6: (20, 3.0), 12: (23, 3.0)}

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — RUN 15-STEP SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

print("Running 15-step simulation (Newton-Raphson at each step)...")
print("-" * 60)

V_sim = np.zeros((N_BUS,    N_STEPS))
P_sim = np.zeros((N_BRANCH, N_STEPS))
Q_sim = np.zeros((N_BRANCH, N_STEPS))
CONV  = np.zeros(N_STEPS, dtype=bool)

active = set(range(N_BRANCH))   # branches in service (shrinks with ransomware)
Pd_cur = BUS_DATA[:,2] / SBASE
Qd_cur = BUS_DATA[:,3] / SBASE
worm_saved = {}

for t in range(N_STEPS):
    step = t + 1; atk = SEQUENCE[t]

    if atk == 'ransomware':
        br_idx, shed_bus = RW[step]
        active.discard(br_idx)
        b = shed_bus - 1
        Pd_cur[b] = 0.0; Qd_cur[b] = 0.0
        br = BRANCH_DATA[br_idx]
        print(f"  Step {step:2d} RANSOMWARE: trip branch {br_idx} "
              f"({int(br[0])}→{int(br[1])}), shed bus {shed_bus}")

    elif atk == 'worms':
        bus_id, factor = WM[step]
        b = bus_id - 1
        worm_saved[b] = (Pd_cur[b], Qd_cur[b])
        Pd_cur[b] = BUS_DATA[b,2]/SBASE * factor
        Qd_cur[b] = BUS_DATA[b,3]/SBASE * factor
        print(f"  Step {step:2d} WORMS:      bus {bus_id} load "
              f"{BUS_DATA[b,2]:.0f}→{BUS_DATA[b,2]*factor:.0f} MW (×{factor:.0f})")

    else:  # normal — restore worm overloads
        for b, (pd, qd) in worm_saved.items():
            Pd_cur[b] = pd; Qd_cur[b] = qd
        worm_saved.clear()

    Vm, Pf, Qf, conv, iters = run_nr(Pd_cur.copy(), Qd_cur.copy(), active)

    CONV[t]     = conv
    V_sim[:,t]  = Vm
    P_sim[:,t]  = Pf
    Q_sim[:,t]  = Qf

    vmin_b = Vm.argmin() + 1
    print(f"    {'✓' if conv else '○'} converged={conv} ({iters} iter) | "
          f"V_min={Vm.min():.4f} p.u. @ bus {vmin_b} | "
          f"violations<0.95: {(Vm<0.95).sum()}")

print("-" * 60)
print(f"Converged: {CONV.sum()}/{N_STEPS} steps")
print(f"V range: [{V_sim.min():.4f}, {V_sim.max():.4f}] p.u.")
print(f"Total violations <0.95 p.u.: {(V_sim<0.95).sum()}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — NODAL INDICES (Xu et al. Eq. 19–33)
# ═══════════════════════════════════════════════════════════════════════════

def get_adj(active_set):
    adj = {b: [] for b in range(1, N_BUS+1)}
    for k in active_set:
        f = int(BRANCH_DATA[k,0]); t = int(BRANCH_DATA[k,1])
        X = BRANCH_DATA[k,3]
        B = 1/X if X > 1e-10 else 1e4
        adj[f].append((t, B)); adj[t].append((f, B))
    return adj

def bottleneck_path(src, dst, adj):
    """Max bottleneck susceptance path (BFS)."""
    if src == dst: return float('inf')
    best = {src: float('inf')}
    q = [src]
    while q:
        c = q.pop(0)
        for nb, B in adj[c]:
            nb_val = min(best[c], B)
            if nb not in best or nb_val > best[nb]:
                best[nb] = nb_val; q.append(nb)
    return best.get(dst, 0.0)

def compute_D(active_set):
    """Frequency Response Index — Eq. 19."""
    adj = get_adj(active_set)
    D = np.zeros(N_BUS)
    gens = list(GEN_H.keys())
    for b in range(1, N_BUS+1):
        if not adj[b]: continue
        adj_sum = sum(s for _,s in adj[b]) + 1e-10
        num = den = 0.0
        for g in gens:
            pb = bottleneck_path(b, g, adj)
            if pb > 0 and np.isfinite(pb):
                r = pb / adj_sum
                if GEN_H[g] > 0: num += (1/GEN_H[g]) * r * pb
                den += pb
        D[b-1] = num/den if den > 1e-10 else 0.0
    D = np.nan_to_num(D)
    return D / D.max() * 0.40 if D.max() > 1e-10 else D

def compute_L(step_idx):
    """Power Flow Coupling Index — Eqs. 20–26."""
    S = np.abs(P_sim[:,step_idx] + 1j*Q_sim[:,step_idx])
    Lo = np.zeros(N_BUS); Li = np.zeros(N_BUS)
    for i in range(N_BUS):
        bus = i+1
        ob = [k for k in range(N_BRANCH) if int(BRANCH_DATA[k,0])==bus]
        ib = [k for k in range(N_BRANCH) if int(BRANCH_DATA[k,1])==bus]
        if ob:
            Si_tot = sum(S[k] for k in ob) + 1e-10
            w = np.zeros(len(ob))
            for idx, k in enumerate(ob):
                j = int(BRANCH_DATA[k,1])-1
                Sij = S[k]
                in_j = [m for m in range(N_BRANCH) if int(BRANCH_DATA[m,1])==(j+1)]
                Sj_tot = sum(S[m] for m in in_j) + 1e-10
                w[idx] = (Sij/Sj_tot) / (Sij/Si_tot + 1e-10)
            ws = w.sum()+1e-10; p = np.clip(w/ws, 1e-10, None)
            Lo[i] = max((1-np.sum(p*np.log(p))) * ws, 0)
        if ib:
            w = np.array([S[k] for k in ib]); ws = w.sum()+1e-10
            p = np.clip(w/ws, 1e-10, None)
            Li[i] = max((1-np.sum(p*np.log(p))) * ws, 0)
    return 0.5*Lo + 0.5*Li

def compute_I(V_actual, V_crit=0.80):
    """Voltage Offset Index — Eq. 27."""
    denom = np.abs(V_BASE - V_crit) + 1e-10
    return np.clip(np.abs((V_actual - V_crit) / denom), 0, 1)

def compute_R(D, L, I):
    """Fault Severity via Game-AHP Entropy — Eqs. 28–33."""
    def n01(x): r = x.max()-x.min(); return (x-x.min())/(r+1e-10)
    X  = np.column_stack([n01(D), n01(L), n01(np.clip(1-I,0,1))])
    W1 = np.array([0.2269, 0.5407, 0.2324])
    P  = np.clip(X/(X.sum(axis=0)+1e-10), 1e-10, None)
    E  = -np.sum(P*np.log(P), axis=0) / np.log(len(X))
    d  = 1-E; W2 = d/(d.sum()+1e-10)
    A  = np.array([[W1@W1, W1@W2],[W2@W1, W2@W2]])
    bv = np.array([W1@W1, W2@W2])
    try:    lam = np.linalg.solve(A, bv)
    except: lam = np.array([0.5, 0.5])
    ls = np.abs(lam)/(np.abs(lam).sum()+1e-10)
    Ws = (ls[0]*W1 + ls[1]*W2); Ws /= Ws.sum()+1e-10
    return np.clip(X@Ws, 0, None), Ws

# Compute at step 3 (first attack, index 2)
ATK_STEP_IDX = 2
active_post3 = set(range(N_BRANCH)) - {RW[3][0]}
print("Computing nodal indices at step 3...")
D_idx = compute_D(active_post3)
L_idx = compute_L(ATK_STEP_IDX)
I_idx = compute_I(V_sim[:, ATK_STEP_IDX])
R_idx, W_star = compute_R(D_idx, L_idx, I_idx)
print(f"  D: max={D_idx.max():.4f}, mean={D_idx.mean():.4f}")
print(f"  L: max={L_idx.max():.2f}, mean={L_idx.mean():.2f}")
print(f"  I: min={I_idx.min():.4f} (bus {I_idx.argmin()+1})")
print(f"  R: max={R_idx.max():.4f} (bus {R_idx.argmax()+1})")
print(f"  Weights: D={W_star[0]:.4f}, L={W_star[1]:.4f}, I={W_star[2]:.4f}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — SIR MODEL
# ═══════════════════════════════════════════════════════════════════════════

def classify_sir():
    active_t = set(range(N_BRANCH))
    R_all = np.zeros((N_BUS, N_STEPS))
    for t in range(N_STEPS):
        if t == 2:  active_t.discard(RW[3][0])
        if t == 8:  active_t.discard(RW[9][0])
        if t == 14: active_t.discard(RW[15][0])
        D_t = compute_D(active_t)
        L_t = compute_L(t)
        I_t = compute_I(V_sim[:,t])
        R_all[:,t], _ = compute_R(D_t, L_t, I_t)

    dR = np.diff(R_all, axis=1)
    base = np.nanmean(np.abs(dR[:, ATK_STEP_IDX]))
    if not (np.isfinite(base) and base > 1e-8):
        base = np.nanmean(np.abs(dR))
    thr = base * 1.1
    print(f"SIR threshold = {thr:.6f}")

    states = np.ones((N_BUS, N_STEPS), dtype=int)
    for t in range(1, N_STEPS):
        for b in range(N_BUS):
            dr = dR[b, t-1]; prev = states[b, t-1]
            if prev == 1:
                states[b, t] = 2 if abs(dr) > thr else 1
            elif prev == 2:
                states[b, t] = 3 if (dr < 0 and abs(dr) < thr*0.9) else 2
            else:
                states[b, t] = 3
    return states

def sir_ode(y, t, lam, mu):
    s,i,r = y
    return [-lam*s*i, lam*s*i-mu*i, mu*i]

def fit_sir_params(t_data, i_obs, s0, i0):
    def obj(p):
        if p[0]<=0 or p[1]<=0: return 1e10
        try:
            sol = odeint(sir_ode,[s0,i0,max(0,1-s0-i0)],t_data,
                        args=(p[0],p[1]),rtol=1e-8,atol=1e-10)
            return float(np.sum((sol[:,1]-i_obs)**2))
        except: return 1e10
    best = (1e10,[0.5,0.3])
    for l0 in [0.1,0.5,1.0,2.0,3.5,5.0]:
        for m0 in [0.05,0.1,0.3,0.5,0.8,1.5]:
            res = minimize(obj,[l0,m0],method='Nelder-Mead',
                          options={'xatol':1e-7,'fatol':1e-9,'maxiter':3000})
            if res.fun < best[0]: best=(res.fun,res.x)
    return max(best[1][0],0.01), max(best[1][1],0.01)

print("Running SIR classification...")
node_states  = classify_sir()
t_steps      = np.arange(1, N_STEPS+1, dtype=float)
i_dens       = (node_states==2).sum(0)/N_BUS
s_dens       = (node_states==1).sum(0)/N_BUS
r_dens       = (node_states==3).sum(0)/N_BUS

MIDP_actual  = i_dens.max()
MIDP_step    = i_dens.argmax()+1
s0 = s_dens[0]; i0 = max(i_dens[0], 1/N_BUS)
lam, mu      = fit_sir_params(t_steps, i_dens, s0, i0)
r0_v         = max(0, 1-s0-i0)
sol_ode      = odeint(sir_ode,[s0,i0,r0_v],t_steps,args=(lam,mu))
MIDP_pred    = sol_ode[:,1].max()
R0_val       = lam/mu
err_pct      = abs(MIDP_actual-MIDP_pred)/max(MIDP_actual,1e-10)*100

print(f"  λ={lam:.4f}, μ={mu:.4f}, R₀={R0_val:.3f}")
print(f"  MIDP actual={MIDP_actual:.4f} (step {MIDP_step}), predicted={MIDP_pred:.4f}")
print(f"  SIR fit error = {err_pct:.2f}%  (target: <5%)")
print()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.facecolor': '#f8f9fa', 'figure.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.35, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'legend.fontsize': 7.5, 'legend.framealpha': 0.88,
})

t_lbl    = [str(i+1) for i in range(N_STEPS)]
bus_ids  = np.arange(1, N_BUS+1)
gen_idx  = [g-1 for g in GEN_H]
load_idx = [b for b in range(N_BUS) if b not in gen_idx]

def shade(ax, yref=None):
    """Shade attack columns and label attack type."""
    for i, atk in enumerate(SEQUENCE):
        if atk != 'normal':
            col = '#c0392b' if atk=='ransomware' else '#d35400'
            ax.axvspan(i+0.6, i+1.4, color=col, alpha=0.13, zorder=0)
    if yref is not None:
        for i, atk in enumerate(SEQUENCE):
            if atk != 'normal':
                col = '#c0392b' if atk=='ransomware' else '#d35400'
                ax.text(i+1, yref, atk[:3].upper(),
                       ha='center', fontsize=6.5, color=col,
                       fontweight='bold', clip_on=False, va='bottom')

def bar_index(ax, vals, title, ylabel, base_col, top_n=5):
    colors = [base_col]*N_BUS
    top    = np.argsort(vals)[-top_n:]
    for idx in top: colors[idx] = '#c0392b'
    bars = ax.bar(bus_ids, vals, color=colors, alpha=0.75,
                 width=0.75, edgecolor='white', lw=0.3)
    for g in gen_idx:
        bars[g].set_edgecolor('#27ae60'); bars[g].set_linewidth(1.5)
    for idx in top:
        ax.text(idx+1, vals[idx]+vals.max()*0.015, str(idx+1),
               ha='center', va='bottom', fontsize=6.5,
               color='#c0392b', fontweight='bold')
    ax.axvspan(29.5, 39.5, color='#27ae60', alpha=0.05)
    ax.set_xticks(bus_ids[::3]); ax.set_xticklabels(bus_ids[::3], fontsize=7.5)
    ax.set_xlim(0.5, N_BUS+0.5)
    ax.set_xlabel('Bus No.', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, pad=6)


# ── GRAPH 1: Voltage Timeseries ──────────────────────────────────────────
print("Plotting Graph 1: Voltage Timeseries...")
fig1, (av, ad) = plt.subplots(2,1,figsize=(14,9),
                               gridspec_kw={'height_ratios':[3,1],'hspace':0.38})

for b in load_idx:
    av.plot(t_steps, V_sim[b,:], color='#5dade2', alpha=0.18, lw=0.8, zorder=1)

gcm = plt.cm.Greens(np.linspace(0.45,0.90,len(gen_idx)))
for k,b in enumerate(gen_idx):
    av.plot(t_steps, V_sim[b,:], color=gcm[k], alpha=0.55, lw=1.4, zorder=2)

HL = {15:('#e74c3c','Bus 15  (320 MW shed step 9)',   2.5,'o'),
      20:('#8e44ad','Bus 20  (628 MW — worm ×3 step 6)',2.5,'s'),
      23:('#e67e22','Bus 23  (247 MW shed step 15)',  2.5,'^'),
      31:('#1a5276','Bus 31  (slack generator)',       2.0,'D'),
      30:('#148f77','Bus 30  (wind farm, H=0)',        2.0,'P'),
      39:('#1f618d','Bus 39  (wind + 1104 MW load)',   2.0,'h')}
for bus,(col,lbl,lw,mk) in HL.items():
    av.plot(t_steps, V_sim[bus-1,:], color=col, lw=lw,
           marker=mk, ms=5, label=lbl, zorder=5)

av.axhspan(0.95,1.05, color='#27ae60', alpha=0.06, label='Normal band 0.95–1.05 p.u.')
av.axhline(0.95, color='#f39c12', lw=1.5, ls='--', alpha=0.85, label='V_min = 0.95 p.u.')
av.axhline(1.05, color='#f39c12', lw=1.0, ls='--', alpha=0.45)
av.axhline(0.80, color='#c0392b', lw=1.5, ls=':', alpha=0.80, label='V_critical = 0.80 p.u.')

shade(av)

vmin3 = V_sim[:,2].min(); b3 = V_sim[:,2].argmin()+1
vmin6 = V_sim[:,5].min(); b6 = V_sim[:,5].argmin()+1
av.annotate(f'Line 14→15 tripped (ransomware)\nV_min={vmin3:.4f} p.u. @ bus {b3}',
           xy=(3,vmin3), xytext=(4.2,vmin3-0.015), fontsize=7.5, color='#c0392b',
           arrowprops=dict(arrowstyle='->',color='#c0392b',lw=1.2),
           bbox=dict(boxstyle='round,pad=0.3',facecolor='#fdfafa',alpha=0.92))
av.annotate(f'300% load overload bus 20 (worms)\nV_min={vmin6:.4f} p.u. @ bus {b6}',
           xy=(6,vmin6), xytext=(7.3,vmin6-0.015), fontsize=7.5, color='#8e44ad',
           arrowprops=dict(arrowstyle='->',color='#8e44ad',lw=1.2),
           bbox=dict(boxstyle='round,pad=0.3',facecolor='#f5eef8',alpha=0.92))

ylims = [min(V_sim.min()-0.01,0.82), max(V_sim.max()+0.01,1.09)]
av.set_ylim(ylims)
shade(av, yref=ylims[1]+0.002)
av.set_xlim(0.5,N_STEPS+0.5)
av.set_xticks(t_steps)
av.set_xticklabels([f'{i+1}\n[{SEQUENCE[i][:3].upper()}]' for i in range(N_STEPS)],fontsize=7)
av.set_xlabel('Timestep  (attack type in brackets)')
av.set_ylabel('Voltage Magnitude (p.u.)')
av.set_title('Bus Voltage Profiles — IEEE 39 Scenario 8\n'
            'Newton-Raphson AC Power Flow  |  Blue=load buses, Green=generator buses',pad=8)
av.legend(loc='lower left', ncol=2, fontsize=7.5)

drops = np.array([(V_BASE-V_sim[:,t]).mean() for t in range(N_STEPS)])
ad.fill_between(t_steps,0,drops*100,where=drops>0, color='#e74c3c',alpha=0.5,label='Mean V drop')
ad.fill_between(t_steps,0,drops*100,where=drops<=0,color='#27ae60',alpha=0.4,label='Mean V rise')
ad.plot(t_steps,drops*100,'k-o',lw=1.5,ms=4,zorder=5)
shade(ad)
ad.set_xlim(0.5,N_STEPS+0.5)
ad.set_xticks(t_steps); ad.set_xticklabels(t_lbl,fontsize=8)
ad.set_ylabel('Mean ΔV (%)'); ad.set_xlabel('Timestep')
ad.set_title('System-Wide Mean Voltage Drop from Base-Case (NR) Solution',pad=5)
ad.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.3f}%'))
ad.legend(fontsize=7.5)

plt.savefig('graphs/graph1_voltage_timeseries.png',
           dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  Saved.")


# ── GRAPH 2: Branch Power Flow ───────────────────────────────────────────
print("Plotting Graph 2: Branch Power Flow...")
atk_buses = {15, 20, 23}
adj_br  = [k for k in range(N_BRANCH)
           if int(BRANCH_DATA[k,0]) in atk_buses or int(BRANCH_DATA[k,1]) in atk_buses]
var_br  = np.argsort(P_sim.max(1)-P_sim.min(1))[::-1][:15].tolist()
show_br = sorted(set(adj_br+var_br))[:18]

fig2,(ap,aq) = plt.subplots(2,1,figsize=(14,9),gridspec_kw={'hspace':0.42})
bcm = plt.cm.tab20(np.linspace(0,1,len(show_br)))

for k,br in enumerate(show_br):
    f=int(BRANCH_DATA[br,0]); t_=int(BRANCH_DATA[br,1])
    lbl=f'Br{br} ({f}→{t_})'
    lw=2.0 if br in adj_br else 1.0
    al=0.95 if br in adj_br else 0.55
    mk='o' if br in adj_br else None
    ap.plot(t_steps,P_sim[br,:],color=bcm[k],lw=lw,alpha=al,marker=mk,ms=3,label=lbl)
    aq.plot(t_steps,Q_sim[br,:],color=bcm[k],lw=lw,alpha=al,marker=mk,ms=3,label=lbl)
    if br in adj_br:
        rA = BRANCH_DATA[br,4]*SBASE if BRANCH_DATA[br,4]>0 else 600  # rough thermal limit
        # use actual branch rating from separate table
for ax in [ap,aq]:
    ax.axhline(0,color='k',lw=0.5,alpha=0.3)
    shade(ax)
    ax.set_xlim(0.5,N_STEPS+0.5); ax.set_xticks(t_steps)
    ax.set_xticklabels(t_lbl,fontsize=8)
    ax.legend(ncol=4,fontsize=6,loc='upper right')

for step,(br_idx,_) in RW.items():
    ap.annotate(f'Br{br_idx}\ntripped→0',xy=(step,0),xytext=(step+0.4,-60),
               fontsize=7,color='#c0392b',
               arrowprops=dict(arrowstyle='->',color='#c0392b',lw=1))

ap.set_ylabel('Active Power (MW)')
ap.set_title('Branch Active Power Flow — IEEE 39 Scenario 8\n'
            'Bold markers = branches adjacent to attacked buses',pad=8)
aq.set_ylabel('Reactive Power (MVAr)'); aq.set_xlabel('Timestep')
aq.set_title('Reactive Power Redistribution — generator AVR response visible at worm events',pad=8)

plt.savefig('graphs/graph2_branch_power.png',
           dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  Saved.")


# ── GRAPH 3: Nodal Indices Dashboard ────────────────────────────────────
print("Plotting Graph 3: Nodal Indices Dashboard...")
fig3 = plt.figure(figsize=(15,10))
gs3  = GridSpec(2,2,figure=fig3,hspace=0.52,wspace=0.35)

ax_D = fig3.add_subplot(gs3[0,0])
bar_index(ax_D, D_idx, 'D — Frequency Response Index\n(higher = weaker inertial support)',
         'D value', '#2980b9')
ax_D.text(0.97,0.97,
         f'H=0: buses 30, 39 (wind farms)\nD=0 → zero inertia contribution\n'
         f'Buses 14–21 most exposed',
         transform=ax_D.transAxes, fontsize=7, ha='right', va='top',
         bbox=dict(boxstyle='round',facecolor='#eaf4fb',alpha=0.85))

ax_L = fig3.add_subplot(gs3[0,1])
bar_index(ax_L, L_idx, 'L — Power Flow Coupling Index\n(higher = fault spreads further)',
         'L value', '#e67e22')
ax_L.text(0.97,0.97,
         f'Topological hub buses dominate\nFault here → wide redistribution',
         transform=ax_L.transAxes, fontsize=7, ha='right', va='top',
         bbox=dict(boxstyle='round',facecolor='#fef9ef',alpha=0.85))

ax_I = fig3.add_subplot(gs3[1,0])
I_vuln = np.clip(1-I_idx, 0, 1)
bar_index(ax_I, I_vuln, f'Voltage Vulnerability (1−I) at Step 3\n(higher = closer to collapse)',
         '1−I', '#27ae60')
thresh80 = np.percentile(I_vuln,80)
ax_I.axhline(thresh80,color='#c0392b',lw=1.5,ls='--',alpha=0.8,
            label=f'80th pct = {thresh80:.3f}')
ax_I.legend(fontsize=7.5)
ax_I.text(0.97,0.97,
         f'V_critical = 0.80 p.u.\nV_min(step3) = {V_sim[:,2].min():.4f} p.u.\n'
         f'@ bus {V_sim[:,2].argmin()+1}',
         transform=ax_I.transAxes, fontsize=7, ha='right', va='top',
         bbox=dict(boxstyle='round',facecolor='#eafaf1',alpha=0.85))

ax_R = fig3.add_subplot(gs3[1,1])
bar_index(ax_R, R_idx, 'R — Fault Severity Index (composite)\n(higher = highest attack priority)',
         'R value', '#8e44ad')
ax_R.text(0.5,-0.20,
         f'Game-AHP weights:  ω_D = {W_star[0]:.4f}   '
         f'ω_L = {W_star[1]:.4f}   ω_I = {W_star[2]:.4f}',
         transform=ax_R.transAxes, fontsize=8, ha='center', color='#444',
         bbox=dict(boxstyle='round',facecolor='#fdfefe',alpha=0.88))
ax_R.text(0.97,0.97,
         f'Highest R bus: {R_idx.argmax()+1}\n(R = {R_idx.max():.4f})\n'
         f'R = ω_D·D + ω_L·L + ω_I·(1−I)',
         transform=ax_R.transAxes, fontsize=7, ha='right', va='top',
         bbox=dict(boxstyle='round',facecolor='#f5eef8',alpha=0.85))

p1 = mpatches.Patch(facecolor='#27ae60',alpha=0.5,label='Generator buses (30–39)')
p2 = mpatches.Patch(facecolor='#c0392b',alpha=0.8,label='Top-5 most vulnerable')
fig3.legend(handles=[p1,p2],loc='lower center',ncol=2,fontsize=9,
           bbox_to_anchor=(0.5,0.01),framealpha=0.9)
fig3.suptitle(
    'Nodal Vulnerability Indices at Step 3 (First Ransomware Attack)\n'
    'IEEE 39-Bus | Scenario 8 | Newton-Raphson AC Power Flow',
    fontsize=12, fontweight='bold', y=1.01)
plt.savefig('graphs/graph3_nodal_indices.png',
           dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  Saved.")


# ── GRAPH 4: SIR Propagation ─────────────────────────────────────────────
print("Plotting Graph 4: SIR Propagation...")
fig4,(as_,ah) = plt.subplots(1,2,figsize=(15,6),
                              gridspec_kw={'width_ratios':[2,1],'wspace':0.28})

t_fine  = np.linspace(1,N_STEPS,500)
sol_fine= odeint(sir_ode,[s0,i0,r0_v],t_fine,args=(lam,mu))

as_.plot(t_steps,i_dens,'r-o',lw=2.5,ms=7,zorder=5,label='Actual i(t)  — infected density')
as_.plot(t_fine,sol_fine[:,1],'b--',lw=2.0,alpha=0.85,
        label=f'SIR ODE  λ={lam:.3f}  μ={mu:.3f}')
as_.plot(t_steps,s_dens,'g-^',lw=1.5,ms=5,alpha=0.7,label='S — Susceptible')
as_.plot(t_steps,r_dens,'k-v',lw=1.5,ms=5,alpha=0.7,label='R — Recovered')
as_.plot(t_fine,sol_fine[:,0],'g--',lw=0.9,alpha=0.35)
as_.plot(t_fine,sol_fine[:,2],'k--',lw=0.9,alpha=0.35)

midp_act_t = t_steps[i_dens.argmax()]
midp_pred_t = t_fine[sol_fine[:,1].argmax()]
as_.plot(midp_act_t, MIDP_actual,'r*',ms=18,zorder=6,
        label=f'Actual MIDP = {MIDP_actual:.4f}')
as_.plot(midp_pred_t,MIDP_pred,  'b*',ms=16,zorder=6,
        label=f'Predicted MIDP = {MIDP_pred:.4f}')
as_.annotate(f'Error = {err_pct:.1f}%',
            xy=((midp_act_t+midp_pred_t)/2,(MIDP_actual+MIDP_pred)/2),
            fontsize=8.5,color='#555',ha='center',
            bbox=dict(boxstyle='round',facecolor='#fffff0',alpha=0.88))

shade(as_, yref=1.04)
as_.axhline(0.2,color='grey',lw=1.2,ls=':',alpha=0.7,label='Alert threshold i=0.2')
as_.set_xlim(0.5,N_STEPS+0.5); as_.set_ylim(-0.02,1.12)
as_.set_xticks(t_steps); as_.set_xticklabels(t_lbl,fontsize=8)
as_.set_xlabel('Timestep'); as_.set_ylabel('Density (fraction of 39 buses)')
as_.set_title('SIR Cascading Failure — IEEE 39 Scenario 8\n'
             'Actual NR-simulated i(t) vs SIR ODE model',pad=8)
as_.legend(loc='upper right',fontsize=7.5)
as_.text(0.03,0.97,
        f'R₀ = λ/μ = {R0_val:.2f}\n'
        f'{"Epidemic cascade  (R₀ > 1)" if R0_val>1 else "Contained  (R₀ < 1)"}',
        transform=as_.transAxes,fontsize=8.5,va='top',
        bbox=dict(boxstyle='round',facecolor='#fef9ef',alpha=0.9))

sm = np.where(node_states==1,0,np.where(node_states==2,0.5,1.0))
im = ah.imshow(sm,aspect='auto',cmap='RdYlGn',vmin=0,vmax=1,origin='upper',
              extent=[0.5,N_STEPS+0.5,N_BUS+0.5,0.5])
ah.axhline(29.5,color='white',lw=1.8,ls='--',alpha=0.8)
ah.text(0.6,29.1,'Generators ↑',color='white',fontsize=7)
for i,atk in enumerate(SEQUENCE):
    if atk!='normal':
        ah.axvline(i+1,color='#ff4444',lw=1.5,ls='--',alpha=0.65)
ah.set_xlabel('Timestep',fontsize=9); ah.set_ylabel('Bus Number',fontsize=9)
ah.set_title('Per-Bus State Evolution\n(Green=S, Yellow=I, Red=R)',pad=8)
ah.set_xticks(t_steps[::2]); ah.set_xticklabels(t_steps[::2].astype(int),fontsize=8)
ah.set_yticks(np.arange(1,N_BUS+1,4)); ah.set_yticklabels(np.arange(1,N_BUS+1,4),fontsize=7.5)
cb = plt.colorbar(im,ax=ah,fraction=0.04,pad=0.04)
cb.set_ticks([0,0.5,1.0]); cb.set_ticklabels(['S','I','R'],fontsize=9)

fig4.suptitle(
    f'SIR Epidemic Model — Cascading Failure  |  IEEE 39  Scenario 8\n'
    f'λ={lam:.4f}   μ={mu:.4f}   R₀={R0_val:.2f}   '
    f'MIDP_actual={MIDP_actual:.4f}   MIDP_predicted={MIDP_pred:.4f}   '
    f'Error={err_pct:.1f}%',
    fontsize=10,fontweight='bold',y=1.02)
plt.savefig('graphs/graph4_sir_propagation.png',
           dpi=150,bbox_inches='tight',facecolor='white')
plt.close(); print("  Saved.")

# ─── Final summary ────────────────────────────────────────────────────────
print()
print("═"*60)
print("FINAL RESULTS — Newton-Raphson AC Power Flow")
print("═"*60)
print(f"  Base-case convergence : {conv0} in {itr0} iterations")
print(f"  Scenario convergence  : {CONV.sum()}/{N_STEPS} steps converged")
print(f"  V range (scenario)    : [{V_sim.min():.4f}, {V_sim.max():.4f}] p.u.")
print(f"  Total V<0.95 events   : {(V_sim<0.95).sum()} (bus×step)")
print(f"  Most affected bus     : {V_sim.min(axis=1).argmin()+1} "
      f"(min V = {V_sim.min(axis=1).min():.4f} p.u.)")
print(f"  MIDP actual           : {MIDP_actual:.4f}")
print(f"  MIDP predicted (SIR)  : {MIDP_pred:.4f}")
print(f"  SIR fit error         : {err_pct:.2f}%  (target <5%)")
print(f"  R₀                    : {R0_val:.4f}")
print(f"  Game-AHP weights      : D={W_star[0]:.4f}, L={W_star[1]:.4f}, I={W_star[2]:.4f}")
print(f"  Highest-R bus         : {R_idx.argmax()+1}")
