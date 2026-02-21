import numpy as np
import matplotlib.pyplot as plt

# 1. Setup high-resolution time array for sharp corners (prevents early interpolation)
t = np.linspace(0, 19, 1000)
n_buses = 39
voltages = np.zeros((n_buses, len(t)))

# 2. Initial steady state (tight band between 0.99 and 1.06)
np.random.seed(42) 
initial_v = np.random.uniform(0.99, 1.06, n_buses)
for i in range(n_buses):
    voltages[i, :] = initial_v[i]

# 3. Define the cascading event timeline
t_start = 5.0
t_trip = 7.0       # Line trips
t_derate = 10.0    # Gen derating
t_escalate = 12.0  # Load escalation
t_end = 15.0       # Attack isolated/recovery

# 4. Group buses by electrical distance / vulnerability
gens = np.arange(0, 10)       # Regulated generator buses
light = np.arange(10, 20)     # Lightly affected (distant)
mod = np.arange(20, 30)       # Moderately stressed
severe = np.arange(30, 39)    # Collapsed (near fault/overload)

# 5. Apply multi-stage degradation
for i in range(n_buses):
    # Define base degradation rates for each phase [phase1, phase2, phase3, phase4]
    if i in gens:
        rates = [0.001, 0.002, 0.001, 0.003] # Barely moves
    elif i in light:
        rates = [0.008, 0.012, 0.018, 0.025] 
    elif i in mod:
        rates = [0.015, 0.025, 0.035, 0.050]
    else: 
        rates = [0.030, 0.050, 0.070, 0.090] # Steep drops

    # Inject natural variance so lines fan out cleanly instead of clumping
    rates = [r * np.random.uniform(0.7, 1.3) for r in rates]

    for j, time in enumerate(t):
        if t_start <= time < t_trip:
            voltages[i, j] = voltages[i, j-1] - rates[0] * (t[j] - t[j-1])
        elif t_trip <= time < t_derate:
            voltages[i, j] = voltages[i, j-1] - rates[1] * (t[j] - t[j-1])
        elif t_derate <= time < t_escalate:
            voltages[i, j] = voltages[i, j-1] - rates[2] * (t[j] - t[j-1])
        elif t_escalate <= time < t_end:
            voltages[i, j] = voltages[i, j-1] - rates[3] * (t[j] - t[j-1])
        elif time >= t_end:
            # Immediate recovery sequence
            voltages[i, j] = initial_v[i] - 0.005 * np.random.rand()

# 6. Plotting and Styling
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 6))

for i in range(n_buses):
    if i == 34:  # Example of a severe bus
        ax.plot(t, voltages[i], color='#d62728', linewidth=2, label='Bus 35 (Collapsed)')
    elif i == 24: # Example of a moderate bus
        ax.plot(t, voltages[i], color='#ff7f0e', linewidth=2, label='Bus 25 (Moderate)')
    elif i == 1:  # Example of a generator bus
        ax.plot(t, voltages[i], color='#1f77b4', linewidth=2, label='Bus 2 (Regulated)')
    else:
        ax.plot(t, voltages[i], alpha=0.35, linewidth=1.2)

# Event Markers
ax.axvline(x=t_start, color='red', linestyle='--', linewidth=1.5, label='Attack Start')
ax.axvline(x=t_trip, color='gray', linestyle=':', alpha=0.6)
ax.axvline(x=t_derate, color='gray', linestyle=':', alpha=0.6)
ax.axvline(x=t_escalate, color='gray', linestyle=':', alpha=0.6)
ax.axvline(x=t_end, color='green', linestyle='--', linewidth=1.5, label='Attack End / Recovery')

# Annotations for the cascading events
ax.text(t_trip + 0.1, 0.95, 'Line Trips', rotation=90, fontsize=9, alpha=0.7)
ax.text(t_derate + 0.1, 0.95, 'Gen Derating', rotation=90, fontsize=9, alpha=0.7)
ax.text(t_escalate + 0.1, 0.95, 'Load Escalates', rotation=90, fontsize=9, alpha=0.7)

ax.set_title('Bus Voltage Profiles During Multi-Stage Attack Event', fontsize=16, pad=15)
ax.set_xlabel('Timestep (s)', fontsize=12)
ax.set_ylabel('Voltage Magnitude (p.u.)', fontsize=12)
ax.set_xlim(-0.5, 19.5)
ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.show()