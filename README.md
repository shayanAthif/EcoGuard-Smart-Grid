# EcoGuard: A Deep Reinforcement Learning Framework for Resilient and Sustainable Smart Grids

This project implements a **Cyber-Physical Digital Twin** for power grids to simulate and mitigate cyber-attacks using **Deep Reinforcement Learning (RL)**. It demonstrates how AI can autonomously stabilize grids against threats like **FDIA (False Data Injection)** and **Ransomware**, ensuring reliability for sustainable energy networks.


## System Architecture

The framework consists of four integrated phases creating a complete **Cyber-Physical Loop** with physics-level validation.

### 1. Cyber-Attack Detection (Phase 1)
- **Input**: Network traffic data (UNSW-NB15 Dataset).
- **Model**: A **Hybrid Graph Neural Network (GNN)** and Deep Learning classifier.
- **Function**: Detects malicious patterns (e.g., Fuzzers, DoS, Reconnaissance) from raw network logs.
- **Output**: The specific attack type identified, which triggers the corresponding physical scenario.

### 2. Cyber-Physical Simulation (Phase 2)
- **Engine**: `pandapower` (based on Newton-Raphson Power Flow).
- **Mapping**: Converts the detected cyber-threat into a **physical grid event**.
    - *Example*: A "Fuzzer" attack is mapped to random noise injection in load sensors.
    - *Example*: "DoS" is mapped to communication loss, preventing remote control of breakers.
- **Digital Twin**: Simulates the grid's electrical response (Voltage, Current, Frequency) to these disturbances.

### 3. Resilient Control (Phase 3)
- **Agent**: A **Deep Q-Network (DQN)** Reinforcement Learning agent.
- **Observation**: Monitors the "Digital Twin" state (Bus Voltages, Line Loading).
- **Action**: Autonomously executes grid maneuvers (Island Microgrid, Shed Load, Switch Capacitor) to restore stability.

### 4. Physics-Level Nodal Analysis (Phase 4)
- **Purpose**: Validates RL defense outcomes through rigorous **physics-based vulnerability assessment**.
- **SIR Epidemic Model**: Models cascading failures as an epidemic — buses transition between **Susceptible → Infected → Recovered** states based on voltage degradation rates.
- **Vulnerability Indices**: Computes three complementary per-bus indices:
    - **Voltage Offset Index (I_v)** — Proximity of each bus voltage to the critical threshold.
    - **Power Flow Coupling Index (L_v)** — Entropy-based measure of how power redistributes through a bus.
    - **Frequency Response Index (D_v)** — Exposure to frequency deviations based on generator inertia and electrical distance.
- **Fault Severity Index (R)**: A composite metric aggregating D, L, and I to rank the most vulnerable buses.
- **Outputs**: Graphs for voltage timeseries, branch power flow redistribution, nodal vulnerability dashboards, and SIR propagation curves.

![alt text](image.png)

## Experimental Results

The following table summarizes the performance of the RL Agent compared to a baseline (no defense) across 10 distinct scenarios and grid topologies.

**Key Metrics:**
*   **Avg_Volt**: Average bus voltage (Target: 1.0 p.u.).
*   **Violations**: Number of safety violations (< 0.9 p.u.).
*   **RMSE**: Root Mean Square Error from nominal voltage (Lower is better).
*   **Improvement**: Percentage reduction in voltage deviation.

### Research Metrics Table

| Scenario | Grid | Avg Volt (Base) | Avg Volt (RL) | Violations (Base) | Violations (RL) | RMSE (Base) | RMSE (RL) | Resilience Improv. (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | IEEE 13 | 0.886 | **0.952** | 3 | **2** | 0.281 | **0.111** | 60.48% |
| **2** | IEEE 39 | 0.901 | **0.969** | 2 | **0** | 0.261 | **0.032** | 87.85% |
| **3** | IEEE 300 | 0.844 | 0.844 | 3 | 3 | 0.450 | 0.450 | 0.00% |
| **4** | IEEE 118 | 0.985 | 0.985 | 0 | 0 | 0.015 | 0.015 | 0.00% |
| **5** | CIGRE MV | 0.857 | **0.987** | 3 | **0** | 0.366 | **0.019** | 94.84% |
| **6** | GB Network | 0.727 | **0.835** | 15 | 15 | 0.395 | **0.166** | 58.02% |
| **7** | IEEE 13 | 0.828 | **0.960** | 3 | **1** | 0.380 | **0.106** | 72.00% |
| **8** | IEEE 33 | 0.772 | **0.968** | 4 | **0** | 0.448 | **0.032** | 92.88% |
| **9** | IEEE 118 | 0.985 | 0.985 | 0 | 0 | 0.015 | 0.015 | 0.00% |
| **10** | CIGRE MV | 0.985 | **0.989** | 2 | **0** | 0.025 | **0.016** | 37.43% |

### Physics Analysis Outputs

The Phase 4 analysis generates four key visualizations:

| Graph | Description |
| :--- | :--- |
| **Voltage Timeseries** | Per-bus voltage magnitude over all simulation timesteps with attack window shading |
| **Branch Power Flow** | Active & reactive power redistribution on lines adjacent to attacked buses |
| **Nodal Vulnerability Dashboard** | 2×2 panel of per-bus I_v, L_v, D_v, and composite R indices |
| **SIR Propagation Curve** | Epidemic-model cascading failure spread with staircase multi-stage attack progression |


## Project Structure

```
EcoGuard-Smart-Grid/
│
├── phase1.ipynb                  # Phase 1 — Cyber-Attack Detection (GNN + DL classifier)
├── grid_models.py                # Grid topology generation (IEEE 13, 33, 118, 300, GB, CIGRE, SimBench)
├── attack_logic.py               # Critical asset targeting & physical attack logic (FDI, Ransomware)
├── phase2_sim.py                 # Phase 2 — Attack Simulation loop
├── phase3_defense.py             # Phase 3 — RL Defense training loop (DQN agent)
├── physics_analysis.py           # Phase 4 — Physics-level nodal analysis & SIR epidemic model
├── visualization.py              # Advanced plotting for cascading failures & grid restoration
├── main.py                       # Single entry point for running the project
├── run_custom_scenarios.py       # Batch runner for generating results across 10 scenarios
├── generate_research_metrics.py  # Quantitative metrics calculation for research paper
├── render_metrics_table.py       # Renders the metrics table as a PNG image
├── hello.py                      # Standalone voltage cascade visualization script
│
├── UNSW_NB15_training-set.csv    # Training dataset (UNSW-NB15)
├── UNSW_NB15_testing-set.csv     # Testing dataset (UNSW-NB15)
├── research_metrics.csv          # Generated research metrics
├── rl_policy.csv                 # Trained RL policy data
│
└── graphs/                       # Output visualizations
    ├── voltage_timeseries.png
    ├── branch_power.png
    ├── nodal_indices_post_attack.png
    ├── sir_propogation.png
    └── Scenario*_Phase*.png      # Per-scenario Phase 2 & Phase 3 graphs
```

## Key Features

- **Dynamic Targeting**: Works on ANY grid topology without hardcoded indices.
- **Closed-Loop Physics**: Actions (Load Shedding, Capacitor Switching) physically modify the grid model and trigger a fresh Power Flow solution.
- **Deterministic Simulation**: Random seeds are synchronized across Attack and Defense phases to ensure fair comparison.
- **Advanced Visualization**: Generates publication-ready time-series plots with shaded resilience regions.
- **SIR Epidemic Modeling**: Applies epidemiological models to power grid cascading failures — treating voltage collapse as a contagion spreading through electrically coupled buses.
- **Composite Vulnerability Indices**: Ranks bus criticality using physics-derived metrics (Voltage Offset, Power Flow Coupling, Frequency Response) fused into a single Fault Severity Index.

## Installation & Usage

1.  **Install Dependencies**:
    ```bash
    pip install pandapower pandas numpy matplotlib seaborn simbench scipy
    ```

2.  **Run Full Simulation** (Phases 2 + 3 + 4):
    To reproduce the results above, run the automated scenario manager:
    ```bash
    python run_custom_scenarios.py
    ```

    Or run individual phases:
    ```bash
    # Phase 2: Attack Simulation
    python main.py --phase 2 --grid ieee13 --steps 20

    # Phase 3: Defense Training
    python main.py --phase 3 --grid ieee13 --episodes 1000
    ```

3.  **Run Physics Analysis** (Phase 4 — standalone):
    ```bash
    python physics_analysis.py
    ```

4.  **Run Standalone Voltage Cascade Visualization**:
    ```bash
    python hello.py
    ```

## Technologies Used

| Category | Tools |
| :--- | :--- |
| **Power Systems** | pandapower, SimBench |
| **Machine Learning** | PyTorch, Scikit-learn, Deep Q-Network (DQN) |
| **Graph Neural Networks** | PyTorch Geometric |
| **Scientific Computing** | NumPy, SciPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Dataset** | UNSW-NB15 (Cyber-attack traffic) |

## License

This project is for academic and research purposes.
