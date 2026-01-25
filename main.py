import argparse
import sys
from phase2_sim import run_phase2_simulation
from phase3_defense import train_agent, run_comparative_simulation

def main():
    parser = argparse.ArgumentParser(description="EcoGuard: Cyber-Physical Grid Resiliency Framework")
    
    parser.add_argument("--phase", type=int, choices=[2, 3], required=True,
                        help="Simulation Phase: 2 (Attack Simulation) or 3 (Defense/RL)")
    
    parser.add_argument("--grid", type=str, default="ieee13", 
                        choices=["ieee13", "ieee33", "ieee123", "ieee118", "ieee300", "cigre_mv", "gbnetwork", "simbench"],
                        help="Grid topology model to use")
    
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of simulation steps (Phase 2)")
    
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of RL training episodes (Phase 3)")

    args = parser.parse_args()

    print(f"Starting Simulation with args: {args}")

    if args.phase == 2:
        run_phase2_simulation(grid_type=args.grid, steps=args.steps)
    elif args.phase == 3:
        train_agent(episodes=args.episodes, grid_type=args.grid)
        # Execute comparative analysis after training
        run_comparative_simulation(steps=15, grid_type=args.grid)
    else:
        print("Invalid phase selected.")

if __name__ == "__main__":
    main()
