from src.analysis.experiments import run_single_experiment, run_paired_t_test
from src.config.loader import load_config

if __name__ == "__main__":
    
    
    config = load_config("config.yaml")
    experiment_mode = config.get("simulation", {}).get("experiment_mode", "single")
    
    if experiment_mode == "single":
        run_single_experiment()
    else:
        n_simulations = config.get("simulation", {}).get("n_simulations", 1000)
        run_paired_t_test(n_simulations)
       

