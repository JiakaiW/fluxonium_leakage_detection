import numpy as np
import qutip
import nevergrad as ng
from CoupledQuantumSystems.IFQ import gfIFQ
from CoupledQuantumSystems.evo import ODEsolve_and_post_process
from CoupledQuantumSystems.optimize import run_optimization_with_progress
import time
import json
import multiprocessing
import argparse

# -------------------------------
# Define your parameters and objects
# -------------------------------
EJ = 4
EC = EJ/2
EL = EJ/30

qbt = gfIFQ(EJ=EJ, EC=EC, EL=EL, flux=0, truncated_dim=13)
e_ops = [qutip.ket2dm(qutip.basis(qbt.truncated_dim, i)) for i in range(4)]

with open('consolidated_optimization_results.json', 'r') as f:
    consolidated_results = json.load(f)

initial_states = [qutip.basis(qbt.truncated_dim, 0), qutip.basis(qbt.truncated_dim, 2)]
def objective(detuning, t_duration, amp1_scaling_factor, amp2_scaling_factor):
    detuning1 = detuning
    detuning2 = detuning
    tlist = np.linspace(0,t_duration,100)

    drive_terms = qbt.get_Raman_DRAG_drive_terms(
            i = 0,
            j = 3,
            k = 2,
            detuning1=detuning1,
            detuning2 = detuning2,
            t_duration=t_duration,
            shape='sin^2',
            amp_scaling_factor = 1,
            amp1_scaling_factor = amp1_scaling_factor,
            amp2_scaling_factor = amp2_scaling_factor,
            amp1_correction_scaling_factor = 0,
            amp2_correction_scaling_factor = 0,
        )

    results = []
    for init_state in initial_states:
        res = ODEsolve_and_post_process(
            y0=init_state,
            tlist=tlist,
            static_hamiltonian=qbt.diag_hamiltonian,
            drive_terms=drive_terms,
            e_ops=e_ops,
            print_progress=False,
        )
        results.append(res)

    one_minus_pop2 = np.abs( 1- (results[0].expect[2][-1] + 0.99* results[0].expect[1][-1]))
    one_minus_pop0 = np.abs(1- (results[1].expect[0][-1] +  0.99* results[1].expect[1][-1]))
    return one_minus_pop2 + one_minus_pop0

def main(detuning_idx, t_duration_idx):
    # ---------------------------------------------------------------------
    # SET UP NEVERGRAD INSTRUMENTATION
    # ---------------------------------------------------------------------
    new_detuning_arr = np.linspace(0.1,0.4,61)
    new_t_duration_arr = np.linspace(50,125,11)
    new_detuning = new_detuning_arr[detuning_idx]
    new_t_duration = new_t_duration_arr[t_duration_idx]
    
    old_detuning_arr = np.array([0.1,0.2,0.3,0.4,0.5])
    old_t_duration_arr = np.array([50.0,100.0,150.0,200.0])
    
    # Use available CPU cores for parallel processing
    num_workers = 8
    
    closest_old_detuning_idx = np.argmin(np.abs(old_detuning_arr - new_detuning))
    closest_old_t_duration_idx = np.argmin(np.abs(old_t_duration_arr - new_t_duration))
    old_detuning = old_detuning_arr[closest_old_detuning_idx]
    old_t_duration = old_t_duration_arr[closest_old_t_duration_idx]
    result = consolidated_results[str(old_detuning)][str(int(old_t_duration))]

    amp1_scaling_factor = result['best_amp1']
    amp2_scaling_factor = result['best_amp2']
    parametrization = ng.p.Instrumentation(
        amp1_scaling_factor=ng.p.Log(init=amp1_scaling_factor, lower=amp1_scaling_factor/5, upper=amp1_scaling_factor*5),
        amp2_scaling_factor=ng.p.Log(init=amp2_scaling_factor, lower=amp2_scaling_factor/5, upper=amp2_scaling_factor*5),
    )
    
    budget = 4000
    optimizer = ng.optimizers.CMA(parametrization=parametrization,
                            budget=budget,
                            num_workers=num_workers)
    
    # Run optimization with progress tracking
    recommendation, progress = run_optimization_with_progress(
        optimizer=optimizer,
        objective_fn=objective,
        param_names=['amp1_scaling_factor', 'amp2_scaling_factor'],
        title="Raman Drive Optimization",
        budget=budget,
        num_workers=num_workers,
        show_live=False,
        detuning=new_detuning,
        t_duration=new_t_duration
    )
    
    # Extract and save results
    best_amp1 = recommendation.kwargs["amp1_scaling_factor"]
    best_amp2 = recommendation.kwargs["amp2_scaling_factor"]

    optimization_results = {
        "detuning": float(new_detuning),
        "t_duration": float(new_t_duration),
        "best_amp1": float(best_amp1),
        "best_amp2": float(best_amp2),
        "best_value": float(progress.best_value),
        "optimization_time": time.time() - progress.start_time,
    }
    
    # Save as JSON with nice formatting
    with open(f"nevergrad_optimized_gf_raman_{new_detuning}_{new_t_duration}.json", "w") as f:
        json.dump(optimization_results, f, indent=4)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Raman Drive Optimization')
    parser.add_argument('detuning_idx', type=int, help='Index for detuning array')
    parser.add_argument('t_duration_idx', type=int, help='Index for t_duration array')
    return parser.parse_args()

if __name__ == '__main__':
    # Set start method to 'spawn' for cross-platform compatibility
    multiprocessing.set_start_method('spawn')
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Call main with parsed arguments
    main(args.detuning_idx, args.t_duration_idx)
