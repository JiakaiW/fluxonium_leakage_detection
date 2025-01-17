import numpy as np
import qutip
import pickle
import os
import nevergrad as ng
from CoupledQuantumSystems.drive import DriveTerm, square_pulse_with_rise_fall
from CoupledQuantumSystems.IFQ import gfIFQ
from CoupledQuantumSystems.evo import ODEsolve_and_post_process
from concurrent.futures import ProcessPoolExecutor
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich import box
import time
import json
import multiprocessing

# -------------------------------
# Define your parameters and objects
# -------------------------------
EJ = 4
EC = EJ/2
EL = EJ/30

qbt = gfIFQ(EJ=EJ, EC=EC, EL=EL, flux=0, truncated_dim=13)
e_ops = [qutip.ket2dm(qutip.basis(qbt.truncated_dim, i)) for i in range(4)]

with open('results_backup_four_level.pkl', 'rb') as f:
    results_dict = pickle.load(f)

initial_states = [qutip.basis(qbt.truncated_dim, 0), qutip.basis(qbt.truncated_dim, 2)]

def objective(detuning, t_duration, amp1_scaling_factor, amp2_scaling_factor):
    detuning1 = detuning
    detuning2 = detuning
    tlist = np.linspace(0,t_duration,t_duration*2)

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

def main():
    # ---------------------------------------------------------------------
    # SET UP NEVERGRAD INSTRUMENTATION
    # ---------------------------------------------------------------------
    detuning_arr = np.array([0.1,0.2,0.3,0.4,0.5])
    t_duration_arr = np.array([50,100,150,200])
    
    # Use available CPU cores for parallel processing
    num_workers = multiprocessing.cpu_count()
    show_live = False  # Set this to False to disable live table
    
    for i,detuning in enumerate(detuning_arr):
        for j, t_duration in enumerate(t_duration_arr):
            amp1_scaling_factor, amp2_scaling_factor = results_dict[(detuning, t_duration)]
            parametrization = ng.p.Instrumentation(
                amp1_scaling_factor=ng.p.Log(init=amp1_scaling_factor, lower=amp1_scaling_factor/4, upper=amp1_scaling_factor*4),
                amp2_scaling_factor=ng.p.Log(init=amp2_scaling_factor, lower=amp2_scaling_factor/4, upper=amp2_scaling_factor*4),
            )
            
            budget = 40
            optimizer = ng.optimizers.CMA(parametrization=parametrization,
                                    budget=budget,
                                    num_workers=num_workers)
            
            # Set up logging
            log_file = f"nevergrad_optimizer_log_{detuning}_{t_duration}.pkl"
            logger = ng.callbacks.ParametersLogger(log_file)
            optimizer.register_callback("tell", logger)
            
            # Run optimization with progress tracking
            recommendation, progress = run_optimization_with_progress(
                optimizer=optimizer,
                objective_fn=objective,
                param_names=['amp1_scaling_factor', 'amp2_scaling_factor'],
                title="Raman Drive Optimization",
                budget=budget,
                num_workers=num_workers,
                show_live=show_live,  # Pass the show_live parameter
                detuning=detuning,
                t_duration=t_duration
            )
            
            # Extract and save results
            best_amp1 = recommendation.kwargs["amp1_scaling_factor"]
            best_amp2 = recommendation.kwargs["amp2_scaling_factor"]
            
            print(f"\nOptimization complete for detuning={detuning}, t_duration={t_duration}")
            print(f"Best amp1: {best_amp1}, best amp2: {best_amp2}")
            
            optimization_results = {
                "detuning": float(detuning),
                "t_duration": float(t_duration),
                "best_amp1": float(best_amp1),
                "best_amp2": float(best_amp2),
                "best_value": float(progress.best_value),
                "optimization_time": time.time() - progress.start_time,
            }
            
            # Save as JSON with nice formatting
            with open(f"nevergrad_optimized_results_{detuning}_{t_duration}.json", "w") as f:
                json.dump(optimization_results, f, indent=4)

if __name__ == '__main__':
    # Set start method to 'spawn' for cross-platform compatibility
    multiprocessing.set_start_method('spawn')
    main()
