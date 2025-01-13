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

class OptimizationProgress:
    def __init__(self, detuning, t_duration, budget):
        self.detuning = detuning
        self.t_duration = t_duration
        self.best_value = float('inf')
        self.best_params = None
        self.current_budget = budget
        self.running_jobs = 0
        self.current_evaluations = []  # List of (params, value) tuples for current batch
        self.start_time = time.time()
        self.total_evaluations = 0  # Track total number of evaluations
    
    def create_table(self):
        # Create main table with all columns
        table = Table(title=f"Optimization Progress (detuning={self.detuning}, t_duration={self.t_duration})", box=box.ROUNDED)
        
        # Status columns
        table.add_column("Time", justify="right", style="cyan", width=8)
        table.add_column("Budget", justify="right", style="cyan", width=8)
        table.add_column("Jobs", justify="right", style="cyan", width=6)
        table.add_column("s/iter", justify="right", style="cyan", width=8)  # New column
        table.add_column("Best", justify="right", style="green", width=10)
        
        # Parameter columns for best result
        table.add_column("Best amp1", justify="right", style="yellow", width=10)
        table.add_column("Best amp2", justify="right", style="yellow", width=10)
        
        # Add header row for current evaluations
        table.add_column("amp1", justify="right", style="magenta", width=10)
        table.add_column("amp2", justify="right", style="magenta", width=10)
        table.add_column("Cost", justify="right", style="red", width=10)

        # Add main status row
        elapsed = time.time() - self.start_time
        # Calculate seconds per iteration
        s_per_iter = "-"
        if self.total_evaluations > 0:
            s_per_iter = f"{elapsed/self.total_evaluations:.2f}"
        
        best_row = [
            f"{elapsed:.2f}s",
            str(self.current_budget),
            str(self.running_jobs),
            s_per_iter,  # Add seconds per iteration
            f"{self.best_value:.6f}",
        ]
        
        if self.best_params:
            best_row.extend([
                f"{self.best_params['amp1_scaling_factor']:.6f}",
                f"{self.best_params['amp2_scaling_factor']:.6f}",
            ])
        else:
            best_row.extend(["-", "-"])  # Only two dashes for two parameters
        
        # Add empty cells for current evaluation columns
        best_row.extend(["-", "-", "-"])  # Three dashes for amp1, amp2, Cost
        table.add_row(*best_row)
        
        # Add current evaluations
        if self.current_evaluations:
            # Sort evaluations by value (best first)
            sorted_evals = sorted(self.current_evaluations, key=lambda x: x[1])  # Show all evaluations
            
            for params, value in sorted_evals:
                # Add empty cells for status columns
                row = ["-", "-", "-", "-", "-"]  # Time, Budget, Jobs, s/iter, Best
                # Add empty cells for best parameters
                row.extend(["-", "-"])  # Only two dashes for Best amp1, amp2
                # Add current evaluation
                row.extend([
                    f"{params['amp1_scaling_factor']:.4f}",
                    f"{params['amp2_scaling_factor']:.4f}",
                    f"{value:.6f}"
                ])
                table.add_row(*row)
        
        return table
    
    def update(self, value, params=None, budget=None, running_jobs=None):
        if value is not None and value < self.best_value:
            self.best_value = value
            self.best_params = params
        if budget is not None:
            self.current_budget = budget
        if running_jobs is not None:
            self.running_jobs = running_jobs
        if params is not None and value is not None:
            self.current_evaluations.append((params, value))
            self.total_evaluations += 1  # Increment total evaluations counter
            # Keep only the most recent batch of evaluations
            if len(self.current_evaluations) > self.running_jobs:
                self.current_evaluations = self.current_evaluations[-self.running_jobs:]

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

def evaluate_candidate(candidate, detuning, t_duration):
    """Evaluate a single candidate"""
    try:
        value = objective(detuning=detuning, t_duration=t_duration, **candidate.kwargs)
        return candidate, value
    except Exception as e:
        print(f"Error evaluating candidate: {e}")
        return candidate, float('inf')

def main():
    # ---------------------------------------------------------------------
    # SET UP NEVERGRAD INSTRUMENTATION
    # ---------------------------------------------------------------------
    detuning_arr = np.array([0.1,0.2,0.3,0.4,0.5])
    t_duration_arr = np.array([50,100,150,200])
    
    for i,detuning in enumerate(detuning_arr):
        for j, t_duration in enumerate(t_duration_arr):
            amp1_scaling_factor, amp2_scaling_factor  = results_dict[(detuning, t_duration)]
            parametrization = ng.p.Instrumentation(
                amp1_scaling_factor=ng.p.Log(init=amp1_scaling_factor, lower=amp1_scaling_factor/4, upper=amp1_scaling_factor*4),
                amp2_scaling_factor=ng.p.Log(init=amp2_scaling_factor, lower=amp2_scaling_factor/4, upper=amp2_scaling_factor*4),
            )
            
            budget = 1000
            num_workers = 20
            optimizer = ng.optimizers.CMA(parametrization=parametrization,
                                    budget=budget,
                                    num_workers=num_workers)
            log_file = f"nevergrad_optimizer_log_{detuning}_{t_duration}.pkl"
            logger = ng.callbacks.ParametersLogger(log_file)
            optimizer.register_callback("tell",  logger)
            # Create progress tracker
            progress = OptimizationProgress(detuning, t_duration, budget)
            
            # Run optimization with live display
            with Live(progress.create_table(), refresh_per_second=2) as live:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    remaining_budget = budget
                    
                    while remaining_budget > 0:
                        # Ask for a batch of candidates
                        candidates = []
                        for _ in range(min(num_workers, remaining_budget)):
                            try:
                                candidate = optimizer.ask()
                                candidates.append(candidate)
                            except Exception as e:
                                print(f"Error asking for candidate: {e}")
                                continue
                        
                        if not candidates:
                            break
                        
                        # Update progress with batch size
                        progress.update(None, running_jobs=len(candidates), budget=remaining_budget)
                        live.update(progress.create_table())
                        
                        # Evaluate candidates in parallel
                        futures = [executor.submit(evaluate_candidate, c, detuning, t_duration) for c in candidates]
                        
                        # Process results as they complete
                        for future in futures:
                            try:
                                candidate, value = future.result()
                                optimizer.tell(candidate, value)
                                progress.update(value, candidate.kwargs)
                                remaining_budget -= 1
                                live.update(progress.create_table())
                            except Exception as e:
                                print(f"Error processing result: {e}")
                                continue
                        
                        # Update display
                        live.update(progress.create_table())
                        time.sleep(0.1)  # Small delay to prevent excessive updates
            
            # Get the recommendation
            recommendation = optimizer.recommend()
            
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
                "budget_used": budget - remaining_budget
            }
            
            # Save as JSON with nice formatting
            with open(f"nevergrad_optimized_results_{detuning}_{t_duration}.json", "w") as f:
                json.dump(optimization_results, f, indent=4)

if __name__ == '__main__':
    # Set start method to 'spawn' for cross-platform compatibility
    multiprocessing.set_start_method('spawn')
    main()
