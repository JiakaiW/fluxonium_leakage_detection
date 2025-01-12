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

class OptimizationProgress:
    def __init__(self, t_tot, budget):
        self.t_tot = t_tot
        self.budget = budget
        self.best_value = float('inf')
        self.best_params = None
        self.current_budget = budget
        self.running_jobs = 0
        self.current_evaluations = []  # List of (params, value) tuples for current batch
        self.start_time = time.time()
        self.total_evaluations = 0  # Track total number of evaluations
    
    def create_table(self):
        # Create main table with all columns
        table = Table(title=f"Optimization Progress (t_tot={self.t_tot})", box=box.ROUNDED)
        
        # Status columns
        table.add_column("Time", justify="right", style="cyan", width=8)
        table.add_column("Budget", justify="right", style="cyan", width=8)
        table.add_column("Jobs", justify="right", style="cyan", width=6)
        table.add_column("s/iter", justify="right", style="cyan", width=8)  # New column
        table.add_column("Best", justify="right", style="green", width=10)
        
        # Parameter columns for best result
        table.add_column("Best amp", justify="right", style="yellow", width=10)
        table.add_column("Best w_d", justify="right", style="yellow", width=10)
        table.add_column("Best ramp", justify="right", style="yellow", width=10)
        
        # Add header row for current evaluations
        table.add_column("amp", justify="right", style="magenta", width=10)
        table.add_column("w_d", justify="right", style="magenta", width=10)
        table.add_column("ramp", justify="right", style="magenta", width=10)
        table.add_column("Cost", justify="right", style="red", width=10)
        
        # Add main status row
        elapsed = time.time() - self.start_time
        # Calculate seconds per iteration
        s_per_iter = "-"
        if self.total_evaluations > 0:
            s_per_iter = f"{elapsed/self.total_evaluations:.1f}"
        
        best_row = [
            f"{elapsed:.1f}s",
            str(self.current_budget),
            str(self.running_jobs),
            s_per_iter,  # Add seconds per iteration
            f"{self.best_value:.6f}",
        ]
        
        if self.best_params:
            best_row.extend([
                f"{self.best_params['amp']:.6f}",
                f"{self.best_params['w_d']:.6f}",
                f"{self.best_params['ramp']:.6f}",
            ])
        else:
            best_row.extend(["-", "-", "-"])
        
        # Add empty cells for current evaluation columns
        best_row.extend(["-", "-", "-", "-"])
        table.add_row(*best_row)
        
        # Add current evaluations
        if self.current_evaluations:
            # Sort evaluations by value (best first)
            sorted_evals = sorted(self.current_evaluations, key=lambda x: x[1])  # Show all evaluations
            
            for params, value in sorted_evals:
                # Add empty cells for status columns
                row = ["-", "-", "-", "-", "-"]  # Time, Budget, Jobs, s/iter, Best
                # Add empty cells for best parameters
                row.extend(["-", "-", "-"])  # Best amp, w_d, ramp
                # Add current evaluation
                row.extend([
                    f"{params['amp']:.4f}",
                    f"{params['w_d']:.4f}",
                    f"{params['ramp']:.4f}",
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
EJ = 3
EJoverEC = 6
EJoverEL = 25
EC = EJ / EJoverEC
EL = EJ / EJoverEL

qbt = gfIFQ(EJ=EJ, EC=EC, EL=EL, flux=0, truncated_dim=20)
e_ops = [
    qutip.basis(qbt.truncated_dim, i) * qutip.basis(qbt.truncated_dim, i).dag()
    for i in range(10)
]

element = np.abs(
    qbt.fluxonium.matrixelement_table("n_operator", evals_count=3)[1, 2]
)
freq = (qbt.fluxonium.eigenvals()[2] - qbt.fluxonium.eigenvals()[1]) * 2 * np.pi


def objective(t_tot, amp, w_d,ramp):
    tlist = np.linspace(0, t_tot * (1+ramp), 100)

    initial_states = [
        qutip.basis(qbt.truncated_dim, 1),
        qutip.basis(qbt.truncated_dim, 2),
    ]

    drive_terms = [
        DriveTerm(
            driven_op=qutip.Qobj(qbt.fluxonium.n_operator(energy_esys=True)),
            pulse_shape_func=square_pulse_with_rise_fall,
            pulse_id="pi",
            pulse_shape_args={
                "w_d": w_d,    # No extra 2pi factor
                "amp": amp,    # No extra 2pi factor
                "t_square": t_tot * (1-ramp),
                "t_rise": t_tot * ramp,
            },
        )
    ]

    # Solve the dynamics for each initial state
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

    # Calculate final populations
    # (We want to transfer |1> -> |2> and |2> -> |1>.)
    one_minus_pop2 = abs(1 - results[0].expect[2][-1])  # from state |1> to |2>
    one_minus_pop1 = abs(1 - results[1].expect[1][-1])  # from state |2> to |1>

    return one_minus_pop2 + one_minus_pop1

# ---------------------------------------------------------------------
# SET UP NEVERGRAD INSTRUMENTATION
# ---------------------------------------------------------------------
# We'll specify a search space for each of t_tot, amp, w_d.
# For example, let's guess t_tot is in [1, 300].
# We'll assume amp in [0, 0.1], w_d in [freq*0.9, freq*1.1] as an example.
# Adjust these bounds to your scenario.


t_tot_list = np.linspace(50, 250, 21)
for t_tot in t_tot_list:
    # init_amp = np.pi / element / t_tot / 3.1415 / 2
    init_wd = qbt.fluxonium.eigenvals()[2] - qbt.fluxonium.eigenvals()[1]
    amp_guess = 1.5 * 250 / t_tot
    parametrization = ng.p.Instrumentation(
        amp=ng.p.Log(init=amp_guess, lower=amp_guess/4, upper=amp_guess*4),
        w_d=ng.p.Log(init=init_wd, lower=0.003, upper=0.0045),
        ramp=ng.p.Scalar(init=0.15, lower=1e-10, upper=0.5),
    )
    
    budget = 4000
    num_workers = 20
    # optimizer_dump_file = f"nevergrad_optimizer_dump_{t_tot}.pkl"
    # if os.path.exists(optimizer_dump_file):
    #     optimizer = ng.optimizers.CMA.load(optimizer_dump_file)
    # else:
    optimizer = ng.optimizers.CMA(parametrization=parametrization,
                            budget=budget,
                            num_workers=num_workers)
    log_file = f"nevergrad_optimizer_log_{t_tot}.pkl"
    logger = ng.callbacks.ParametersLogger(log_file)
    optimizer.register_callback("tell",  logger)
    # Create progress tracker
    progress = OptimizationProgress(t_tot, budget)
    
    def evaluate_candidate(candidate):
        """Evaluate a single candidate"""
        try:
            value = objective(t_tot=t_tot, **candidate.kwargs)
            return candidate, value
        except Exception as e:
            print(f"Error evaluating candidate: {e}")
            return candidate, float('inf')
    
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
                futures = [executor.submit(evaluate_candidate, c) for c in candidates]
                
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
    best_amp = recommendation.kwargs["amp"]
    best_w_d = recommendation.kwargs["w_d"]
    best_ramp = recommendation.kwargs["ramp"]
    
    print(f"\nOptimization complete for t_tot={t_tot}")
    print(f"Best amp: {best_amp}, best w_d: {best_w_d}, best ramp: {best_ramp}")
    
    results_dict = {
        "t_tot": float(t_tot),  # Ensure t_tot is JSON serializable
        "best_amp": float(best_amp),  # Convert numpy types to Python float
        "best_w_d": float(best_w_d),
        "best_ramp": float(best_ramp),
        "best_value": float(progress.best_value),  # Also save the best value achieved
        "optimization_time": time.time() - progress.start_time,  # Save total optimization time
        "budget_used": budget - remaining_budget
    }
    
    # Save as JSON with nice formatting
    with open(f"nevergrad_optimized_results_{t_tot}.json", "w") as f:
        json.dump(results_dict, f, indent=4)
