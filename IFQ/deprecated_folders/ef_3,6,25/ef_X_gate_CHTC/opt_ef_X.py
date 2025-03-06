import numpy as np
import qutip
import nevergrad as ng
from CoupledQuantumSystems.IFQ import gfIFQ
from CoupledQuantumSystems.evo import ODEsolve_and_post_process
from CoupledQuantumSystems.optimize import run_optimization_with_progress
from CoupledQuantumSystems.drive import DriveTerm, square_pulse_with_rise_fall
import time
import json
import multiprocessing
import argparse

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

initial_states = [
    qutip.basis(qbt.truncated_dim, 1),
    qutip.basis(qbt.truncated_dim, 2),
]

def objective(t_tot, amp, w_d, ramp):
    tlist = np.linspace(0, t_tot, 100)

    results = []
    for init_state in initial_states:
        res = ODEsolve_and_post_process(
            y0=init_state,
            tlist=tlist,
            static_hamiltonian=qbt.diag_hamiltonian,
            drive_terms=[
                    DriveTerm(
                        driven_op=qutip.Qobj(qbt.fluxonium.n_operator(energy_esys=True)),
                        pulse_shape_func=square_pulse_with_rise_fall,
                        pulse_id="pi",
                        pulse_shape_args={
                            "w_d": w_d,    # No extra 2pi factor
                            "amp": amp,    # No extra 2pi factor
                            "t_square": t_tot * (1-ramp*2),
                            "t_rise": t_tot * ramp,
                        },
                    )
                ],
            e_ops=e_ops,
            print_progress=False,
        )
        results.append(res)

    one_minus_pop2 = abs(1 - results[0].expect[2][-1]+ 0.99* results[0].expect[0][-1])
    one_minus_pop1 = abs(1 - results[1].expect[1][-1]+ 0.99* results[1].expect[0][-1])
    return one_minus_pop2 + one_minus_pop1

def main(t_tot_idx):
    t_tot_list = np.linspace(50, 120, 29)
    t_tot = t_tot_list[t_tot_idx]

    old_t_tot_arr = np.array([50.0,60.0,70.0,80.0,90.0,100.0,110.0,120.0,130.0,140.0,150.0,160.0,170.0,180.0,190.0,200.0,210.0,220.0,230.0,240.0,250.0])
    closest_old_t_tot_idx = np.argmin(np.abs(old_t_tot_arr - t_tot))
    old_t_tot = old_t_tot_arr[closest_old_t_tot_idx]
    with open('consolidated_optimization_results.json', 'r') as f:
        consolidated_results = json.load(f)
    result = consolidated_results[str(old_t_tot)]
    amp_guess = result['best_amp']
    w_d_guess = result['best_w_d']
    ramp_guess = result['best_ramp']

    parametrization = ng.p.Instrumentation(
        amp=ng.p.Log(init=amp_guess, lower=2, upper=20),
        w_d=ng.p.Log(init=w_d_guess, lower=0.002, upper=0.0045),
        ramp=ng.p.Scalar(init=ramp_guess, lower=1e-10, upper=0.5),
    )

    budget = 6000
    num_workers = 8
    optimizer = ng.optimizers.CMA(parametrization=parametrization,
                            budget=budget,
                            num_workers=num_workers)
    
    recommendation, progress = run_optimization_with_progress(
        optimizer=optimizer,
        objective_fn=objective,
        param_names=['amp', 'w_d','ramp'],
        title="ef X gate optimization",
        budget=budget,
        num_workers=num_workers,
        show_live=False,
        t_tot=t_tot
    )

    # Extract and save results
    best_amp = recommendation.kwargs["amp"]
    best_w_d = recommendation.kwargs["w_d"]
    best_ramp = recommendation.kwargs["ramp"]


    results_dict = {
        "t_tot": float(t_tot),  # Ensure t_tot is JSON serializable
        "best_amp": float(best_amp),  # Convert numpy types to Python float
        "best_w_d": float(best_w_d),
        "best_ramp": float(best_ramp),
        "best_value": float(progress.best_value),  # Also save the best value achieved
        "optimization_time": time.time() - progress.start_time,  # Save total optimization time
    }

    # Save as JSON with nice formatting
    with open(f"nevergrad_optimized_ef_X_{t_tot}.json", "w") as f:
        json.dump(results_dict, f, indent=4)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Raman Drive Optimization')
    parser.add_argument('t_tot_idx', type=int, help='Index for t_tot array')
    return parser.parse_args()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    args = parse_arguments()
    main(args.t_tot_idx)