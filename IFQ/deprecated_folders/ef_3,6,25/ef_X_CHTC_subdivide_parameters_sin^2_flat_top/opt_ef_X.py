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
from itertools import product

# -------------------------------
# Define your parameters and objects
# -------------------------------
EJ = 3
EJoverEC = 6
EJoverEL = 25
EC = EJ / EJoverEC
EL = EJ / EJoverEL

# Define parameter sub-regimes
def create_regimes(lower, upper, num_divisions, name):
    """Create evenly spaced regimes between lower and upper bounds"""
    edges = np.linspace(lower, upper, num_divisions + 1)
    return [
        {
            "name": f"{name}_{i}",
            "lower": edges[i],
            "upper": edges[i + 1]
        }
        for i in range(num_divisions)
    ]

# Define number of divisions for each parameter
N_AMP_DIVISIONS = 6
N_WD_DIVISIONS = 5
N_RAMP_DIVISIONS = 1

# Create regimes with evenly spaced divisions
AMP_REGIMES = create_regimes(8, 24, N_AMP_DIVISIONS, "amp")
WD_REGIMES = create_regimes(0.0028, 0.0045, N_WD_DIVISIONS, "wd")
RAMP_REGIMES = create_regimes(1e-10, 0.5, N_RAMP_DIVISIONS, "ramp")

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

def objective(amp, w_d, ramp, t_tot):
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

def main(t_tot_idx, amp_regime_idx, wd_regime_idx):
    t_tot_list = np.linspace(50, 120, 29)
    t_tot_center = t_tot_list[t_tot_idx]
    t_tot_lower = t_tot_center - 1.25
    t_tot_upper = t_tot_center + 1.25

    # Load initial guesses from previous optimization
    old_t_tot_arr = np.array([50.0,60.0,70.0,80.0,90.0,100.0,110.0,120.0,130.0,140.0,150.0,160.0,170.0,180.0,190.0,200.0,210.0,220.0,230.0,240.0,250.0])
    closest_old_t_tot_idx = np.argmin(np.abs(old_t_tot_arr - t_tot_center))
    old_t_tot = old_t_tot_arr[closest_old_t_tot_idx]
    with open('consolidated_optimization_results.json', 'r') as f:
        consolidated_results = json.load(f)
    result = consolidated_results[str(old_t_tot)]
    amp_guess = result['best_amp']
    w_d_guess = result['best_w_d']
    ramp_guess = result['best_ramp']
    t_tot_guess = t_tot_center

    # Get the specific regimes for this run
    amp_regime = AMP_REGIMES[amp_regime_idx]
    wd_regime = WD_REGIMES[wd_regime_idx]
    ramp_regime = RAMP_REGIMES[0]
    
    regime_name = f"{amp_regime['name']}_{wd_regime['name']}_{ramp_regime['name']}_t{t_tot_center:.1f}"
    print(f"\nOptimizing regime: {regime_name}")

    # Create parametrization for this regime
    parametrization = ng.p.Instrumentation(
        amp=ng.p.Log(
            init=min(max(amp_guess, amp_regime['lower']), amp_regime['upper']),
            lower=amp_regime['lower'],
            upper=amp_regime['upper']
        ),
        w_d=ng.p.Log(
            init=min(max(w_d_guess, wd_regime['lower']), wd_regime['upper']),
            lower=wd_regime['lower'],
            upper=wd_regime['upper']
        ),
        ramp=ng.p.Scalar(
            init=min(max(ramp_guess, ramp_regime['lower']), ramp_regime['upper']),
            lower=ramp_regime['lower'],
            upper=ramp_regime['upper']
        ),
        t_tot=ng.p.Scalar(
            init=t_tot_guess,
            lower=t_tot_lower,
            upper=t_tot_upper
        )
    )

    budget = 16
    num_workers = 8
    optimizer = ng.optimizers.CMA(
        parametrization=parametrization,
        budget=budget,
        num_workers=num_workers
    )
    
    recommendation, progress = run_optimization_with_progress(
        optimizer=optimizer,
        objective_fn=objective,
        param_names=['amp', 'w_d', 'ramp', 't_tot'],
        title=f"ef X gate optimization - {regime_name}",
        budget=budget,
        num_workers=num_workers,
        show_live=False
    )

    # Extract results for this regime
    best_amp = recommendation.kwargs["amp"]
    best_w_d = recommendation.kwargs["w_d"]
    best_ramp = recommendation.kwargs["ramp"]
    best_t_tot = recommendation.kwargs["t_tot"]
    best_value = float(progress.best_value)

    results_dict = {
        "regime": regime_name,
        "t_tot_center": float(t_tot_center),
        "best_t_tot": float(best_t_tot),
        "best_amp": float(best_amp),
        "best_w_d": float(best_w_d),
        "best_ramp": float(best_ramp),
        "best_value": best_value,
        "optimization_time": time.time() - progress.start_time,
        "regime_bounds": {
            "amp": {"lower": amp_regime['lower'], "upper": amp_regime['upper']},
            "w_d": {"lower": wd_regime['lower'], "upper": wd_regime['upper']},
            "ramp": {"lower": ramp_regime['lower'], "upper": ramp_regime['upper']},
            "t_tot": {"lower": t_tot_lower, "upper": t_tot_upper}
        }
    }

    # Save results to JSON
    output_filename = f"nevergrad_optimized_ef_X_{t_tot_center:.1f}_{regime_name}.json"
    with open(output_filename, "w") as f:
        json.dump(results_dict, f, indent=4)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Raman Drive Optimization')
    parser.add_argument('t_tot_idx', type=int, help='Index for t_tot array')
    parser.add_argument('amp_regime_idx', type=int, help=f'Index for amplitude regime (0 to {N_AMP_DIVISIONS-1})')
    parser.add_argument('wd_regime_idx', type=int, help=f'Index for w_d regime (0 to {N_WD_DIVISIONS-1})')
    return parser.parse_args()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    args = parse_arguments()
    main(args.t_tot_idx, args.amp_regime_idx, args.wd_regime_idx)