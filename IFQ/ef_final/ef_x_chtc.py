import numpy as np
import qutip
from CoupledQuantumSystems.drive import *
from scipy.optimize import minimize
from CoupledQuantumSystems.IFQ import gfIFQ
from CoupledQuantumSystems.evo import ODEsolve_and_post_process
import argparse
import json

EJ = 3
EC = EJ/4
EL = EJ/20.5

qbt = gfIFQ(EJ = EJ,EC = EC, EL = EL, flux = 0,truncated_dim=30)
e_ops = [qutip.basis(qbt.truncated_dim, i)*qutip.basis(qbt.truncated_dim, i).dag() for i in range(10)]

element = np.abs(qbt.fluxonium.matrixelement_table('n_operator',evals_count=3)[1,2]) 
freq = (qbt.fluxonium.eigenvals()[2]-qbt.fluxonium.eigenvals()[1]) * 2 * np.pi

def get_loss(t_tot,amp,w_d,ramp,plot=False):
    tlist = np.linspace(0, t_tot, 100)
    initial_states = [qutip.basis(qbt.truncated_dim, 1),qutip.basis(qbt.truncated_dim, 2)]
    drive_terms = [
            DriveTerm(
                driven_op=qutip.Qobj(
                    qbt.fluxonium.n_operator(energy_esys=True)),
                pulse_shape_func=square_pulse_with_rise_fall,
                pulse_id='pi',
                pulse_shape_args={
                    'w_d': w_d,  # Without 2pi
                    'amp': amp,  # Without 2pi
                    't_square': t_tot*(1-2*ramp),
                    't_rise': t_tot*ramp
                },
            )
        ]
    results = [ODEsolve_and_post_process(
                y0=initial_states[i],
                tlist=tlist,
                static_hamiltonian=qbt.diag_hamiltonian,
                drive_terms=drive_terms,
                # c_ops=c_ops,
                print_progress = False,
                e_ops=e_ops,
                ) for i in range(len(initial_states))]
    one_minus_pop2 = np.abs( 1- (results[0].expect[2][-1]))
    one_minus_pop1 = np.abs(1- (results[1].expect[1][-1]))
    return one_minus_pop2 + one_minus_pop1



def parse_arguments():
    parser = argparse.ArgumentParser(description='Run X gate Optimization')
    parser.add_argument('t_tot_idx', type=int, help='Index for t_tot array')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments()
    t_tot_idx = args.t_tot_idx
    t_tot_array = np.linspace(30,200,35)
    t_tot = t_tot_array[t_tot_idx]
    amp = np.pi/element/t_tot/3.1415/2
    w_d = qbt.fluxonium.eigenvals()[2]-qbt.fluxonium.eigenvals()[1]
    def objective(x):
        amp = x[0]
        w_d = x[1]
        ramp = x[2]
        return get_loss(t_tot,amp,w_d,ramp,plot=False)

    initial_guess =[amp,w_d,0.1]
    result = minimize(objective, initial_guess, method='Nelder-Mead')
    
    optimization_results = {
        "t_tot": float(t_tot),
        "amp": float(result.x[0]),
        "w_d": float(result.x[1]),
        "ramp": float(result.x[2]),
        "best_value": float(result.fun),
    }
    
    # Save as JSON with nice formatting
    with open(f"result_t_tot_{t_tot_idx}.json", "w") as f:
        json.dump(optimization_results, f, indent=4)
