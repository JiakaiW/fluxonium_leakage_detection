import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import qutip
from CoupledQuantumSystems.drive import *
from scipy.optimize import minimize
from CoupledQuantumSystems.IFQ import gfIFQ
from CoupledQuantumSystems.evo import ODEsolve_and_post_process

EJ = 3
EJoverEC = 6
EJoverEL = 25
EC = EJ / EJoverEC
EL = EJ / EJoverEL
n_lvls = 30
qbt = gfIFQ(EJ = EJ,EC = EC, EL = EL, flux = 0,truncated_dim=n_lvls)

state_0_dressed = qutip.basis(qbt.truncated_dim, 1)
state_1_dressed = qutip.basis(qbt.truncated_dim, 2)

t_tot_arr = [150]
w_d_arr = np.linspace(0.0055,0.0065,40)
amp_arr = np.linspace(3,7,40)

e_ops = [qutip.ket2dm(qutip.basis(qbt.truncated_dim, 0)),
         qutip.ket2dm(qutip.basis(qbt.truncated_dim, 1)),
         qutip.ket2dm(qutip.basis(qbt.truncated_dim, 2))]
n_op = qutip.Qobj(qbt.fluxonium.n_operator(energy_esys=True))

list_of_initial_states = [state_0_dressed,
                          state_1_dressed]

from tqdm import tqdm
import pickle

for t_tot in t_tot_arr:
    list_of_tlist = []
    list_of_drive_terms = []
    list_of_e_ops = []
    
    # Prepare parameters for this t_tot
    for w_d in w_d_arr:
        for amp in amp_arr:
            list_of_tlist.append(np.linspace(0, t_tot, 51))
            list_of_drive_terms.append([DriveTerm(driven_op=n_op,
                                        pulse_shape_func=sin_squared_pulse_with_modulation,
                                        pulse_id='pi',
                                        pulse_shape_args={"w_d": w_d,    
                                        "amp": amp,    
                                        "t_duration": t_tot,
                                        })])
            list_of_e_ops.append(e_ops)
    
    # Run parallel solver for this t_tot
    results = qbt.run_qutip_mesolve_parrallel(list_of_initial_states,
                            list_of_tlist,
                            list_of_drive_terms,
                            e_ops = list_of_e_ops,
                            c_ops = None,
                            show_multithread_progress=True,
                            show_each_thread_progress=False)
    ave_transfer_prob_list = []
    for results_of_the_same_evo in results:
        one_minus_pop2 = abs(1 - results_of_the_same_evo[0].expect[2][-1])
        one_minus_pop1 = abs(1 - results_of_the_same_evo[1].expect[1][-1])
        ave_transfer_prob_list.append((one_minus_pop2 + one_minus_pop1) / 2)
    # Save results for this t_tot
    filename = f'results_ttot_{t_tot}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump({
            't_tot': t_tot,
            'ave_transfer_prob_list': ave_transfer_prob_list,
            'w_d_arr': w_d_arr,
            'amp_arr': amp_arr
        }, f)