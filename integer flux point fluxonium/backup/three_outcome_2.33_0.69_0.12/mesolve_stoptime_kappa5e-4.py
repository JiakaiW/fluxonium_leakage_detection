
import sys
original_sys_path = sys.path.copy()
sys.path.append('../')
from utils import *

ql = 20
ol = 30
system = fluxonium_oscillator_system(
    EJ = 2.33,
    EC = 0.69,
    EL = 0.12,
    Er = 7.16518677,
    g_strength = 0.18,
    qubit_level = ql,
    osc_level = ol,
    kappa = 0.0005,
    products_to_keep=[[ql, ol] for ql in [1,2] for ol in range(25) ],
    computaional_states = '1,2',
    w_d = 7.16475
    )

tot_time =733 + 20
tlist = np.linspace(0, tot_time, tot_time)


state_0_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(1,0)])
state_1_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(2,0)])
state_plus_dressed = (state_0_dressed  +  state_1_dressed).unit()
state_minus_dressed = (state_0_dressed - state_1_dressed).unit()

results = system.run_mesolve_on_driving_osc(
    initial_states = [system.truncate_function(state) for state in [state_plus_dressed, state_minus_dressed]],
    tlist = tlist,
    osc_decay = True,
    e_ops=[
        system.a_trunc.dag()*system.a_trunc
    ],
    amp = 0.0015,
)


import pickle
with open('../pickles/mesolve_temp_kappa5em4.pkl', 'wb') as file:
    pickle.dump(results, file)
