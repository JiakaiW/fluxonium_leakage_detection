
import sys
original_sys_path = sys.path.copy()
sys.path.append('../')
from utils import *
max_ql = 25
max_ol = 50
system = fluxonium_oscillator_system(
    EJ = 2.65,
    EC = 0.6,
    EL = 0.13,
    Er = 7.17391479,
    g_strength = 0.13,
    qubit_level = max_ql,
    osc_level = max_ol,
    # kappa = 0.001,
    products_to_keep=[[ql, ol] for ql in [0,1,2,7] for ol in range(max_ol) ],
    computaional_states = '1,2',
    )

# t_stop = 400
tot_time =2000
tlist = np.linspace(0, tot_time, tot_time)

results = system.run_mesolve_on_driving_osc(
    initial_states = [system.truncate_function(qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(i, 0)])) for i in range(3)],
    tlist = tlist,
    osc_decay = False,
    amp = 0.003,
    # t_stop = t_stop
)


import pickle
with open('../pickles/sesolve_12_large_dim_2.65_g0.13_a0.003_go_back.pkl', 'wb') as file:
    pickle.dump(results, file)
