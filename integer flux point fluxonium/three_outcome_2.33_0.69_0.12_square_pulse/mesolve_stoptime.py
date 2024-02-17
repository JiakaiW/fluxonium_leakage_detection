
import sys
original_sys_path = sys.path.copy()
sys.path.append('../')
from utils import *

ql = 20
ol = 10
system = fluxonium_oscillator_system(
    EJ = 2.33,
    EC = 0.69,
    EL = 0.12,
    Er = 7.16518677,
    g_strength = 0.18,
    qubit_level = ql,
    osc_level = ol,
    kappa = 0.0005,
    products_to_keep=[[ql, ol] for ql in [0,1,2,3] for ol in range(6) ],
    computaional_states = '1,2',
    w_d = 7.16475
    )

tot_time =800
tlist = np.linspace(0, tot_time, tot_time)

results = system.run_mesolve_on_driving_osc(
    initial_states = [system.truncate_function(qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(i, 0)])) for i in range(4)],
    tlist = tlist,
    osc_decay = True,
    e_ops=[
        system.a_trunc.dag()*system.a_trunc
    ],
    amp = 0.0015,
)


import pickle
with open('../pickles/mesolve_temp.pkl', 'wb') as file:
    pickle.dump(results, file)
