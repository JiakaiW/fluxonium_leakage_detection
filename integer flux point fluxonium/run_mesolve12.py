import sys
# original_sys_path = sys.path.copy()
# sys.path.append('../')
from utils import *

tot_time =474
tlist = np.linspace(0, tot_time, tot_time)


qubit_level = 25
osc_level = 50

list_of_systems = []
list_of_kwargs = []

for kappa in [1e-3,1e-2]:
    for i, levels in zip([0,1,2],[50,30,30]):
        system = fluxonium_oscillator_system(
            EJ = 3,
            EC = 0.6,
            EL = 0.13,
            Er = 7.2622522,
            g_strength = 0.3,
            qubit_level = qubit_level,
            osc_level = osc_level,
            kappa = kappa,
            products_to_keep=[[ql, ol] for ql in [i] for ol in range(levels) ],
            computaional_states = '1,2',
            )

        kwargs = {'intial_state': system.truncate_function(qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(i, 0)])),
                    'tlist': tlist,
                    'osc_decay' : True,
                    'e_ops' : [system.a_trunc,system.a_trunc.dag()*system.a_trunc],
                    'amp' : 0.004,
                    't_stop':None,}
        
        list_of_systems.append(system)
        list_of_kwargs.append(kwargs)

        print("added one system to jobs list")

results = run_fluxonium_osc_system_mesolve_jobs(list_of_systems,list_of_kwargs)

import pickle
with open('../pickles/mesolve_12.pkl', 'wb') as file:
    pickle.dump(results, file)
