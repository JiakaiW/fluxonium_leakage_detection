import sys
original_sys_path = sys.path.copy()
sys.path.append('../')
from utils import *


###########################################################
# The work flow of computing pauli error model:
#   initialize density matrices |a><b|,
#   apply operator U0^{-1} (in our case U0 is identity so we don't do this step.)
#   evolve under noisy U
#   take inner product of the result with P|a><b|P   
#   the number we get is the "probability" of Pauli error P.
#
# In single qubit case, there's only four initial "states" |a><b|
#   to run with.
#
# This script: 
#   evolve those four |a><b| and store to pickle
#
###########################################################

if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    tot_time =463
    tlist = np.linspace(0, tot_time, tot_time)
    qubit_level = 25
    osc_level = 40
    list_of_systems = []
    list_of_kwargs = []

    for kappa in [1e-3,0]:
        system = fluxonium_oscillator_system(
            EJ = 2.65,
            EC = 0.6,
            EL = 0.13,
            Er = 6.81289062,
            g_strength = 0.23,
            qubit_level = qubit_level,
            osc_level = osc_level,
            kappa =kappa,
            products_to_keep=[[ql, ol] for ql in [0,1] for ol in range(25) ],
            computaional_states = '0,1',
            )

        state_0_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(0,0)])
        state_1_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(1,0)])

        initial_states  = [state_0_dressed * state_0_dressed.dag(),
                           state_0_dressed * state_1_dressed.dag(),
                           state_1_dressed * state_0_dressed.dag(),
                           state_1_dressed * state_1_dressed.dag()]

        for state in initial_states:
            kwargs = {'intial_state': system.truncate_function(state),
                        'tlist': tlist,
                        'osc_decay' : True if kappa != 0 else False,
                        'amp' : 0.004}
            list_of_systems.append(system)
            list_of_kwargs.append(kwargs)

    results = run_fluxonium_osc_system_mesolve_jobs(list_of_systems,
                                                    list_of_kwargs,
                                                    post_processing = ['pad_back','partial_trace_computational_states'],
                                                    max_workers=8)
        
    with open('../pickles/mesolve_01_pauli_dm.pkl', 'wb') as file:
        pickle.dump(results, file)
