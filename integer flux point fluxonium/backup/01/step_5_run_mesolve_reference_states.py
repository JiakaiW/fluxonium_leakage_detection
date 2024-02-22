import sys
original_sys_path = sys.path.copy()
sys.path.append('../')
from utils import *


###########################################################
# This file create fluxonium_oscillator_system, truncate it 
#    down to computational subspace and run detection
#
# Because we need to partial trace out the qubit, we still 
#    pad it back to full dimension (using the functino mesolve_and_pad)
#
# We then trace out the qubit and truncate to two levels again,
#    store the states to the ODE result (using the function 
#    dressed_to_2_level_dm on every state of the ODE result)
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

    for kappa in [1e-3,1e-2]:
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
        state_plus_dressed = (state_0_dressed  +  state_1_dressed).unit()
        state_minus_dressed = (state_0_dressed - state_1_dressed).unit()
        initial_states  = [state_0_dressed,state_1_dressed,state_plus_dressed,state_minus_dressed ]

        for state in initial_states:
            kwargs = {'intial_state': system.truncate_function(state),
                        'tlist': tlist,
                        'osc_decay' : True,
                        'amp' : 0.004}
            list_of_systems.append(system)
            list_of_kwargs.append(kwargs)

    results = run_fluxonium_osc_system_mesolve_jobs(list_of_systems,
                                                    list_of_kwargs,
                                                    post_processing = ['pad_back','partial_trace_computational_states'],
                                                    max_workers=8)

        
    with open('../pickles/mesolve_01_reference_states_two_level.pkl', 'wb') as file:
        pickle.dump(results, file)
