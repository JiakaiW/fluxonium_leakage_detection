import sys
original_sys_path = sys.path.copy()
sys.path.append('../')
from backup.utils import *


if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    
    t_stop = 418
    tot_time =440
    tlist = np.linspace(0, tot_time, tot_time)

    qubit_level = 25
    osc_level = 50

    kappa = 1e-3

    system = fluxonium_oscillator_system(
        EJ = 2.5,
        EC = 0.65,
        EL = 0.12,
        Er = 7.05877808,
        g_strength = 0.23,
        qubit_level = qubit_level,
        osc_level = osc_level,
        kappa =kappa,
        products_to_keep=[[ql, ol] for ql in [1,2] for ol in range(20) ]+ \
                        [[ql, ol] for ql in [9,10] for ol in range(15) ] + \
                        [[ql, ol] for ql in [12] for ol in range(22) ],
        computaional_states = '1,2',
            drive_transition = ((3,0),(3,1))

        )

    # state_leakage = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(0,0)])
    state_0_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(1,0)])
    state_1_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(2,0)])
    state_plus_dressed = (state_0_dressed  +  state_1_dressed).unit()
    state_minus_dressed = (state_0_dressed - state_1_dressed).unit()
    initial_states  = [state_0_dressed,
                       state_1_dressed,
                       state_plus_dressed,
                       state_minus_dressed ]
    for i in [0,1,2,3]:
        result = system.run_mcsolve_on_driving_osc(initial_state = system.truncate_function(initial_states[i]),
                                                    tlist = tlist,
                                                    osc_decay = True,
                                                    e_ops = [system.a_trunc,
                                                            system.a_trunc.dag()*system.a_trunc],
                                                    amp = 0.003,
                                                    t_stop = t_stop
                                                    )


        with open(f'../pickles/mcsolve_three_outcome_state{i+2}.pkl', 'wb') as file:
            pickle.dump(result, file)
        

