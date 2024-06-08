import sys
original_sys_path = sys.path.copy()
sys.path.append('../')
from backup.utils import *


if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    
    t_stop = 400
    tot_time =450
    tlist = np.linspace(0, tot_time, tot_time)

    qubit_level = 25
    osc_level = 40

    kappa = 1e-3

    system = fluxonium_oscillator_system(
        EJ = 2.65,
        EC = 0.6,
        EL = 0.13,
        Er = 7.17391479,
        g_strength = 0.13,
        qubit_level = qubit_level,
        osc_level = osc_level,
        kappa =kappa,
        products_to_keep=[[ql, ol] for ql in [0,7] for ol in range(osc_level) ],
        computaional_states = '1,2',
        )

    state_leakage = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(0,0)])

    result = system.run_mcsolve_on_driving_osc(initial_state = system.truncate_function(state_leakage),
                                                tlist = tlist,
                                                osc_decay = True,
                                                e_ops = [system.a_trunc,
                                                        system.a_trunc.dag()*system.a_trunc],
                                                amp = 0.003,
                                                t_stop = t_stop
                                                )


    with open(f'../pickles/mcsolve_2.65_g0.13_a0.003_10level_qbt_state{0}.pkl', 'wb') as file:
        pickle.dump(result, file)
    

    # state_0_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(1,0)])
    # state_1_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(2,0)])
    # state_plus_dressed = (state_0_dressed  +  state_1_dressed).unit()
    # state_minus_dressed = (state_0_dressed - state_1_dressed).unit()
    # initial_states  = [state_leakage,
    #                    state_0_dressed,
    #                    state_1_dressed,
    #                    state_plus_dressed,
    #                    state_minus_dressed ]