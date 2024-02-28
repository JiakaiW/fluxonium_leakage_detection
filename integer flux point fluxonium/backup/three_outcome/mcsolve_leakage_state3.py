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
        products_to_keep=[[ql, ol] for ql in [3] for ol in range(45) ] + [[ql, ol] for ql in [12] for ol in range(30)],
        computaional_states = '1,2',
        drive_transition = ((3,0),(3,1))
        )

    state_3 = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(0,0)])

    result = system.run_mcsolve_on_driving_osc(initial_state = system.truncate_function(state_3),
                                                tlist = tlist,
                                                osc_decay = True,
                                                e_ops = [system.a_trunc,
                                                        system.a_trunc.dag()*system.a_trunc],
                                                amp = 0.003,
                                                t_stop = t_stop
                                                )


    with open(f'../pickles/mcsolve_three_outcome_state{1}.pkl', 'wb') as file:
        pickle.dump(result, file)
    