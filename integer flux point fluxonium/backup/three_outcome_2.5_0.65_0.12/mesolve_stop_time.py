import sys
original_sys_path = sys.path.copy()
sys.path.append('../')
from utils import *


if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    square_t_stop = 594.3
    tot_time = int(square_t_stop + 30)
    tlist = np.linspace(0, tot_time, tot_time)

    qubit_level = 25
    osc_level = 50

    system = fluxonium_oscillator_system(
        EJ = 2.5,
        EC = 0.65,
        EL = 0.12,
        Er = 7.05877808,
        g_strength = 0.2,
        qubit_level = qubit_level,
        osc_level = osc_level,
        kappa = 1e-3
,
        products_to_keep=[[ql, ol] for ql in [1,2] for ol in range(25) ],
        computaional_states = '1,2',
        w_d = 7.0585
        )

    state_0_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(1,0)])
    state_1_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(2,0)])
    state_plus_dressed = (state_0_dressed  +  state_1_dressed).unit()
    state_minus_dressed = (state_0_dressed - state_1_dressed).unit()

    results = system.run_mesolve_on_driving_osc(initial_states = [system.truncate_function(state) for state in [state_plus_dressed, state_minus_dressed]],
                                                tlist = tlist,
                                                osc_decay = True,
                                                e_ops = [system.a_trunc,
                                                        system.a_trunc.dag()*system.a_trunc],
                                                amp = 0.0015,
                                                )


    with open(f'../pickles/mesolve_temp_2.5_kappa1em3.pkl', 'wb') as file:
        pickle.dump(results, file)
        

