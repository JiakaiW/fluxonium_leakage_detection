
import sys
sys.path.append('../')
from utils_models import *


if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    ql = 12
    ol = 20
    system = FluxoniumOscillatorSystem(
        computaional_states = '1,2',
        EJ = 2.33,
        EC = 0.69,
        EL = 0.12,
        Er = 7.16518677,
        g_strength = 0.18,
        qubit_level = ql,
        osc_level = ol,
        products_to_keep=[[ql, ol] for ql in [1,2] for ol in range(15) ],
    )

    tot_time =820
    tlist = np.linspace(0, tot_time, tot_time)


    state_0_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(1,0)])
    state_1_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(2,0)])
    state_plus_dressed = (state_0_dressed  +  state_1_dressed).unit()
    state_minus_dressed = (state_0_dressed - state_1_dressed).unit()
    initial_states  = [
        state_plus_dressed,
        state_minus_dressed
        ]

    results = system.run_mesolve_parrallel(
        initial_states = [system.truncate_function(state) for state in initial_states],
        tlist = tlist,
        drive_terms = [DriveTerm( 
                        driven_op=system.a_trunc + system.a_trunc.dag(),
                        pulse_shape_func=square_pulse_with_rise_fall,
                        pulse_shape_args={
                            'w_d': 7.1649,
                            'amp': 0.0013,
                            't_square': tot_time,
                        })],
        c_ops = [np.sqrt(1e-3) * system.a_trunc],
        e_ops=[
            system.a_trunc, system.a_trunc.dag()*system.a_trunc
        ],
        post_processing = ['pad_back','partial_trace_computational_states'],
    )


    import pickle
    with open('../pickles/mesolve_temp_kappa1em3_1649_compu_see_fidelity.pkl', 'wb') as file:
        pickle.dump(results, file)
