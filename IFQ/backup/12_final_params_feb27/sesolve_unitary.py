
import sys
sys.path.append('../')
from utils_models import *


if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    max_ql = 12
    max_ol = 50
    system = FluxoniumOscillatorSystem(
        computaional_states = '1,2',
        EJ = 2.65,
        EC = 0.6,
        EL = 0.13,
        Er = 7.17391479,
        g_strength = 0.12,
        qubit_level = max_ql,
        osc_level = max_ol,
        products_to_keep=[[ql, ol] for ql in [1,2] for ol in range(50) ],
    )

    tot_time =1600
    tlist = np.linspace(0, tot_time, tot_time)


    state_0_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(1,0)])
    state_1_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(2,0)])
    state_plus_dressed = (state_0_dressed  +  state_1_dressed).unit()
    state_minus_i_dressed = (state_0_dressed - 1j * state_1_dressed).unit()
    initial_states  = [
        state_0_dressed,
        state_1_dressed,
        state_plus_dressed,
        state_minus_i_dressed
        ]
    


    results = system.run_mesolve_parrallel(
        initial_states = [system.truncate_function(state) for state in initial_states],
        tlist = tlist,
        drive_terms = [DriveTerm( 
                        # driven_op= -1j*(system.a_trunc - system.a_trunc.dag()),
                        driven_op= system.truncate_function(system.hilbertspace.op_in_dressed_eigenbasis(system.osc.n_operator)),
                        pulse_shape_func=square_pulse_with_rise_fall,
                        pulse_shape_args={
                            'w_d': 7.1730,
                            'amp': 0.005,
                            't_rise': 30,
                            't_square': tot_time,
                        })],
        c_ops = None,
        e_ops=[
            system.a_trunc, system.a_trunc.dag()*system.a_trunc
        ],
        post_processing = ['pad_back','partial_trace_computational_states'],
    )

    import pickle
    with open('../pickles/12_sesolve.pkl', 'wb') as file:
        pickle.dump(results, file)
