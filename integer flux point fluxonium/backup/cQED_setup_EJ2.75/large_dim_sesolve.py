
import sys
sys.path.append('../')
from utils_models import *


if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    max_ql = 21
    max_ol = 50

    system = FluxoniumOscillatorSystem(
        EJ = 2.75,
        EC = 0.6,
        EL = 0.13,
        Er = 7.20701708,
        g_strength = 0.23,
        qubit_level = max_ql,
        osc_level = max_ol,
        products_to_keep=[[ql, ol] for ql in range(max_ql) for ol in range(max_ol) ],
        computaional_states = '1,2',
        )
    tot_time =700
    tlist = np.linspace(0, tot_time, tot_time)

    state_leakage_dressed =  qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(0,0)])
    state_0_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(1,0)])
    state_1_dressed = qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(2,0)]) 
    initial_states  = [
        state_leakage_dressed,
        state_0_dressed,
        state_1_dressed
        ]
    


    results = system.run_mesolve_parrallel(
        initial_states = [system.truncate_function(state) for state in initial_states],
        tlist = tlist,
        drive_terms = [DriveTerm( 
                        # driven_op= (system.a_trunc + system.a_trunc.dag()),
                        driven_op= system.truncate_function(system.hilbertspace.op_in_dressed_eigenbasis(system.osc.n_operator)),
                        pulse_shape_func=square_pulse_with_rise_fall,
                        pulse_shape_args={
                                            'w_d': 7.20666,
                                            'amp': 0.003,
                                            't_rise': 40,
                                            't_square': tot_time
                                        })],
        c_ops = None,
        e_ops=[
            system.a_trunc, system.a_trunc.dag()*system.a_trunc
        ],
        post_processing = ['pad_back'],
    )

    import pickle
    with open('../pickles/magic_large_dim.pkl', 'wb') as file:
        pickle.dump(results, file)
