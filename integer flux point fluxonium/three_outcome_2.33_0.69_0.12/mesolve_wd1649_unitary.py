
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
        EJ = 2.33,
        EC = 0.69,
        EL = 0.12,
        Er = 7.16518677,
        g_strength = 0.18,
        qubit_level = max_ql,
        osc_level = max_ol,
        products_to_keep=[[ql, ol] for ql in [0,3] for ol in range(max_ol) ],
    )

    tot_time =900
    tlist = np.linspace(0, tot_time, tot_time)


    initial_states  = [
        qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(0,0)]),
        qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(3,0)]),
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
                            't_square': 900,
                        })],
        c_ops = None,
        e_ops=[
            system.a_trunc, system.a_trunc.dag()*system.a_trunc
        ],
        post_processing = ['pad_back','partial_trace_eigen_states'],
    )


    import pickle
    with open('../pickles/mesolve_temp_1649_see_domi_frequency.pkl', 'wb') as file:
        pickle.dump(results, file)
