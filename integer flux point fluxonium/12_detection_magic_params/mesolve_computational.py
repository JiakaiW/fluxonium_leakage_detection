
import sys
original_sys_path = sys.path.copy()
sys.path.append('../')
from utils_models import *

if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    max_ql = 21
    max_ol = 98

    system_computational = FluxoniumOscillatorSystem(
        EJ = 2.75,
        EC = 0.6,
        EL = 0.13,
        Er = 7.20701708,
        g_strength = 0.23,
        qubit_level = max_ql,
        osc_level = max_ol,
        products_to_keep=[[ql, ol] for ql in [1,2] for ol in range(25) ],
        computaional_states = '1,2',
        )


    tot_time =700
    tlist = np.linspace(0, tot_time, tot_time)



    state_0_dressed = qutip.basis(system_computational.hilbertspace.dimension, system_computational.product_to_dressed[(1,0)])
    state_1_dressed = qutip.basis(system_computational.hilbertspace.dimension, system_computational.product_to_dressed[(2,0)])
    state_plus_dressed = (state_0_dressed  +  state_1_dressed).unit()
    state_minus_dressed = (state_0_dressed  -  state_1_dressed).unit()
    state_plus_i_dressed = (state_0_dressed + 1j * state_1_dressed).unit()
    state_minus_i_dressed = (state_0_dressed - 1j * state_1_dressed).unit()
    initial_states  = [
        state_0_dressed,
        state_1_dressed,
        state_plus_dressed,
        state_minus_dressed,
        state_plus_i_dressed,
        state_minus_i_dressed,

        state_0_dressed * state_1_dressed.dag(),
        state_1_dressed * state_0_dressed.dag(),
        ]

    list_of_systems = []
    list_of_kwargs = []
    system = system_computational
    for kappa in [1e-3]:
        for y0 in initial_states:
            list_of_systems.append(system)
            list_of_kwargs.append( {
                'y0':system.truncate_function(y0) ,
                'tlist':tlist,
                'drive_terms':[DriveTerm( 
                                        driven_op= system.truncate_function(system.hilbertspace.op_in_dressed_eigenbasis(system.osc.n_operator)),
                                        # driven_op=  -1j*(system.a_trunc - system.a_trunc.dag())  ,
                                        # driven_op=  system.a_trunc + system.a_trunc.dag()  ,
                                        pulse_shape_func=square_pulse_with_rise_fall,
                                        pulse_shape_args={
                                            'w_d': 7.20666,
                                            'amp': 0.003,
                                            't_rise': 40,
                                            't_square': tot_time
                                        })],
                'e_ops':[system.a_trunc , system.a_trunc.dag()*system.a_trunc],
                'c_ops':[np.sqrt(kappa) * system.a_trunc]
                })
        

    results = run_parallel_ODEsolve_and_post_process_jobs_with_different_systems(
        list_of_systems,
        list_of_kwargs,
        post_processing = ['pad_back','partial_trace_computational_states']
    )


    import pickle
    with open('../pickles/magic_compu_1em3.pkl', 'wb') as file:
        pickle.dump(results, file)
