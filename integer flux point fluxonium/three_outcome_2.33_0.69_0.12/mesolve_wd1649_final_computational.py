
import sys
original_sys_path = sys.path.copy()
sys.path.append('../')
from utils_models import *

if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    max_ql = 20
    max_ol = 50
    system_computational = FluxoniumOscillatorSystem(
        EJ = 2.33,
        EC = 0.69,
        EL = 0.12,
        Er = 7.16518677,
        g_strength = 0.18,
        qubit_level = max_ql,
        osc_level = max_ol,
        products_to_keep=[[ql, ol] for ql in [1,2] for ol in range(20) ]+ \
                        [[ql, ol] for ql in [9,10,11] for ol in range(12) ] ,
        computaional_states = '1,2',
        )


    state_0_dressed = qutip.basis(system_computational.hilbertspace.dimension, system_computational.product_to_dressed[(1,0)])
    state_1_dressed = qutip.basis(system_computational.hilbertspace.dimension, system_computational.product_to_dressed[(2,0)])
    state_plus_dressed = (state_0_dressed  +  state_1_dressed).unit()
    state_minus_dressed = (state_0_dressed - state_1_dressed).unit()
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

    tot_time = 840
    tlist = np.linspace(0, tot_time, tot_time)

    list_of_systems = []
    list_of_kwargs = []
    system = system_computational
    for kappa in [1e-3,5e-4]:
        for system, y0 in initial_states:
            list_of_systems.append(system)
            list_of_kwargs.append( {
                'y0':system.truncate_function(y0) ,
                'tlist':tlist,
                'drive_terms':[DriveTerm( 
                                        driven_op=system.a_trunc + system.a_trunc.dag(),
                                        pulse_shape_func=square_pulse_with_rise_fall,
                                        pulse_shape_args={
                                            'w_d': 7.1649,
                                            'amp': 0.0013,
                                            't_square': tot_time+10
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
    with open('../pickles/mesolve_wd1649_final_computational.pkl', 'wb') as file:
        pickle.dump(results, file)
