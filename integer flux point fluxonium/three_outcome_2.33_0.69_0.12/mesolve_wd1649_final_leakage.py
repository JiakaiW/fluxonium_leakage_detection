
import sys
original_sys_path = sys.path.copy()
sys.path.append('../')
from utils_models import *

if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    max_ql = 13
    max_ol = 50
    system_leak0 = FluxoniumOscillatorSystem(
        EJ = 2.33,
        EC = 0.69,
        EL = 0.12,
        Er = 7.16518677,
        g_strength = 0.18,
        qubit_level = max_ql,
        osc_level = max_ol,
        products_to_keep=[[ql, ol] for ql in [0] for ol in range(50) ] + \
                        [[ql, ol] for ql in [7] for ol in range(35) ] ,
        computaional_states = '1,2',
        )
    
    system_leak3 = FluxoniumOscillatorSystem(
        EJ = 2.33,
        EC = 0.69,
        EL = 0.12,
        Er = 7.16518677,
        g_strength = 0.18,
        qubit_level = max_ql,
        osc_level = max_ol,
        products_to_keep=[[ql, ol] for ql in [3] for ol in range(45) ]+ \
                        [[ql, ol] for ql in [12] for ol in range(25) ] ,
        computaional_states = '1,2',
        )


    initial_states  = [
        qutip.basis(system_leak0.hilbertspace.dimension, system_leak0.product_to_dressed[(0,0)]),
        qutip.basis(system_leak3.hilbertspace.dimension, system_leak3.product_to_dressed[(3,0)]),
        ]

    tot_time = 920
    tlist = np.linspace(0, tot_time, tot_time)

    list_of_systems = []
    list_of_kwargs = []
    for kappa in [1e-3]:
        for system, y0 in zip([system_leak0,system_leak3],initial_states):

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
                                            't_rise': 40,
                                            't_square': tot_time+10
                                        })],
                'e_ops':[system.a_trunc , system.a_trunc.dag()*system.a_trunc],
                'c_ops':[np.sqrt(kappa) * system.a_trunc]
                })
        

    results = run_parallel_ODEsolve_and_post_process_jobs_with_different_systems(
        list_of_systems,
        list_of_kwargs,
        post_processing = ['pad_back']
    )


    import pickle
    with open('../pickles/mesolve_wd1649_final_leakage_new_op.pkl', 'wb') as file:
        pickle.dump(results, file)
