
import sys
sys.path.append('../')
from utils_models import *

if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    max_ql = 30
    max_ol = 75
    EJ = 3
    EC = EJ/4
    EL = EJ/21
    Er = 8.32993958

    g = 0.27
    w_d = 8.330000924693827
    amp = 0.002

    tot_time =1000
 
    system  =  FluxoniumOscillatorSystem(
                    EJ = EJ,
                    EC = EC,
                    EL = EL,
                    Er = Er,
                    g_strength = g,
                    qubit_level = max_ql,
                    osc_level = max_ol,
                    products_to_keep=[[ql, ol] for ql in [1,2] for ol in range(max_ol) ],
                    computaional_states = '1,2',
                    )
 
    
    systems = [system, system]


   
    tlist = np.linspace(0, tot_time, tot_time)


    initial_states  = [
        qutip.basis(max_ql * max_ol, system.product_to_dressed[(ql,0)]) for ql in [1,2]
        ]

    list_of_systems = []
    list_of_kwargs = []
    for kappa in [1e-3]:
        for system, y0 in zip(systems, initial_states):
            list_of_systems.append(system)
            list_of_kwargs.append( {
                'y0':system.truncate_function(y0) ,
                'tlist':tlist,
                'drive_terms':[DriveTerm( 
                                        driven_op= system.driven_operator,
                                        pulse_shape_func=square_pulse_with_rise_fall,
                                        pulse_shape_args={
                                            'w_d': w_d ,
                                            'amp': amp,
                                            't_rise': 20,
                                            't_square': tot_time
                                        })],
                'e_ops':[system.a_trunc , system.a_trunc.dag()*system.a_trunc],
                'c_ops':[kappa *qutip.lindblad_dissipator(system.a_trunc) ]
                })
        

    results = run_parallel_ODEsolve_and_post_process_jobs_with_different_systems(
        list_of_systems,
        list_of_kwargs,
        post_processing = ['pad_back']
    )


    import pickle
    with open('../pickles/EJ3_compu.pkl', 'wb') as file:
        pickle.dump(results, file)
