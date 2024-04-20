import sys
sys.path.append('../')
from utils_models import *


if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()
    max_ql = 12
    max_ol = 20

    t_stop = 360

    tlist = np.linspace(0,t_stop,t_stop)
    freqs = np.linspace(7.1714,7.1754,100)



    list_of_systems = []
    list_of_kwargs = []
    for initial_i , qls in zip([0,1,2],[[0], [1],[2] ,[3]]):
        system = FluxoniumOscillatorSystem(
            computaional_states = '1,2',
            EJ = 2.65,
            EC = 0.6,
            EL = 0.13,
            Er = 7.17391479 ,
            qubit_level = max_ql,
            osc_level = max_ol,
            g_strength = 0.12,
            products_to_keep=[[ql,ol] for ql in qls for ol in range(max_ol)] 
            )
        for freq in freqs:
            list_of_systems.append(system)
            list_of_kwargs.append( {
                'y0':system.truncate_function(qutip.basis(system.hilbertspace.dimension, initial_i)) ,
                'tlist':tlist,
                'drive_terms':[DriveTerm( 
                driven_op=system.truncate_function(system.hilbertspace.op_in_dressed_eigenbasis(system.osc.n_operator)),
                pulse_shape_func=square_pulse_with_rise_fall,
                pulse_shape_args={
                    'w_d': freq,
                    'amp': 0.002,
                    't_rise': 30,
                    't_square': 10000
                })],
                'e_ops':[system.a_trunc.dag()*system.a_trunc]
                }
            )
    results = run_parallel_ODEsolve_and_post_process_jobs_with_different_systems(
        list_of_systems,
        list_of_kwargs,
        post_processing = [], # Don't need padding back, only want photon number expectation
        store_states = False
    )

    with open(f'../pickles/12_response.pkl', 'wb') as file:
        pickle.dump(results, file)