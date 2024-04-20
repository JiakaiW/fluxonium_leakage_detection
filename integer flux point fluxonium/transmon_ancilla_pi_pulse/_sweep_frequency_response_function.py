import sys
sys.path.append('../')
from utils_models import *


if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()
    max_fl = 10
    max_tl = 5

    t_stop = 856

    tlist = np.linspace(0,t_stop,t_stop)
    w_t_ori = 7.190996762003433
    freqs = np.linspace(w_t_ori - 4e-3,w_t_ori + 4e-3,100)

    fluxonium = scqubits.Fluxonium(EJ=2.7,
                         EC=0.6,
                         EL=0.13,
                         flux=0,cutoff=110,
                         truncated_dim=20)
    tune_tmon = scqubits.TunableTransmon(
        EJmax=50.0,
        EC=0.5,
        d=0.01,
        flux=0.403471,
        ng=0.0,
        ncut=30
        )

    system = FluxoniumTunableTransmonSystem(
        
        fluxonium  = fluxonium,
        tune_tmon = tune_tmon,
        computaional_states = '1,2',
        g_strength = 0.2,
        )
    
    list_of_systems = []
    list_of_kwargs = []
    for initial_i , qls in zip([0,1,2],
                               [[0], [1],[2]]):

        for freq in freqs:
            list_of_systems.append(system)
            list_of_kwargs.append( {
                'y0':system.truncate_function(qutip.basis(system.hilbertspace.dimension, system.hilbertspace.dressed_index((initial_i,0)))) ,
                'tlist':tlist,
                'drive_terms':[DriveTerm( 
                driven_op=system.truncate_function(system.hilbertspace.op_in_dressed_eigenbasis(system.tune_tmon.n_operator)),
                pulse_shape_func=square_pulse_with_rise_fall,
                pulse_shape_args={
                    'w_d': freq,
                    'amp': 0.002,
                    't_rise': 30,
                    't_square': 856.6969140188583 - 30
                })],
                'e_ops':[system.truncate_function(qutip.basis(system.hilbertspace.dimension, system.hilbertspace.dressed_index((initial_i,1)))) * system.truncate_function(qutip.basis(system.hilbertspace.dimension, system.hilbertspace.dressed_index((initial_i,1)))).dag()]
                }
            )
    results = run_parallel_ODEsolve_and_post_process_jobs_with_different_systems(
        list_of_systems,
        list_of_kwargs,
        post_processing = [], # Don't need padding back, only want photon number expectation
        store_states = False
    )

    with open(f'../pickles/transmon_response.pkl', 'wb') as file:
        pickle.dump(results, file)