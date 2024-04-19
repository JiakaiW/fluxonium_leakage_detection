
import sys
sys.path.append('../')
from utils_models import *
import pickle
if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    max_ql = 30
    max_ol = 75
    EJ = 3
    EC = EJ/4
    EL = EJ/20.5
    Er = 8.46111172

    g = 0.2
    w_d = 8.460155465243822
    amp = 0.003

    tot_time =700
    tlist = np.linspace(0, tot_time, tot_time)
    kappa = 1e-3

    for initial_i in range(3):
        system =  FluxoniumOscillatorSystem(
                        EJ = EJ,
                        EC = EC,
                        EL = EL,
                        Er = Er,
                        g_strength = g, 
                        qubit_level = max_ql,
                        osc_level = max_ol,
                        products_to_keep=[[ql, ol] for ql in [initial_i] for ol in range(max_ol) ],
                        computaional_states = '1,2',
                        )
        print('finished setting up system')
        result = ODEsolve_and_post_process(
            y0=system.truncate_function(qutip.basis(max_ql * max_ol, system.product_to_dressed[(initial_i,0)])),
            tlist = tlist,
            static_hamiltonian=system.diag_dressed_hamiltonian,
            drive_terms = [DriveTerm( 
                                driven_op= system.driven_operator,
                                pulse_shape_func=square_pulse_with_rise_fall,
                                pulse_shape_args={
                                    'w_d': w_d ,
                                    'amp': amp,
                                    't_rise': 20,
                                    't_square': tot_time
                                })],
            # c_ops = [kappa *qutip.lindblad_dissipator(system.a_trunc) ],
            c_ops = [np.sqrt(kappa) * system.a_trunc],
            # e_ops = [system.a_trunc , system.a_trunc.dag()*system.a_trunc],
            method = 'qutip.mcsolve',
            file_name = 'try_mcsolve'
        )
    
        # with open('../pickles/EJ3_leak_2em3.pkl', 'wb') as file:
        #     pickle.dump(result, file)
