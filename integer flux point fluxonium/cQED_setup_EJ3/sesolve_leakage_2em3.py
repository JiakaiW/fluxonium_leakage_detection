
import sys
sys.path.append('../')
from utils_models import *
import pickle
if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    max_ql = 30
    max_ol = 80
    EJ = 3
    EC = EJ/4
    EL = EJ/21
    Er = 8.32993958

    g = 0.27
    w_d = 8.330000924693827
    amp = 0.002

    tot_time =1000

    system_leak =  FluxoniumOscillatorSystem(
                    EJ = EJ,
                    EC = EC,
                    EL = EL,
                    Er = Er,
                    g_strength = g, 
                    qubit_level = max_ql,
                    osc_level = max_ol,
                    products_to_keep=[[ql, ol] for ql in [0] for ol in range(max_ol) ],
                    computaional_states = '1,2',
                    )
    system_one =  FluxoniumOscillatorSystem(
                    EJ = EJ,
                    EC = EC,
                    EL = EL,
                    Er = Er,
                    g_strength = g,
                    qubit_level = max_ql,
                    osc_level = max_ol,
                    products_to_keep=[[ql, ol] for ql in [1] for ol in range(max_ol) ],
                    computaional_states = '1,2',
                    )

    system_two =  FluxoniumOscillatorSystem(
                    EJ = EJ,
                    EC = EC,
                    EL = EL,
                    Er = Er,
                    g_strength = g,
                    qubit_level = max_ql,
                    osc_level = max_ol,
                    products_to_keep=[[ql, ol] for ql in [2] for ol in range(max_ol) ],
                    computaional_states = '1,2',
                    )
    
    systems = [system_leak, system_one, system_two]


   
    tlist = np.linspace(0, tot_time, tot_time)

    kappa = 1e-3
    for ql, system in enumerate(systems):
        result = ODEsolve_and_post_process(
            y0=system.truncate_function(qutip.basis(max_ql * max_ol, system.product_to_dressed[(ql,0)])),
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
            c_ops = [kappa *qutip.lindblad_dissipator(system.a_trunc) ],
            e_ops = [system.a_trunc , system.a_trunc.dag()*system.a_trunc],
            method = 'qutip.mcsolve',
            file_name = 'try_mcsolve'
        )
    
        # with open('../pickles/EJ3_leak_2em3.pkl', 'wb') as file:
        #     pickle.dump(result, file)
