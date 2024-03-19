from utils_models import *
import dynamiqs as dq
dq.set_precision( 'double')
import jax.numpy as jnp
import numpy as np


if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    max_ql = 20
    max_ol = 20

    system = FluxoniumOscillatorSystem(
            computaional_states = '1,2',
            EJ = 2.65,
            EC = 0.6,
            EL = 0.13,
            Er = 7.17391479 ,
            qubit_level = max_ql,
            osc_level = max_ol,
            g_strength = 0.12,
            )
    result = system.run_dq_mesolve_parrallel(initial_states = [system.truncate_function(qutip.basis(system.hilbertspace.dimension, system.product_to_dressed[(i,0)])) for i in range(1)],
                        tlist = np.linspace(0, 5, 5), 
                        drive_terms = [DriveTerm( 
                                        driven_op=system.truncate_function(system.hilbertspace.op_in_dressed_eigenbasis(system.osc.n_operator)),
                                        pulse_shape_func=square_pulse_with_rise_fall,
                                        pulse_shape_args={
                                            'w_d': 7.1732,
                                            'amp': 0.002,
                                            't_rise': 30,
                                            't_square': 10000
                                        })],                        
                        c_ops  = None,
                        e_ops = [system.a_trunc.dag()*system.a_trunc],
                        post_processing = [],)