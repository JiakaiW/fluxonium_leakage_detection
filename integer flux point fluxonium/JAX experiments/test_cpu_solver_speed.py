
import os 
os.environ['JAX_JIT_PJIT_API_MERGE'] = '0'
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

import sys
sys.path.append('../')
from utils import *


if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    system = fluxonium_oscillator_system(
        computaional_states = '1,2',
        products_to_keep = [[ql, ol] for ql in range(3) for ol in range(15) ] ,
        drive_transition=((0,0),(0,1))
    )


    result = system.run_jax_cpu_solve(
        initial_state = system.truncate_function(qutip.basis(system.hilbertspace.dimension, 1)), 
        tlist  =  jnp.linspace(0,100, 100), 
        osc_decay = True,
        amp = 0.003
        )
