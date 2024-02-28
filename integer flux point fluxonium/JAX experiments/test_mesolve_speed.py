import sys
sys.path.append('../')
from backup.utils import *


if __name__ == '__main__':
    # For Windows and MacOS compatibility:
    from multiprocessing import freeze_support
    freeze_support()

    system = fluxonium_oscillator_system(
        computaional_states = '1,2',
        products_to_keep = [[ql, ol] for ql in range(3) for ol in range(15) ] ,
        drive_transition=((0,0),(0,1))
    )

    result = system.run_mesolve_on_driving_osc(
        initial_states = [system.truncate_function(qutip.basis(system.hilbertspace.dimension, 1))], 
        tlist  =  np.linspace(0,100, 100), 
        osc_decay = True,
        post_processing = [],
        amp = 0.003
        )