
import os

os.environ['JAX_JIT_PJIT_API_MERGE'] = '0'
import jax.numpy as jnp
from jax import jit, vmap
import qiskit.pulse
from qiskit_dynamics.array import Array
Array.set_default_backend('jax')
from qiskit_dynamics import Solver
from qiskit_dynamics.models import HamiltonianModel, LindbladModel
from qiskit_dynamics.solvers.solver_classes import validate_and_format_initial_state


from bidict import bidict
import concurrent
import importlib.util
import ipywidgets as widgets

import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import numpy as np
import pickle
import qutip
import scqubits
import sympy as sym
from typing import List, Any, Union, Tuple
from tqdm.notebook import tqdm
from tqdm import tqdm
import uuid




############################################################################
#
#
# fluxonium_oscillator_system and run_fluxonium_osc_system_mesolve_jobs are used
#  to reduce duplicating code about creating qutip objects and running simulations
#
############################################################################


class fluxonium_oscillator_system:
    '''
    Because the code for initializing the system, do truncation, and run mesovle is re-used so often, 
    I decide to create a class for it
    '''
    def __init__(self,
                computaional_states:str, # = '0,1' or '1,2'
                drive_transition: Tuple[int] = None,
                EJ:float = 3,
                EC:float = 0.6,
                EL:float = 0.13,
                Er:float = 7.2622522,
                g_strength:float = 0.3,
                qubit_level:float = 30,
                osc_level:float = 30,
                kappa = 0.001,
                products_to_keep: List[List[int]]= None,
                w_d:float = None
                ):
        '''
        Initialize objects before truncation
        '''
        self.qubit_level = qubit_level
        self.osc_level = osc_level
        self.qbt = scqubits.Fluxonium(EJ=EJ,EC=EC,EL=EL,flux=0,cutoff=110,truncated_dim=qubit_level)
        self.osc = scqubits.Oscillator(E_osc=Er,truncated_dim=osc_level)
        self.hilbertspace = scqubits.HilbertSpace([self.qbt, self.osc])
        self.hilbertspace.add_interaction(g_strength=g_strength,op1=self.qbt.n_operator,op2=self.osc.creation_operator,add_hc=True)
        self.hilbertspace.generate_lookup()
        self.product_to_dressed = generate_single_mapping(self.hilbertspace.hamiltonian())
        self.computaional_states = [int(computaional_states[0]),int(computaional_states[-1])]
        '''
        Define how to truncate
        '''
        if products_to_keep != None:
            self.products_to_keep = products_to_keep
        else:
            self.products_to_keep = [[ql, ol] for ql in [1,2,3] for ol in range(10) ] \
                                + [[ql, ol] for ql in [0] for ol in range(30) ]

        
        '''
        Truncate objects
        '''
        self.a = qutip.Qobj(self.hilbertspace.op_in_dressed_eigenbasis(self.osc.annihilation_operator)[:, :])
        self.a_trunc = self.truncate_function(self.a)
        self.evals = self.hilbertspace["evals"][0]
        self.diag_dressed_hamiltonian = self.truncate_function(qutip.Qobj((
                2 * np.pi * qutip.Qobj(np.diag(self.evals),
                dims=[self.hilbertspace.subsystem_dims] * 2)
        )[:, :]))
        if w_d!= None:
            self.w_d = w_d
        elif drive_transition!= None:
            self.w_d = transition_frequency(self.hilbertspace,self.product_to_dressed[drive_transition[0]],self.product_to_dressed[drive_transition[1]] ) 
        elif computaional_states == '1,2':
            self.w_d = transition_frequency(self.hilbertspace,self.product_to_dressed[(0,0)],self.product_to_dressed[(0,1)] ) 
        elif computaional_states == '0,1':
            self.w_d = transition_frequency(self.hilbertspace,self.product_to_dressed[(2,0)],self.product_to_dressed[(2,1)] ) 
        else:
            raise Exception('computaional_states not supported')
        self.c_ops = [np.sqrt(kappa) * self.a_trunc]
        

    def truncate_function(self,qobj):
        return truncate_custom(qobj, self.products_to_keep, self.product_to_dressed)
    
    def pad_back_function(self,qobj):
        return pad_back_custom(qobj, self.products_to_keep, self.product_to_dressed)
        
    def run_mesolve_on_driving_osc(self,
                    initial_states, # truncated initial states
                    tlist,
                    osc_decay = False,
                    e_ops = None,
                    amp = 0.004,
                    max_workers = None,
                    t_stop=None,
                    t_rise=None,
                    post_processing = ['pad_back'],
                    driven_operator = None,
                    ):
        '''
        This function runs mesolve on multiple initial states using multi-processing,
          and return a list of qutip.solver.result
        '''
        
        if driven_operator == None:
            driven_operator = self.a_trunc+self.a_trunc.dag()
        H_with_drive = [
            self.diag_dressed_hamiltonian,
            [driven_operator, square_cos_with_rise_fall],
        ]
        additional_args = {'w_d': self.w_d, 'amp': amp, 't_stop': t_stop, 't_rise': t_rise}
        post_processing_funcs = []
        post_processing_args = []
        if 'pad_back' in post_processing:
            post_processing_funcs.append(pad_back_custom)
            post_processing_args.append((self.products_to_keep, 
                                self.product_to_dressed))

        results = [None] * len(initial_states)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(mesolve_and_post_processing, 
                                       rho0=initial_states[i], 
                                       H_with_drive=H_with_drive,
                                       tlist=tlist,
                                       additional_args = additional_args,
                                       c_ops=self.c_ops if osc_decay else None,
                                       e_ops = e_ops,

                                        post_processing_funcs=post_processing_funcs,
                                        post_processing_args=post_processing_args,
                                       ): i for i in range(len(initial_states))}
            
            for future in concurrent.futures.as_completed(futures):
                original_index = futures[future]
                results[original_index] = future.result()

        return results
    
    def run_mcsolve_on_driving_osc(self,
                            initial_state,
                            tlist,
                            osc_decay = True,
                            e_ops = None,
                            amp = 0.004,
                            t_stop=None,
                            t_rise=None,
                            driven_operator = None,
                            ):
        '''
        This function runs mcsolve on one initial states return one qutip.solver.result
        '''
        if driven_operator == None:
            driven_operator = self.a_trunc+self.a_trunc.dag()
        H_with_drive = [
            self.diag_dressed_hamiltonian,
            [driven_operator, square_cos_with_rise_fall],
        ]
        additional_args = {'w_d': self.w_d, 'amp': amp, 't_stop': t_stop, 't_rise': t_rise}

        result = qutip.mcsolve(psi0=initial_state, 
                                   H=H_with_drive,
                                   progress_bar = qutip.ui.progressbar.EnhancedTextProgressBar(),
                                   tlist=tlist,
                                   args = additional_args,
                                   c_ops=self.c_ops if osc_decay else None,
                                   ntraj = 500,
                                   e_ops = e_ops,
                                   options=qutip.Options(store_states=True,num_cpus = None)
                                   )
        return result

    def run_jax_gpu_solve(self,
                    initial_state, 
                    tlist, 
                    osc_decay = True,
                    amp = 0.004,
                    t_stop=None,
                    t_rise=None,
                    driven_operator = None,
                    chunk_size=1,
                    signal_sample_dt = 0.1):
        ########################################################################################
        #
        # 1) truncating through time is still needed as of Feb 18 2024, if the t_span used to
        #    call ham_solver.solve os too large you get VRAM issue (max_dt won't save it)
        #
        ########################################################################################
        if driven_operator == None:
            driven_operator = self.a_trunc+self.a_trunc.dag()

        initial_state = qobj_to_Array(initial_state)
        if osc_decay:
            ham_solver = Solver(
                hamiltonian_operators=[qobj_to_Array(driven_operator)],
                static_hamiltonian=2 * np.pi * qobj_to_Array(self.diag_dressed_hamiltonian),
                static_dissipators= [qobj_to_Array(op) for op in self.c_ops],
                hamiltonian_channels=['d0'],
                channel_carrier_freqs={'d0': self.w_d * 2 * np.pi},
                dt=signal_sample_dt,
                evaluation_mode = "dense_vectorized"
            )
            # if ham_solver.model.dim > 52:
            #     raise Exception('dimension too large for gpu evolution with superoperators')
            if initial_state.shape[0] != ham_solver.model.dim**2:
                initial_state = initial_state @ initial_state.conj().T  
                initial_state = initial_state.flatten(order='F')  
        else:
            ham_solver = Solver(
                hamiltonian_operators=[qobj_to_Array(driven_operator)],
                static_hamiltonian=2 * np.pi * qobj_to_Array(self.diag_dressed_hamiltonian),
                hamiltonian_channels=['d0'],
                channel_carrier_freqs={'d0': self.w_d * 2 * np.pi},
                dt=signal_sample_dt,
                evaluation_mode = "dense"
            )
            # if ham_solver.model.dim > 3500:
            #     raise Exception('dimension too large for gpu even just for unitary evolution')
        try:
            validate_and_format_initial_state(initial_state, ham_solver.model)
        except:
            print(f"y0 shape: {initial_state.shape}, model shape {ham_solver.model.dim}")
        
        max_dt = 1

        signals = get_qiskit_square_pulse_with_sin_squared_edges(
                                                w_d_without_2pi = self.w_d,
                                                amp_without_2pi = amp,
                                                t_rise = t_rise if t_rise is not None else 0,
                                                t_stop = t_stop if t_stop is not None else tlist[-1]
                                                )

        if chunk_size >= 1:
            ###################################
            # Here the tlist of the result is still the original tlist
            ###################################
            tlist_chunks = [tlist[i:i + chunk_size] for i in range(0, len(tlist), chunk_size)]
            current_state = initial_state
            chunk_results = []

            for chunk in tqdm(tlist_chunks,desc=f"solving through chunks"):
                result = ham_solver.solve(
                    y0=current_state, 
                    t_span=[chunk[0], chunk[-1]],
                    signals=signals, 
                    method='jax_expm_parallel', 
                    t_eval=jnp.linspace(chunk[0], chunk[-1], len(chunk)), 
                    max_dt=max_dt 
                )

                if chunk_results == []:
                    chunk_results.extend(result.y)
                else:
                    chunk_results.extend(result.y[1:])

                current_state = result.y[-1]
            ode_result = qutip.solver.Result()
            ode_result.times=tlist
            ode_result.states=chunk_results
            return ode_result
        else:
            ###################################
            # Here the tlist of the result is a new one
            ###################################
            current_state = initial_state
            chunk_results = []
            t_results = []
            total_time = tlist[-1] - tlist[0]
            num_intervals = int(len(tlist) / chunk_size)
            interval_length = total_time / num_intervals

            for i in tqdm(range(num_intervals),desc=f"solving through chunks"):
                t_start = tlist[0] + i * interval_length
                t_end = t_start + interval_length
                t_eval_interval = jnp.array([t_end])
                print('solving with jax_expm_parallel')
                result = ham_solver.solve(
                    y0=current_state,
                    t_span=[t_start, t_end],
                    signals=signals,
                    method='jax_expm_parallel',
                    t_eval=t_eval_interval,
                    max_dt=max_dt
                )

                chunk_results.append(result.y[-1])
                t_results.append(t_eval_interval[-1])
                current_state = result.y[-1] 
            
            ode_result = qutip.solver.Result()
            ode_result.times=t_results
            ode_result.states=chunk_results
            return ode_result



    def run_jax_cpu_solve(self,
                    initial_state, 
                    tlist, 
                    osc_decay = True,
                    amp = 0.004,
                    t_stop=None,
                    t_rise=None,
                    driven_operator = None,
                    chunk_size=1,
                    signal_sample_dt = 0.1):
        ########################################################################################
        #
        # 1) truncating through time is still needed as of Feb 18 2024, if the t_span used to
        #    call ham_solver.solve os too large you get VRAM issue (max_dt won't save it)
        #
        ########################################################################################
        if driven_operator == None:
            driven_operator = self.a_trunc+self.a_trunc.dag()

        initial_state = qobj_to_Array(initial_state)
        
        ham_solver = Solver(
            hamiltonian_operators=[qobj_to_Array(driven_operator)],
            static_hamiltonian=2 * np.pi * qobj_to_Array(self.diag_dressed_hamiltonian),
            static_dissipators= [qobj_to_Array(op) for op in self.c_ops] if  osc_decay else None,
            hamiltonian_channels=['d0'],
            channel_carrier_freqs={'d0': self.w_d * 2 * np.pi},
            dt=signal_sample_dt,
            evaluation_mode = "sparse"
        )
        if osc_decay and initial_state.shape[0] != ham_solver.model.dim**2:
            initial_state = initial_state @ initial_state.conj().T  

        signals = get_qiskit_square_pulse_with_sin_squared_edges(
                                                w_d_without_2pi = self.w_d,
                                                amp_without_2pi = amp,
                                                t_rise = t_rise if t_rise is not None else 0,
                                                t_stop = t_stop if t_stop is not None else tlist[-1]
                                                )

        result = ham_solver.solve(
                y0=initial_state, 
                t_span=[0, tlist[-1]],
                signals=signals, 
                method='jax_odeint', 
                t_eval=tlist, 
            )

        ode_result = qutip.solver.Result()
        ode_result.times= result.tlist
        ode_result.states= [qutip.Qobj(state) for state in result.y]
        return ode_result




def square_cos(t, args):
    w_d = args['w_d']
    amp = args['amp']
    t_stop = args.get('t_stop', None)
    if t_stop != None:
        if t <= t_stop:
            cos = np.cos(w_d * 2 * np.pi * t)
            return 2 * np.pi * amp * cos
        else:
            return 0
    else:
        cos = np.cos(w_d * 2 * np.pi * t)
        return 2 * np.pi * amp * cos
    
def square_cos_with_rise_fall(t, args):
    w_d = args['w_d']
    amp = args['amp']
    t_rise = args.get('t_rise', 0)  # Default rise time is 0 for no rise
    t_stop = args.get('t_stop', None)
    if t_rise == None:
        t_rise = 0
    # Function for the cosine modulation
    def cos_modulation():
        return 2 * np.pi * amp * np.cos(w_d * 2 * np.pi * t)

    # Check if rise and fall times are specified
    if t_rise > 0 and t_stop is None:
        raise Exception('t_rise doesnt work when t_stop isnt provided')
    elif t_rise > 0 and t_stop is not None:
        t_fall_start = t_stop - t_rise  # Calculate when the fall should start
        if 0 <= t <= t_rise:  # Rise segment
            return np.sin(np.pi * t / (2 * t_rise))**2 * cos_modulation()
        elif t_rise < t <= t_fall_start:  # Constant amplitude segment
            return cos_modulation()
        elif t_fall_start < t <= t_stop:  # Fall segment
            return np.sin(np.pi * (t_stop - t) / (2 * t_rise))**2 * cos_modulation()
        else:
            return 0
    else:  # No rise or fall, behave like the original function
        return cos_modulation() if t_stop is None or t <= t_stop else 0

def get_qiskit_square_pulse_with_sin_squared_edges(w_d_without_2pi,amp_without_2pi = 0.03, t_rise = 15,t_stop = 100, draw = False):
    t_square =  t_stop - 2 * t_rise
    signal_sample_dt = 0.1 # Sample rate of the backend in ns.
    base_drive_amplitude = amp_without_2pi * 2 * np.pi
    _t = sym.symbols('t')
    _amp = sym.symbols('amp')
    _duration = sym.symbols('duration')
    with qiskit.pulse.build(name="square") as square:
        if t_rise > 0:
            qiskit.pulse.play(
                qiskit.pulse.library.ScalableSymbolicPulse(
                    pulse_type="SinSquaredRingUp",
                    duration= int(t_rise/signal_sample_dt), 
                    amp=base_drive_amplitude, 
                    angle=0,
                    envelope= _amp * sym.sin(sym.pi * _t  / (2 * t_rise / signal_sample_dt) )**2,
                ), 
                qiskit.pulse.DriveChannel(0)
            )
        qiskit.pulse.play(
            qiskit.pulse.Constant(
                duration = int(t_square/signal_sample_dt), 
                amp = base_drive_amplitude
            ), 
            qiskit.pulse.DriveChannel(0)
        )
        if t_rise > 0:
            qiskit.pulse.play(
                qiskit.pulse.library.ScalableSymbolicPulse(
                    pulse_type="SinSquaredRingDown",
                    duration=int(t_rise/signal_sample_dt), 
                    amp=base_drive_amplitude,
                    angle=0, 
                    envelope=  _amp* sym.sin(sym.pi * (_duration - _t ) / (2 * t_rise / signal_sample_dt))**2,
                ), 
                qiskit.pulse.DriveChannel(0)
            )
    if draw:
        square.draw()
        from qiskit_dynamics.pulse import InstructionToSignals
        converter = InstructionToSignals(signal_sample_dt, carriers={"d0": carrier_freq})
        signals = converter.get_signals(square)
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, title in zip(axs, ["envelope", "signal"]):
            signals[0].draw(0, tot_time, 2000, title, axis=ax)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Amplitude")
            ax.set_title(title)
    return square

def mesolve_and_post_processing(
            rho0,
            H_with_drive,
            tlist, 
            additional_args,
            c_ops = None,
            e_ops = None,
            post_processing_funcs=[],
            post_processing_args=[],
            ):
    result = qutip.mesolve(
        H=H_with_drive,
        rho0=rho0,
        tlist=tlist,
        c_ops=c_ops,
        e_ops = e_ops,
        args=additional_args,
        options=qutip.Options(store_states=True, nsteps=80000, num_cpus=1),
        progress_bar = qutip.ui.progressbar.EnhancedTextProgressBar(),
    )
    for func, args in zip(post_processing_funcs, post_processing_args):
        result.states = [func(state, *args) for state in tqdm(result.states, desc=f"Processing states with {func.__name__}")]


    return result



def run_fluxonium_osc_system_mesolve_jobs(
        list_of_systems: List[fluxonium_oscillator_system],
        list_of_kwargs: list[Any],
        max_workers = None,
        post_processing = ['pad_back'],
    ):
    '''
    This function helps run mesolve of different fluxonium_oscillator_system and different initial states concurrently
    Args:
        list_of_systems: list of fluxonium_oscillator_system
        list_of_kwargs: list of kwargs dictionaries used to call run_mesolve_on_driving_osc
            a single kwargs should be a dictionary like {'intial_state',
                                                        'tlist',
                                                        'osc_decay' : False,
                                                        'e_ops' : None,
                                                        'amp' : 0.004,
                                                        't_stop':None,
                                                        'driven_operator': system.a_trunc+system.a_trunc.dag()}
    '''
    assert len(list_of_systems) == len(list_of_kwargs)
    # Step-1, define functions
    list_of_H_with_drive = []
    list_of_additional_args = []
    for system,kwargs in zip(list_of_systems,list_of_kwargs):
        driven_operator = kwargs.get('driven_operator',system.a_trunc+system.a_trunc.dag())
        H_with_drive = [
            system.diag_dressed_hamiltonian,
            [driven_operator, square_cos_with_rise_fall],
        ]
        list_of_H_with_drive.append(H_with_drive)
        list_of_additional_args.append({'w_d': system.w_d, 'amp': kwargs['amp'], 't_stop': kwargs.get('t_stop',None), 't_rise': kwargs.get('t_rise',None)})

    results = [None] * len(list_of_systems)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i in range(len(list_of_systems)):
            post_processing_funcs = []
            post_processing_args = []
            
            if 'pad_back' in post_processing:
                post_processing_funcs.append(pad_back_custom)
                post_processing_args.append((list_of_systems[i].products_to_keep, 
                                            list_of_systems[i].product_to_dressed))
            if 'partial_trace_computational_states' in post_processing:
                post_processing_funcs.append(dressed_to_2_level_dm)
                post_processing_args.append((
                                            list_of_systems[i].product_to_dressed,
                                            list_of_systems[i].qubit_level, 
                                            list_of_systems[i].osc_level,
                                            list_of_systems[i].computaional_states[0],
                                            list_of_systems[i].computaional_states[1],
                                            None))
            future = executor.submit(
                mesolve_and_post_processing, 
                rho0=list_of_kwargs[i]['intial_state'], 
                H_with_drive=list_of_H_with_drive[i],
                tlist=list_of_kwargs[i]['tlist'], 
                additional_args=list_of_additional_args[i],
                c_ops=list_of_systems[i].c_ops if list_of_kwargs[i]['osc_decay'] else None,
                e_ops=list_of_kwargs[i].get('e_ops', None),

                post_processing_funcs=post_processing_funcs,
                post_processing_args=post_processing_args,
            )
            futures[future] = i
        
        for future in concurrent.futures.as_completed(futures):
            original_index = futures[future]
            results[original_index] = future.result()
    return results




############################################################################
#
#
# Functions about manipulating qutip/scqubit objects
#
#
############################################################################

def truncate_custom(qobj: qutip.Qobj, products_to_keep: list, product_to_dressed: dict) -> qutip.Qobj:
    indices_to_keep = [dressed_level for (qubit_level, oscillator_level), dressed_level in product_to_dressed.items() if [qubit_level, oscillator_level] in products_to_keep]
    try:
        indices_to_keep.sort()
    except:
        print(indices_to_keep)
    if qobj.shape[1] == 1:  # is ket
        truncated_vector = qobj.full()[indices_to_keep, :]
        return qutip.Qobj(truncated_vector)
    else:  # is operator or density matrix
        truncated_matrix = qobj.full()[np.ix_(indices_to_keep, indices_to_keep)]
        return qutip.Qobj(truncated_matrix)

def pad_back_custom(qobj: qutip.Qobj, products_to_keep: Union[list,None], product_to_dressed: dict) -> qutip.Qobj:
    if products_to_keep == None:
        # for compatibility
        return qobj
    indices_to_keep = [dressed_level for (qubit_level, oscillator_level), dressed_level in product_to_dressed.items() if [qubit_level, oscillator_level] in products_to_keep]
    indices_to_keep.sort()

    full_dimension = max(product_to_dressed.values()) + 1

    if qobj.shape[1] == 1:  # is ket
        padded_vector = np.zeros((full_dimension, 1), dtype=complex)
        padded_vector[indices_to_keep, :] = qobj.full()
        return qutip.Qobj(padded_vector)
    else:  # is operator or density matrix
        padded_matrix = np.zeros((full_dimension, full_dimension), dtype=complex)
        padded_matrix[np.ix_(indices_to_keep, indices_to_keep)] = qobj.full()
        return qutip.Qobj(padded_matrix)

def test_truncate_and_pad_custom():
    # Define the mapping between product basis states and dressed states
    product_to_dressed = {(0, 0): 0, 
                            (1, 0): 1, 
                            (0, 1): 2, 
                            (1, 1): 3, 
                            (2, 0): 4, 
                            (0, 2): 5, 
                            (1, 2): 6, 
                            (2, 1): 7, 
                            (2, 2): 8}

    # Specify which products to keep
    products_to_keep = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]

    # Create an example qubit operator
    qubit_operator = qutip.tensor(qutip.Qobj(np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])), qutip.identity(3))

    # Truncate using custom function
    truncated_qubit_operator = truncate_custom(qubit_operator, products_to_keep, product_to_dressed)

    # Pad back using custom function
    padded_back_qubit_operator = pad_back_custom(truncated_qubit_operator, products_to_keep, product_to_dressed)

    # Expected truncated matrix based on products_to_keep
    truncated_expected = np.array([
        [0, 0, 0, 1, 0 ,2],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [3, 0, 0, 4, 0, 5],
        [0, 0, 3, 0, 4, 0],
        [6, 0, 0, 7, 0, 8]
    ], dtype=complex)

    assert np.allclose(truncated_qubit_operator.full(), truncated_expected), f"Truncated does not match: \n{truncated_qubit_operator.full()}"

    padded_expected = np.array([
        [0, 0, 0, 1, 0, 0 ,2, 0,0],
        [0, 0, 0, 0, 0, 0, 0, 0,0],
        [0, 0, 0, 0, 0, 1, 0, 0,0],
        [3, 0, 0, 4, 0, 0, 5, 0,0],
        [0, 0, 0, 0, 0, 0, 0, 0,0],
        [0, 0, 3, 0, 0, 4, 0, 0,0],
        [6, 0, 0, 7, 0, 0, 8, 0,0],
        [0, 0, 0, 0, 0, 0, 0, 0,0],
        [0, 0, 0, 0, 0, 0, 0, 0,0],
    ], dtype=complex)

    assert np.allclose(padded_back_qubit_operator.full(), padded_expected), f"Padded does not match:\n{padded_expected == padded_back_qubit_operator.full()}"

def generate_single_mapping(H_with_interaction_no_drive) -> np.ndarray:
    """
    The input should be in product basis
    Maps product of bare states to dressed state
    Returns a dictionary like {(0,0,0):0,(0,0,1):1}
    Use this function instead of scqubit's because I can change the overlap threshold here
    """
    evals, evecs = H_with_interaction_no_drive.eigenstates()
    overlap_matrix = scqubits.utils.spectrum_utils.convert_evecs_to_ndarray(evecs)
    OVERLAP_THRESHOLD = 0.1
    product_state_names = []
    dims = H_with_interaction_no_drive.dims[0]
    system_size = len(dims)
    def generate_product_states(current_state, ele_index):
        if ele_index == system_size:
            product_state_names.append(tuple(current_state))
            return
        
        for l in range(dims[ele_index]):
            current_state[ele_index] = l
            generate_product_states(current_state.copy(), ele_index + 1)

    current_state = [0] * system_size
    generate_product_states(current_state, 0)

    total_dim = math.prod(dims)
    dressed_indices_of_product_states = [None] * total_dim

    # for every energy eigenstate, from the lowerst to the highest, find the product state
    for dressed_index in range(len(evals)):
        max_position = (np.abs(overlap_matrix[dressed_index, :])).argmax()
        max_overlap = np.abs(overlap_matrix[dressed_index, max_position])
        overlap_matrix[:, max_position] = 0
        dressed_indices_of_product_states[int(max_position)] = dressed_index
        if (max_overlap**2 < OVERLAP_THRESHOLD):
            print(f'max overlap^2 {max_overlap**2} below threshold for dressed state {dressed_index} with eval {evals[dressed_index]}')
    product_to_dressed = {}
    for product, dressed in zip(product_state_names,dressed_indices_of_product_states):
        product_to_dressed[product] = dressed
    return product_to_dressed


def transition_frequency(hilbertspace,s0: int, s1: int) -> float:
    return (hilbertspace.energy_by_dressed_index(s1)- hilbertspace.energy_by_dressed_index(s0))

def find_dominant_frequency(expectation,tlist,dominant_frequency_already_found = None,plot = False,plot_freq = False):
    # In case alpha oscillates not at drive frequency, we do fourier transform to make the plot of coherent state look better 

    if dominant_frequency_already_found != None:
        expectation = expectation * np.exp(-1j*2*np.pi*dominant_frequency_already_found*tlist)

    expectation_fft = np.fft.fft(expectation)
    frequencies = np.fft.fftfreq(len(tlist), d=(tlist[1] - tlist[0]))  # assuming tlist is uniformly spaced

    # Identify the dominant frequency: 
    # (we exclude the zero frequency, which usually has the DC offset)
    dominant_freq_idx = np.argmax(np.abs(expectation_fft[1:])) + 1
    dominant_freq = frequencies[dominant_freq_idx]

    if plot:
        # Print the dominant frequency
        print(f"The dominant oscillation frequency is: {dominant_freq:.3f} (in the same units as 1/timestep)")

        fft_shifted = np.fft.fftshift(expectation_fft)
        frequencies_shifted = np.fft.fftshift(frequencies)
        plt.plot(frequencies_shifted, np.abs(fft_shifted))
        plt.xlabel('Frequency (arbitrary units)')
        plt.ylabel('Magnitude')
        plt.title('FFT of the Expectation Value')
        plt.grid(True)
        plt.show()
    elif plot_freq:
        plt(expectation_fft)
        plt.show()
    else:
        return dominant_freq



def dressed_to_2_level_dm(dressed_dm,product_to_dressed, qubit_level, osc_level,computational_0,computational_1,products_to_keep=None):
    dressed_dm_data = pad_back_custom(dressed_dm, products_to_keep, product_to_dressed)
    if dressed_dm_data.shape[1] == 1:
        dressed_dm_data = qutip.ket2dm(dressed_dm_data)
    dressed_dm_data = dressed_dm_data.full()
    rho_product = np.zeros((qubit_level * osc_level, qubit_level * osc_level), dtype=complex)
    for (ql, ol), dressed_level in product_to_dressed.items():
        index1 = ql * osc_level + ol
        # Loop again to populate the density matrix
        for (ql2, ol2), dressed_level2 in product_to_dressed.items():
            index2 = ql2 * osc_level + ol2
            # TODO  the order of product_state and product_state2 doesn't make sense to me, but it produces the right result. :(
            element = dressed_dm_data[dressed_level, dressed_level2]
            rho_product[index1, index2] += element
    rho_product = qutip.Qobj(rho_product, dims=[[qubit_level, osc_level], [qubit_level, osc_level]])
    qubit_rho = rho_product.ptrace(0)

    rho_2_level = qutip.Qobj(np.array([
            [qubit_rho[computational_0, computational_0],qubit_rho[computational_0, computational_1]],
            [qubit_rho[computational_1, computational_0],qubit_rho[computational_1, computational_1]]
        ]),dims=[[2],[2]])

    return rho_2_level
    
def compute_and_store_2_level_dm(args):
    results,file_name, i, j, product_to_dressed, qubit_level, osc_level,computational_0, computational_1,products_to_keep  = args
    
    rho_2_level = dressed_to_2_level_dm(results[i].states[j], product_to_dressed, qubit_level, osc_level, computational_0, computational_1,products_to_keep)
    
    with open(file_name, 'wb') as f:
        pickle.dump(rho_2_level, f)


def qobj_to_Array(matrix):
    if type(matrix) == qutip.qobj.Qobj:
        return Array(matrix.full()) 
    else:
        return Array(matrix) # Should be able to directly convert to Array
    


############################################################################
#
#
# Functions about visualizing data or results
#
#
############################################################################




def compute_expectation(ket_or_dm, operator):
    # Check if the input is a ket or a density matrix
    if ket_or_dm.shape[-1] == 1:  # Input is a ket
        return (jnp.linalg.multi_dot([jnp.conj(ket_or_dm).T, operator, ket_or_dm]))[0][0]
    else:  # Input is a density matrix
        return jnp.trace(jnp.dot(operator, ket_or_dm))
        
def get_vectorized_compute_expectation_function():
    # Vectorize the function over the kets
    vmapped_function = vmap(compute_expectation, in_axes=(0, None))
    return  jit(vmapped_function)

def plot_population(results,qubit_level,osc_level,product_to_dressed,a,w_d,tlist,fourier=False,fix_ylim = True,plot_only_pn_alpha = False):
    product_states = [(ql,ol) for ql in range(qubit_level) for ol in range(osc_level)]
    idxs = [product_to_dressed[(s1, s2)] for (s1, s2) in product_states]
    tot_dims = qubit_level*osc_level

    nlevels = len(results)


    a_op = jnp.array(a.full())
    pn_op = jnp.array((a.dag()*a).full())

    # Vectorize the function compute_expectation over the kets
    vectorized_compute_expectation = vmap(compute_expectation, in_axes=(0, None))
    vectorized_compute_expectation = jit(vectorized_compute_expectation)

    for i in range(nlevels):
        if hasattr(results[i], 'y'):
            states = jnp.array(results[i].y)  # assuming y contains JAX arrays or density matrices
        elif hasattr(results[i], 'states'):
            states = jnp.stack([jnp.array(q.full()) for q in results[i].states])  # assuming states contains QObj or density matrices

        results[i].expect = []
        if not plot_only_pn_alpha:
            for idx in idxs:
                dressed_state = jnp.zeros(tot_dims).at[idx].set(1).reshape(-1, 1)
                dressed_state_op = jnp.outer(dressed_state, jnp.conj(dressed_state).T)
                expectations = vectorized_compute_expectation(states, dressed_state_op)
                results[i].expect.append(expectations)
        alpha_expect = vectorized_compute_expectation(states, a_op)
        pns_expect = vectorized_compute_expectation(states, pn_op)
        results[i].expect.append(alpha_expect)
        results[i].expect.append(pns_expect)

    if fourier == True:
        first_dominant_freq =find_dominant_frequency(results[0].expect[-2],tlist)
    else:
        first_dominant_freq = w_d


    fig, axes = plt.subplots(4,nlevels, figsize=(9, 6))

    for i in range(nlevels):
        if not plot_only_pn_alpha:
            qubit_state_population = [np.zeros(shape=len(tlist))]*qubit_level
            for idx, product_state in enumerate(product_states):
                ql = product_state[0]
                qubit_state_population[ql] += results[i].expect[idx]
            for ql in range(nlevels):
                axes[0][i].plot(tlist, qubit_state_population[ql], label=r"$\overline{|%s\rangle}$" % (f"{ql}"))
        

        #*np.exp(-1j * 2 * np.pi * first_dominant_freq * tlist) # *np.exp(-1j * 2 * np.pi * dominant_freq * tlist)  

        alpha = results[i].expect[-2]*np.exp(-1j * 2 * np.pi * first_dominant_freq * tlist)

        # Coherent state eigenval
        real = alpha.real
        imag = alpha.imag
        axes[1][i].plot(tlist,imag , label=r"imag alpha")
        axes[2][i].plot(tlist, real, label=r"real alpha")
        axes[3][i].plot(-imag, real, label=r"imag alpha VS real alpha")
        
        # Photon number
        axes[0][i].plot(tlist, results[i].expect[-1], label=r"photon number")

    if fix_ylim: 
        axes[0][nlevels-1].legend(loc='center', ncol=1, bbox_to_anchor=(1.5, 0.5))
        axes[1][nlevels-1].legend(loc='center', ncol=1, bbox_to_anchor=(1.3, 0.5))
        axes[2][nlevels-1].legend(loc='center', ncol=1, bbox_to_anchor=(1.3, 0.5))
        axes[3][nlevels-1].legend(loc='center', ncol=1, bbox_to_anchor=(1.4, 0.5))
        plt.ylabel("population")
        plt.xlabel("t (ns)")
        for row in [0,1,2,3]:
            max_x_range,min_x_range,max_y_range,min_y_range = 0,0,0,0
            for col in range(nlevels):
                ymin, ymax = axes[row][col].get_ylim()
                xmin, xmax = axes[row][col].get_xlim()
                if ymax > max_y_range:
                    max_y_range = ymax
                if ymin < min_y_range:
                    min_y_range = ymin
                if xmax > max_x_range:
                    max_x_range = xmax
                if xmin < min_x_range:
                    min_x_range = xmin
            for col in range(nlevels):
                axes[row][col].set_ylim(min_y_range, max_y_range)
                axes[row][col].set_xlim(min_x_range,max_x_range)
                # Set the third row y range equal x range
                if row == 3:
                    axes[row][col].set_ylim(min(min_x_range,min_y_range), max(max_x_range,max_y_range))
                    axes[row][col].set_xlim(min(min_x_range,min_y_range),max(max_x_range,max_y_range))
    # plt.yscale('log')
    for ax in axes.flat:
        ax.minorticks_on()
        ax.grid(True)
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    for col in axes[3]:
        col.set_aspect('equal', 'box')
    plt.show()


def plot_specturum(qubit, resonator, hilbertspace, num_levels = 20,
                    flagged_transitions = [[[0,0],[0,1]],[[1,0],[1,1]],[[2,0],[2,1]]]):
    product_to_dressed = generate_single_mapping(hilbertspace.hamiltonian())
    energy_text_size = 8
    # clear_output(wait=True)
    
    fig, old_ax = qubit.plot_wavefunction(which = [0,1,2,3,4,5,6,7,8,9,10,11])
    left, bottom, width, height = 1, 0, 1, 1  
    ax = fig.add_axes([left, bottom, width, height])
    fig.set_size_inches(8, 8)

    product_to_dressed = bidict(product_to_dressed)
    qls = [product[0] for product in [product_to_dressed.inv[l] for l in range(num_levels)]]
    rls = [product[1] for product in [product_to_dressed.inv[l] for l in range(num_levels)]]
    max_qubit_level = max(qls) +1 
    max_resonator_level = max(rls) +1
    qubit_ori_energies = qubit.eigenvals(max_qubit_level)
    resonator_ori_energies = resonator.eigenvals(max_resonator_level)

    
    max_rl_for_ql = [0] *3
    for l in range(num_levels):
        (ql,rl) = product_to_dressed.inv[l]
        original = (qubit_ori_energies[ql] + resonator_ori_energies[rl])#* 2 * np.pi
        x1,x2 = ql-0.25,ql+0.25
        ax.plot([x1, x2], [original, original], linewidth=1, color='red')
        ax.text(ql, original, f"{original:.3f}", fontsize=energy_text_size, ha='center', va='bottom')

        dressed_state_index = product_to_dressed[(ql,rl)]
        dressed = hilbertspace.energy_by_dressed_index(dressed_state_index)#* 2 * np.pi
        ax.plot([x1, x2], [dressed, dressed], linewidth=1, color='green')
        ax.text(ql, dressed, f"{dressed:.3f}", fontsize=energy_text_size, ha='center', va='top')

        if ql in [0,1,2]:
            if rl > max_rl_for_ql[ql]:
                max_rl_for_ql[ql]=rl

    flagged_transitions = []
    for ql in [0,1,2]:
        for i in range(max_rl_for_ql[ql] ):
            flagged_transitions.append([[ql,i],[ql,i+1]])
    for transition in flagged_transitions:
        state1, state2 = transition[0],transition[1]
        dressed_index1 = product_to_dressed[(state1[0],state1[1])]
        dressed_index2 = product_to_dressed[(state2[0],state2[1])]
        if dressed_index1!= None and dressed_index2!= None:
            energy1 = hilbertspace.energy_by_dressed_index(dressed_index1)#* 2 * np.pi
            energy2 = hilbertspace.energy_by_dressed_index(dressed_index2)#* 2 * np.pi
            ax.plot([state1[0], max_qubit_level], [energy2, energy2], linewidth=1, color='green')
            ax.plot([state1[0], state2[0]], [energy1, energy2], linewidth=1, color='green')
            ax.text((state1[0]+ state2[0])/2, (energy1+ energy2)/2, f"{energy2-energy1:.3f}", fontsize=energy_text_size, ha='center', va='top')
        else:
            print("dressed_state_index contain None")
    plt.show()

def plot_heatmap(result, time_index, product_to_dressed, qubit_levels, oscillator_levels,norm ):
    if hasattr(result, 'states'):
        dm = result.states[time_index]
    elif hasattr(result, 'y'):
        dm = result.y[time_index]

    if dm.shape[1] == 1:
        dm = qutip.ket2dm(dm)
    
    dm = 0.5 * (dm + dm.dag())
    dm = dm / dm.tr()
    
    # dm = pad_back_function(dm)
    grid = np.zeros(( qubit_levels,oscillator_levels))

    for qubit_level in range(qubit_levels):
        for oscillator_level in range(oscillator_levels):
            product_state = (qubit_level, oscillator_level)
            dressed_state = product_to_dressed[product_state]
            if dressed_state < dm.dims[0][0]:
                # Create a basis state corresponding to the dressed state
                basis_state = qutip.basis(dm.dims[0][0], dressed_state)
                # Calculate the expectation value
                expectation_value = qutip.expect(basis_state * basis_state.dag(), dm)
            else:
                expectation_value = 0
            grid[ qubit_level,oscillator_level] = expectation_value
    grid[grid < 1e-11] = 1e-11
    plt.imshow(grid, cmap='viridis', origin='lower', norm=norm)
    plt.colorbar(label='Expectation Value')
    plt.xlabel('Oscillator Level')
    plt.ylabel('Qubit Level')
    plt.title(f'Expectation Values at t = {result.times[time_index]}')
    plt.show()

def interactive_heatmap(result, product_to_dressed, qubit_levels, oscillator_levels,norm = LogNorm()):
    if hasattr(result, 'times'):
        times = result.times
    elif hasattr(result, 't'):
        times = result.t
    time_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(times) - 1,
        step=1,
        description='Time Index:',
        continuous_update=False
    )
    
    widgets.interact(lambda time_index: plot_heatmap(result, time_index, product_to_dressed, qubit_levels, oscillator_levels,norm),
                     time_index=time_slider)
    




def get_shift_accurate(ele,omega_i, omega_j, omega_r):
    return abs(ele)**2 / (omega_j-omega_i-omega_r) - abs(ele)**2 / (omega_i-omega_j-omega_r)



def get_EJ_Er_sweep_data(EJ_values, 
         Er_values,
         EC,
         EL,
         computational_state = [0,1],
         leakage_state = 2,
    ):

    qubit_level = 25
    
    Z1 = np.zeros_like(np.meshgrid(EJ_values, Er_values)[0])
    Z2 = np.zeros_like(np.meshgrid(EJ_values, Er_values)[0])

    highest_level_to_transition_from = 16
    transitions_to_0 = [[] for _ in range(highest_level_to_transition_from)]
    transitions_to_1 = [[] for _ in range(highest_level_to_transition_from)]
    transitions_to_2 = [[] for _ in range(highest_level_to_transition_from)]

    # for every EJ
    for i in tqdm(range(len(EJ_values)), desc="sweeping"):
        qbt = scqubits.Fluxonium(EJ=EJ_values[i],EC=EC,EL=EL,flux=0,cutoff=110,truncated_dim=qubit_level)
        num_evals = qubit_level
        evals = qbt.eigenvals(num_evals)
        elements = qbt.matrixelement_table('n_operator',evals_count = num_evals)
        
        # record the transitions to plot reference curves
        for l in range(highest_level_to_transition_from):
            transitions_to_0[l].append(abs(evals[l]-evals[0]))
            transitions_to_1[l].append(abs(evals[l]-evals[1]))
            transitions_to_2[l].append(abs(evals[l]-evals[2]))

        # get estimated dispersive shifts
        for j in range(len(Er_values)):
            Er = Er_values[j]
            shifts = [
                sum([get_shift_accurate(elements[0,ql2], evals[0], evals[ql2], Er) for ql2 in range(num_evals)] ),
                sum([get_shift_accurate(elements[1,ql2], evals[1], evals[ql2], Er) for ql2 in range(num_evals)]),
                sum([get_shift_accurate(elements[2,ql2], evals[2], evals[ql2], Er) for ql2 in range(num_evals)] )
            ]

            Z1[j, i] = abs(shifts[computational_state[1]]-shifts[computational_state[0]])
            Z2[j, i] = abs(shifts[leakage_state]-shifts[computational_state[1]])
    return (transitions_to_0,
            transitions_to_1,
            transitions_to_2,
            Z1,
            Z2)


def get_EJ_Er_sweep_data_diagonalization(EJ_values, 
         Er_values,
         EC,
         EL,
         g,
         computational_state = [0,1],
         leakage_state = 2,
    ):

    qubit_level = 20
    
    Z1 = np.zeros_like(np.meshgrid(EJ_values, Er_values)[0])
    Z2 = np.zeros_like(np.meshgrid(EJ_values, Er_values)[0])

    highest_level_to_transition_from = 16
    transitions_to_0 = [[] for _ in range(highest_level_to_transition_from)]
    transitions_to_1 = [[] for _ in range(highest_level_to_transition_from)]
    transitions_to_2 = [[] for _ in range(highest_level_to_transition_from)]

    # for every EJ
    for i in tqdm(range(len(EJ_values)), desc="sweeping"):
        qbt = scqubits.Fluxonium(EJ=EJ_values[i],EC=EC,EL=EL,flux=0,cutoff=110,truncated_dim=qubit_level)
        num_evals = qubit_level
        evals = qbt.eigenvals(num_evals)
        elements = qbt.matrixelement_table('n_operator',evals_count = num_evals)
        
        # record the transitions to plot reference curves
        for l in range(highest_level_to_transition_from):
            transitions_to_0[l].append(abs(evals[l]-evals[0]))
            transitions_to_1[l].append(abs(evals[l]-evals[1]))
            transitions_to_2[l].append(abs(evals[l]-evals[2]))

        # get estimated dispersive shifts
        for j in range(len(Er_values)):
            Er = Er_values[j]
            # shifts = [
            #     sum([get_shift_accurate(elements[0,ql2], evals[0], evals[ql2], Er) for ql2 in range(num_evals)] ),
            #     sum([get_shift_accurate(elements[1,ql2], evals[1], evals[ql2], Er) for ql2 in range(num_evals)]),
            #     sum([get_shift_accurate(elements[2,ql2], evals[2], evals[ql2], Er) for ql2 in range(num_evals)] )
            # ]
            try:
                osc = scqubits.Oscillator(
                E_osc=Er,
                truncated_dim=6
                )
                hilbertspace = scqubits.HilbertSpace([qbt, osc])
                hilbertspace.add_interaction(
                    g_strength=g,
                    op1=qbt.n_operator,
                    op2=osc.creation_operator,
                    add_hc=True
                )
                hilbertspace.generate_lookup()
                chi0 = transition_frequency(hilbertspace,hilbertspace.dressed_index((0,0)),hilbertspace.dressed_index((0,1))) - Er
                chi1 = transition_frequency(hilbertspace,hilbertspace.dressed_index((1,0)),hilbertspace.dressed_index((1,1))) - Er
                chi2 = transition_frequency(hilbertspace,hilbertspace.dressed_index((2,0)),hilbertspace.dressed_index((2,1))) - Er
                shifts = [
                    chi0,
                    chi1,
                    chi2
                ]

                Z2[j, i] = abs(shifts[leakage_state]-shifts[computational_state[1]])
                Z1[j, i] = abs(shifts[computational_state[1]]-shifts[computational_state[0]])
            except:
                Z2[j, i] = np.nan   
                Z1[j, i] = np.nan
    return (transitions_to_0,
            transitions_to_1,
            transitions_to_2,
            Z1,
            Z2)




def plot_EJ_Er_sweep(
        EJ_values, 
        Er_values,
        transitions_to_0,
        transitions_to_1,
        transitions_to_2,
        Z1,
        Z2,
        computational_state = [0,1],
         leakage_state = 2,
        legend = False,    
        norm1= LogNorm(vmin=1e-5,vmax=1e-4),
        norm2 = LogNorm(vmin=1e-2,vmax=1),
        big_pic = False,
):
    X, Y = np.meshgrid(EJ_values, Er_values)
    # Plotting
    if not big_pic:
        fig = plt.figure(figsize=(2*(3+3/8), 
                            (3+3/8)/1.8))
    else:
        fig = plt.figure(figsize=(2*(3+3/8)*3, 
                            (3+3/8)/1.8*3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.2, 1], wspace=0.4)

    # ax1 = plt.subplot(gs[0])
    # plt.text(-0.25, 1, '(a)', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', color='black')
    # plt.plot(EJ_values, transition_21, '-', linewidth=2,color = 'black')
    # plt.xlabel('EJ')
    # plt.ylabel(r'$\omega_{12}$')
    
    transitions = [transitions_to_0, transitions_to_1, transitions_to_2]
    line_styles = [
        (0,(1, 1)),
        (0,(3, 3)),
        (0,(3,2,1,2)),
        (0, (1,1,1,1,3,3)),
        (0, (1,1,1,1,1,1,3,3)),
        (0, (1,1,2,2,1,1,4,5)),
        (0, (5,5)),
        (0, (5,1,5,1))
    ]
    line_style_counter = [[],[],[]]
    next_line_style_idx = 0
    def plot_transition_curves(initial_level):
        nonlocal next_line_style_idx
        if line_style_counter[initial_level] == []:
            for level, list_of_transitions in enumerate(transitions[initial_level]):
                if level % 2 == 1 - (initial_level%2) and np.max(list_of_transitions) > Er_values[0] and np.min(list_of_transitions) < Er_values[-1]:
                    plt.plot(EJ_values, list_of_transitions, linestyle = line_styles[next_line_style_idx], linewidth=2,color = 'black',label = rf'$\omega_{{{initial_level},{level}}}$')
                    line_style_counter[initial_level].append(next_line_style_idx)
                    next_line_style_idx += 1
                    next_line_style_idx = next_line_style_idx % len(line_styles)
        else:
            local_counter = 0
            for level, list_of_transitions in enumerate(transitions[initial_level]):
                if level % 2 == 1 - (initial_level%2) and np.max(list_of_transitions) > Er_values[0] and np.min(list_of_transitions) < Er_values[-1]:
                    try:
                        plt.plot(EJ_values, list_of_transitions, linestyle = line_styles[line_style_counter[initial_level][local_counter]], linewidth=2,color = 'black',label = rf'$\omega_{{{initial_level},{level}}}$')
                    except:
                        print(line_styles[line_style_counter[initial_level][local_counter]])
                    local_counter += 1

    ################################################################################
    # Heatmap about the diff of shift from the two computational states
    ################################################################################
    ax0 = plt.subplot(gs[0])
    plt.text(0.05, 1, '(a)', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', color='black')
    z1_plot = plt.pcolormesh(X, Y, Z1, shading='auto', cmap='inferno', norm=norm1)
    plt.colorbar(z1_plot)


    plot_transition_curves(computational_state[0])
    plot_transition_curves(computational_state[1])


    if legend:
        plt.legend(loc='lower left')
    plt.xlabel('EJ')
    plt.ylabel('Er')
    ax0.set_xlim([EJ_values[0], EJ_values[-1]])
    ax0.set_ylim([Er_values[0], Er_values[-1]])
    ax0.grid(which='major', color='grey', linestyle='-', linewidth=0.5)
    ax0.minorticks_on()
    ax0.grid(which='minor', color='grey', linestyle=':', linewidth=0.5)



    ################################################################################
    # Heatmap about the diff of shift from the first computational state and the leakage state
    ################################################################################
    ax1 = plt.subplot(gs[1])
    plt.text(0.05, 1, '(b)', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', color='black')
    z2_plot = plt.pcolormesh(X, Y, Z2, shading='auto', cmap='inferno', norm=norm2)
    plt.colorbar(z2_plot)

    plot_transition_curves(computational_state[0])
    plot_transition_curves(leakage_state)

    if legend:
        plt.legend(loc='lower left')
    plt.xlabel('EJ')
    # plt.ylabel('Er')
    ax1.set_xlim([EJ_values[0], EJ_values[-1]])
    ax1.set_ylim([Er_values[0], Er_values[-1]])
    ax1.grid(which='major', color='grey', linestyle='-', linewidth=0.5)
    ax1.minorticks_on()
    ax1.grid(which='minor', color='grey', linestyle=':', linewidth=0.5)
    # ax3.set_xticks([]) 
    # ax3.set_yticks([])


    ################################################################################
    # Additional subplot for overlay
    ################################################################################
    ax2 = plt.subplot(gs[2])
    plt.text(0.05, 1, '(c)', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', color='black')
    # Overlay of Z1 and Z2
    plt.pcolormesh(X, Y, Z1, shading='auto', cmap='inferno', alpha=0.5, norm=norm1)
    plt.pcolormesh(X, Y, Z2, shading='auto', cmap='inferno', alpha=0.5, norm=norm2)

    plot_transition_curves(computational_state[0])
    plot_transition_curves(computational_state[1])
    plot_transition_curves(leakage_state)

    plt.xlabel('EJ')
    # plt.ylabel('Er')
    ax2.set_xlim([EJ_values[0], EJ_values[-1]])
    ax2.set_ylim([Er_values[0], Er_values[-1]])
    ax2.grid(which='major', color='grey', linestyle='-', linewidth=0.5)
    ax2.minorticks_on()
    ax2.grid(which='minor', color='grey', linestyle=':', linewidth=0.5)

    plt.tight_layout()

    return fig, (ax0, ax1, ax2)

    