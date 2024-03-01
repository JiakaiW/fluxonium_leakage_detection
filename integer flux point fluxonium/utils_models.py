
import concurrent
from dataclasses import dataclass
from itertools import product
import numpy as np
import pickle
import qutip
import scqubits
from typing import List, Any, Union, Tuple, Callable, Dict
from tqdm.notebook import tqdm

# from utils_qiskit import *
from utils_visualization import *
from utils_basic_funcs import *


############################################################################
#
# Classes about modelling the system and running ODE solvers
#   the code is centered around qutip, for functions that use qiskit-dynamics, 
#   I convert objects to jnp locally
#
############################################################################


@dataclass
class DriveTerm:
    driven_op: qutip.Qobj
    pulse_shape_func: Callable
    pulse_shape_args: Dict


class CoupledSystem:
    '''
    A parent class for quantum systems involving qubits and oscillators,
    
    This class is meant to be very generic, any specific setup can inherit from this 
        class and define commonly used attributes in the child class and be as customized as wanted
    '''
    def __init__(self,
                 hilbertspace,
                 products_to_keep,
                 qbt_position,
                 computaional_states):
        self.qbt_position = qbt_position
        self.computaional_states = computaional_states
        self.hilbertspace = hilbertspace
        self.hilbertspace.generate_lookup()
        self.product_to_dressed = generate_single_mapping(self.hilbertspace.hamiltonian())
        if products_to_keep == None or products_to_keep == []:
            products_to_keep =list(product(*[range(dim) for dim in self.hilbertspace.subsystem_dims])) 
        self.products_to_keep = products_to_keep

        self.evals = self.hilbertspace["evals"][0]
        self.diag_dressed_hamiltonian = self.truncate_function(qutip.Qobj((
                2 * np.pi * qutip.Qobj(np.diag(self.evals),
                dims=[self.hilbertspace.subsystem_dims] * 2)
        )[:, :]))

    def truncate_function(self,qobj):
        return truncate_custom(qobj, self.products_to_keep, self.product_to_dressed)
    
    def pad_back_function(self,qobj):
        return pad_back_custom(qobj, self.products_to_keep, self.product_to_dressed)
        
    def run_mesolve_parrallel(self,
                    initial_states: qutip.Qobj, # truncated initial states
                    tlist: np.array, 
                    drive_terms: List[DriveTerm],
                    c_ops: Union[None,List[qutip.Qobj]] = None,
                    e_ops:Union[None,List[qutip.Qobj]] = None,

                    post_processing = ['pad_back'],
                    ):
        '''
        This function runs mesolve on multiple initial states using multi-processing,
          and return a list of qutip.solver.result
        '''

        post_processing_funcs = []
        post_processing_args = []
        if 'pad_back' in post_processing:
            post_processing_funcs.append(pad_back_custom)
            post_processing_args.append((self.products_to_keep, 
                                self.product_to_dressed))
        if 'partial_trace_computational_states' in post_processing:
            post_processing_funcs.append(dressed_to_2_level_dm)
            post_processing_args.append((
                                        self.product_to_dressed,
                                        self.qbt_position, 
                                        self.computaional_states[0],
                                        self.computaional_states[1],
                                        None))
            

        results = [None] * len(initial_states)
        with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
            futures = {executor.submit(ODEsolve_and_post_process, 
                                        y0=initial_states[i], 
                                        tlist=tlist,    

                                        static_hamiltonian = [self.diag_dressed_hamiltonian],
                                        drive_terms=drive_terms,
                                        c_ops=c_ops,
                                        e_ops = e_ops,
                                    
                                        method = 'mesolve',
                                        post_processing_funcs=post_processing_funcs,
                                        post_processing_args=post_processing_args,
                                        ): i for i in range(len(initial_states))}
            
            for future in concurrent.futures.as_completed(futures):
                original_index = futures[future]
                results[original_index] = future.result()

        return results
    



class FluxoniumOscillatorSystem(CoupledSystem):
    '''
    To model leakage detection of 12 fluxonium
    '''
    def __init__(self,
                computaional_states:str, # = '0,1' or '1,2'
                drive_transition: Tuple[int] = None,

                EJ:float = 2.33,
                EC:float = 0.69,
                EL:float = 0.12,
                qubit_level:float = 13,
                
                
                Er:float = 7.16518677,
                osc_level:float = 30,
                kappa = 0.001,

                g_strength:float = 0.18,

                products_to_keep: List[List[int]]= None,
                w_d:float = None
                ):
        '''
        Initialize objects before truncation
        '''
        
        self.qbt = scqubits.Fluxonium(EJ=EJ,EC=EC,EL=EL,flux=0,cutoff=110,truncated_dim=qubit_level)
        self.osc = scqubits.Oscillator(E_osc=Er,truncated_dim=osc_level,l_osc=1.0)
        hilbertspace = scqubits.HilbertSpace([self.qbt, self.osc])
        
        # hilbertspace.add_interaction(g_strength=g_strength,op1=self.qbt.n_operator, op2=self.osc.annihilation_operator,add_hc=True) 
        '''
        This interaction produces the same dispersive shift as the n_operator, 
         but it tends to mess up the phases of the qubit coupled to different levels of resonator
         it also shortens the amount of time the photon population returns to zeroby half
        '''

        hilbertspace.add_interaction(g_strength=g_strength,op1=self.qbt.n_operator, op2=self.osc.n_operator,add_hc=False) # Edited
        '''
        This interaction produces the same dispersive shift as the first one, 
         and the phases of the qubit coupled to different levels of resonator is always about the same or
         different by pi. (whether drive by a+a.dag or n operator has nothing to do with messing up phase)
        '''

        # if products_to_keep is None:
        #     products_to_keep = [[ql, ol] for ql in [1,2,3] for ol in range(10) ] \
        #                         + [[ql, ol] for ql in [0] for ol in range(30) ]

        super().__init__(hilbertspace = hilbertspace,
                         products_to_keep = products_to_keep,
                         qbt_position = 0,
                        computaional_states = [int(computaional_states[0]),int(computaional_states[-1])])

        self.a = qutip.Qobj(self.hilbertspace.op_in_dressed_eigenbasis(self.osc.annihilation_operator)[:, :])
        self.a_trunc = self.truncate_function(self.a)
        self.driven_operator =   -1j*(self.a_trunc - self.a_trunc.dag() ) # TODO:
        self.c_ops = [np.sqrt(kappa) * self.a_trunc] 

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
        
        
        

class FluxoniumOscillatorFilterSystem(CoupledSystem):
    '''
    To model leakage detection of 12 fluxonium with purcell filter
    '''
    def __init__(self,
                computaional_states:str, # = '0,1' or '1,2'

                EJ:float = 2.33,
                EC:float = 0.69,
                EL:float = 0.12,
                qubit_level:float = 13,
                
                
                Er:float = 7.16518677,
                osc_level:float = 20,
                
                Ef:float = 7.13,
                filter_level:float = 7,
                kappa_f = 1.5, # Ef *2pi = omega_f,  kappa_f = omega_f / Q , kappa_f^{-1} = 0.67 ns

                g_strength:float = 0.18,
                G_strength:float = 0.3, # G satisfies a relation with omega_r in equation 10 of Phys. Rev A 92. 012325 (2015)

                products_to_keep: List[List[int]]= None,
                w_d:float = None,
                ):


        # Q_f = 30
        # kappa_f = Ef * 2 * np.pi / Q_f
        # kappa_r = 0.0001 #we want a really small effective readout resonator decay rate to reduce purcell decay
        # G_strength =np.sqrt(kappa_f * kappa_r * ( 1 + (2*(Er-Ef)*2*np.pi/kappa_f )**2 ) /4)

        self.G_strength = G_strength

        self.qbt = scqubits.Fluxonium(EJ=EJ,EC=EC,EL=EL,flux=0,cutoff=110,truncated_dim=qubit_level)
        self.osc = scqubits.Oscillator(E_osc=Er,truncated_dim=osc_level)
        self.filter = scqubits.Oscillator(E_osc=Ef,truncated_dim=filter_level)
        hilbertspace = scqubits.HilbertSpace([self.qbt, self.osc,self.filter])
        hilbertspace.add_interaction(g_strength=g_strength,op1=self.qbt.n_operator,op2=self.osc.creation_operator,add_hc=True)
        hilbertspace.add_interaction(g_strength=G_strength,op1=self.osc.creation_operator,op2=self.filter.annihilation_operator,add_hc=True)

        super().__init__(hilbertspace = hilbertspace,
                         products_to_keep = products_to_keep,
                         qbt_position = 0,
                        computaional_states = [int(computaional_states[0]),int(computaional_states[-1])])

        self.a = qutip.Qobj(self.hilbertspace.op_in_dressed_eigenbasis(self.osc.annihilation_operator)[:, :])
        self.a_trunc = self.truncate_function(self.a)

        self.b = qutip.Qobj(self.hilbertspace.op_in_dressed_eigenbasis(self.filter.annihilation_operator)[:, :])
        self.b_trunc = self.truncate_function(self.b)
        self.driven_operator =   self.b_trunc+self.b_trunc.dag()
        self.c_ops = [np.sqrt(kappa_f) * self.b_trunc]
        
        if w_d!= None:
            self.w_d = w_d

     


def ODEsolve_and_post_process(
            y0: qutip.Qobj,
            tlist: np.array, 

            static_hamiltonian: qutip.Qobj,
            drive_terms: List[DriveTerm],
            c_ops: Union[None,List[qutip.Qobj]] = None,
            e_ops:Union[None,List[qutip.Qobj]] = None,

            store_states = True,
            method:str = 'mesolve',
            post_processing_funcs:List=[],
            post_processing_args:List=[],
            ):
    '''
    This method is now more modular:

    It should take in:
        a static hamiltonian, 
        a list of "drive terms", 
            then assemble the two into an H_with_drive
        a list of c_ops
    '''
    static_hamiltonian = [static_hamiltonian] if type(static_hamiltonian) != list else static_hamiltonian
    
    H_with_drives =  static_hamiltonian + \
          [[drive_term.driven_op, drive_term.pulse_shape_func] for drive_term in drive_terms]
    
    additional_args = {}
    for drive_term in drive_terms:
        for key in drive_term.pulse_shape_args:
            if key in additional_args:
                raise ValueError(f"Duplicate key found: {key}")
            else:
                additional_args[key] = drive_term.pulse_shape_args[key]

    if method == 'mesolve':
        result = qutip.mesolve(
            rho0=y0,
            H=H_with_drives,
            tlist=tlist,
            c_ops=c_ops,
            e_ops = e_ops,
            args=additional_args,
            options=qutip.Options(store_states=store_states, nsteps=120000, num_cpus=1),
            progress_bar = qutip.ui.progressbar.EnhancedTextProgressBar(),
        )

    elif method == 'mcsolve':
        result = qutip.mcsolve(psi0=y0, 
                            H= H_with_drives,
                            tlist=tlist,
                            args = additional_args,
                            c_ops=c_ops,
                            e_ops = e_ops,
                            ntraj = 500,
                            options=qutip.Options(store_states=True,num_cpus = None),
                            progress_bar = qutip.ui.progressbar.EnhancedTextProgressBar(),
                            )

    # for func, args in zip(post_processing_funcs, post_processing_args):
    #     result.states = [func(state, *args) for state in tqdm(result.states, desc=f"Processing states with {func.__name__}")]

    # Editted post processing so that it doesn't overwrite result.states
    last_attribute_name = "states"
    for func, args in zip(post_processing_funcs, post_processing_args):
        new_attr_name = f"states_{func.__name__}" 
        processed_states = [func(state, *args) for state in tqdm( getattr(result, last_attribute_name), desc=f"Processing states with {func.__name__}")]
        setattr(result, new_attr_name, processed_states)
        last_attribute_name = new_attr_name
    return result



def run_parallel_ODEsolve_and_post_process_jobs_with_different_systems(
        list_of_systems: List[CoupledSystem],
        list_of_kwargs: list[Any],
        max_workers = None,
        store_states = True,
        post_processing = ['pad_back'],
    ):
    '''
    This function helps run mesolve using the ODEsolve_and_post_process function concurrently
    Args:
        list_of_systems: list of CoupledSystem
        list_of_kwargs: list of kwargs dictionaries used to call ODEsolve_and_post_process
            a single kwargs should be a dictionary like {'y0',
                                                        'tlist',

                                                        'drive_terms',
                                                        'c_ops',
                                                        'e_ops'
                                                        }
    '''
    assert len(list_of_systems) == len(list_of_kwargs)
    
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
                                            list_of_systems[i].qbt_position, 
                                            list_of_systems[i].computaional_states[0],
                                            list_of_systems[i].computaional_states[1],
                                            None))
            future = executor.submit(
                ODEsolve_and_post_process, 
                y0=list_of_kwargs[i]['y0'], 
                tlist=list_of_kwargs[i]['tlist'], 

                static_hamiltonian=[list_of_systems[i].diag_dressed_hamiltonian],
                drive_terms=list_of_kwargs[i].get('drive_terms', None),
                c_ops=list_of_kwargs[i].get('c_ops', None),
                e_ops=list_of_kwargs[i].get('e_ops', None),
                store_states = store_states,
                post_processing_funcs=post_processing_funcs,
                post_processing_args=post_processing_args)
            futures[future] = i
        
        for future in concurrent.futures.as_completed(futures):
            original_index = futures[future]
            results[original_index] = future.result()
    return results







############################################################################
#
#
# Ancilliary functions about pulse shaping and time dynamics
#
#
############################################################################

    
def square_pulse_with_rise_fall(t, args):
    w_d = args['w_d']
    amp = args['amp']
    t_start = args.get('t_start', 0)  # Default start time is 0
    t_rise = args.get('t_rise', 0)  # Default rise time is 0 for no rise
    t_square = args.get('t_square', 0)  # Duration of constant amplitude

    def cos_modulation():
        return 2 * np.pi * amp * np.cos(w_d * 2 * np.pi * t)
    
    t_fall_start = t_start + t_rise + t_square  # Start of fall
    t_end = t_fall_start + t_rise  # End of the pulse

    if t < t_start:  # Before pulse start
        return 0
    elif   t_rise > 0  and t_start <= t <= t_start + t_rise:  # Rise segment
        return np.sin(np.pi * (t - t_start) / (2 * t_rise))**2 * cos_modulation()
    elif t_start + t_rise < t <= t_fall_start:  # Constant amplitude segment
        return cos_modulation()
    elif  t_rise > 0  and  t_fall_start < t <= t_end:  # Fall segment
        return np.sin(np.pi * (t_end - t) / (2 * t_rise))**2 * cos_modulation()
    else:  # After pulse end
        return 0

    