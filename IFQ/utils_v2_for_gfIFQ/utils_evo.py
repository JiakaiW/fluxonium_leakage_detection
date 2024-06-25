import concurrent
from datetime import datetime

from loky import get_reusable_executor
import numpy as np
import pickle
import qutip
from typing import List, Any, Union, Tuple, Callable, Dict
from tqdm import tqdm
from utils_basic_funcs import *
from utils_drive import *

def post_process(
            result:qutip.solver.Result,
            post_processing_funcs:List=[],
            post_processing_args:List=[],
            ):
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

def ODEsolve_and_post_process(
            y0: qutip.Qobj,
            tlist: np.array, 

            static_hamiltonian: qutip.Qobj,
            drive_terms: List[DriveTerm],
            c_ops: Union[None,List[qutip.Qobj]] = None,
            e_ops:Union[None,List[qutip.Qobj]] = None,

            store_states = True,
            method:str = 'qutip.mesolve',
            post_processing_funcs:List=[],
            post_processing_args:List=[],

            file_name: str = None
            ):
    '''
    This method is only used for qutip's cpu solvers. For dynamiqs solver call CoupledSystem.run_dq_mesolve_parrallel

    It should take in:
        a static hamiltonian, 
        a list of "drive terms", 
            then assemble the two into an H_with_drive
        a list of c_ops
    '''
    
    H_with_drives =  [static_hamiltonian] + \
          [[drive_term.driven_op, drive_term.pulse_shape_func] for drive_term in drive_terms]
    
    additional_args = {}
    for drive_term in drive_terms:
        for key in drive_term.pulse_shape_args:
            if key in additional_args:
                raise ValueError(f"Duplicate key found: {key}")
            else:
                additional_args[key] = drive_term.pulse_shape_args[key]

    if method == 'qutip.mesolve':
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

    elif method == 'qutip.mcsolve':
        print("called qutip.mcsolve")
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
    else:
        raise Exception("solver method not supported")


    result = post_process(result,
                                 post_processing_funcs,
                                post_processing_args)
    if file_name!= None:
        current_datetime = datetime.now()
        datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        with open(f'{file_name} {datetime_string}.pkl', 'wb') as file:
            pickle.dump(result, file)
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
    with get_reusable_executor(max_workers=max_workers, context='loky') as executor:
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
                                            list_of_systems[i].filtered_product_to_dressed,
                                            list_of_systems[i].sign_multiplier,
                                            None
                                            ))
            future = executor.submit(
                ODEsolve_and_post_process, 
                y0=list_of_kwargs[i]['y0'], 
                tlist=list_of_kwargs[i]['tlist'], 

                static_hamiltonian=list_of_systems[i].diag_dressed_hamiltonian,
                drive_terms=list_of_kwargs[i].get('drive_terms', None),
                c_ops=list_of_kwargs[i].get('c_ops', None),
                e_ops=list_of_kwargs[i].get('e_ops', None),
                store_states = store_states,
                post_processing_funcs=post_processing_funcs,
                post_processing_args=post_processing_args,
                file_name = f'{i}')
            futures[future] = i
        
        for future in concurrent.futures.as_completed(futures):
            original_index = futures[future]
            results[original_index] = future.result()
    return results

def run_dq_ODEsolve_and_post_process_jobs_with_different_systems_but_same_y0(
        list_of_systems: List[CoupledSystem],
        list_of_kwargs: list[Any],
        max_workers = None,
        store_states = True,
        post_processing = ['pad_back'],
    ):
    '''
    This function runs dynamiqs's mesolve or sesolve using the concurrency of dynamiqs, 
       and then use cpu parrallization to do post processing.
    Function signature is intentionally kept the same as run_parallel_ODEsolve_and_post_process_jobs_with_different_systems

    Because we utilize dynamiqs' concurrency, we will run every hamiltonian with every initial states when calling dq.sesolve. 
    And most importantly we will apply the jump_ops to every simulation! We don't have the same jump_ops with different systems.
    '''
    pass

