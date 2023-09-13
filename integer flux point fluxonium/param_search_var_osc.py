import numpy as np
import sys
import pickle
import scqubits
import math
import gzip
# from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, Manager
import os

def generate_single_mapping(H_with_interaction_no_drive) -> np.ndarray:
    """
    Maps product of bare states to dressed state
    Returns a dictionary like {(0,0,0):0,(0,0,1):1}
    Use this function instead of scqubit's because I can change the overlap threshold here
    """
    evals, evecs = H_with_interaction_no_drive.eigenstates()
    overlap_matrix = scqubits.utils.spectrum_utils.convert_evecs_to_ndarray(evecs)
    # OVERLAP_THRESHOLD = 0.02
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
    dressed_indices = [None] * total_dim
    for dressed_index in range(len(evals)):
        max_position = (np.abs(overlap_matrix[dressed_index, :])).argmax()
        max_overlap = np.abs(overlap_matrix[dressed_index, max_position])
        overlap_matrix[:, max_position] = 0
        dressed_indices[int(max_position)] = dressed_index
        # if (max_overlap**2 < OVERLAP_THRESHOLD):
        #     print(f'max overlap^2 {max_overlap**2} below threshold for dressed state {dressed_index} with eval {evals[dressed_index]}')
    product_to_dressed = {}
    for product, dressed in zip(product_state_names,dressed_indices):
        product_to_dressed[product] = dressed
    return product_to_dressed



class search_job:
    def __init__(self, EJ, EC_values, EL_values,Er_values):
        self.EJ = EJ
        self.EC_values = EC_values
        self.EL_values = EL_values
        self.Er_values = Er_values
        self.results = None

    # def run(self):
    #     total_iterations = max(len(self.EC_values), len(self.EL_values), len(self.Er_values))
    #     current_iteration = 0

    #     def get_estimations_with_progress(EJ, EC, EL, Er):
    #         result = get_estimations(EJ, EC, EL, Er)
    #         nonlocal current_iteration
    #         current_iteration += 1
    #         print(f"Progress: {current_iteration}/{total_iterations} iterations completed.")
    #         return result

    #     self.results = np.vectorize(get_estimations_with_progress)(self.EJ, self.EC_values, self.EL_values, self.Er_values)
    def run(self):

        # # Initialize lists to collect results
        # list_one_two_transition = []
        # list_differential_stark = []
        # list_qubit_zero_lamb = []
        # list_detunning_qubit = []


        # args_list = [(self.EJ, EC, EL, Er) for EC, EL, Er in zip(self.EC_values, self.EL_values, self.Er_values)]

        # with ProcessPoolExecutor(max_workers=8) as executor:
        #     results = list(executor.map(wrapper, args_list))

        # for result in results:
        #     one_two_transition, differential_stark, qubit_zero_lamb, detunning_qubit = result
        #     list_one_two_transition.append(one_two_transition)
        #     list_differential_stark.append(differential_stark)
        #     list_qubit_zero_lamb.append(qubit_zero_lamb)
        #     list_detunning_qubit.append(detunning_qubit)

        # self.results = (
        #     np.array(list_one_two_transition),
        #     np.array(list_differential_stark),
        #     np.array(list_qubit_zero_lamb),
        #     np.array(list_detunning_qubit)
        # )
        manager = Manager()
        result_dict = manager.dict()

        args_list = [(self.EJ, EC, EL, Er) for EC, EL, Er in zip(self.EC_values, self.EL_values, self.Er_values)]
        batch_size = int(len(args_list) / os.cpu_count())
        batched_args_list = [args_list[i:i + batch_size] for i in range(0, len(args_list), batch_size)]

        # Use multiprocessing to execute the tasks
        with Pool(processes=os.cpu_count()) as pool:
            pool.starmap(batch_wrapper, [(args, result_dict) for args in enumerate(batched_args_list)])

        # Assemble the results
        sorted_results = []
        for batch_results in sorted(result_dict.values()):
            for index, result in sorted(batch_results):
                sorted_results.append(result)

        # Convert lists to NumPy arrays
        list_one_two_transition, list_differential_stark, list_qubit_zero_lamb, list_detunning_qubit = zip(*sorted_results)
        self.results = (
            np.array(list_one_two_transition),
            np.array(list_differential_stark),
            np.array(list_qubit_zero_lamb),
            np.array(list_detunning_qubit)
        )




def get_estimations(EJ, EC, EL, Er):
    g_strength = 0.3
    qubit_level = 9
    osc_level =16

    qbt = scqubits.Fluxonium(EJ=EJ,EC=EC,EL=EL,flux=0,cutoff=110,truncated_dim=qubit_level)
    q_evals = qbt.eigenvals()
    one_two_transition = q_evals[2]-q_evals[1]
    E_osc = Er
    osc = scqubits.Oscillator(E_osc=E_osc,truncated_dim=osc_level)
    hilbertspace = scqubits.HilbertSpace([qbt, osc])
    hilbertspace.add_interaction(g_strength=g_strength,op1=qbt.n_operator,op2=osc.creation_operator,add_hc=True)
    hilbertspace.generate_lookup()
    product_to_dressed = generate_single_mapping(hilbertspace.hamiltonian())
    energies=  hilbertspace.eigenvals(qubit_level*osc_level)
    def stark(ql1,ql2,ol):
        return abs(energies[product_to_dressed[(ql2,ol)]]-energies[product_to_dressed[(ql1,ol)]])
    def lamb(ol1,ol2,ql):
        return abs(energies[product_to_dressed[(ql,ol2)]]-energies[product_to_dressed[(ql,ol1)]])
    def detuning(ql1,ql2):
        return abs((energies[product_to_dressed[(ql1,1)]]-energies[product_to_dressed[(ql1,0)]])  -
                        (energies[product_to_dressed[(ql2,1)]]-energies[product_to_dressed[(ql2,0)]]))
    
    differential_stark_on_qubit_12_from_osc01 = abs(stark(1,2,0)-stark(1,2,1)) # For reducing differential phase on off-diagonal elements of the qubit
    differential_stark_on_qubit_12_from_osc12 = abs(stark(1,2,1)-stark(1,2,2))
    qubit_zero_lamb_on_osc01_12 = abs(lamb(0,1,0)-lamb(1,2,0)) # For easy populating photons
    qubit_zero_lamb_on_osc01_23 = abs(lamb(0,1,0)-lamb(2,3,0))
    detunning_qubit01 = detuning(0,1) + detuning(0, 2)
    print("done one")
    return  (one_two_transition,#Want it small
             differential_stark_on_qubit_12_from_osc01+differential_stark_on_qubit_12_from_osc12,#Want it small
             qubit_zero_lamb_on_osc01_12+qubit_zero_lamb_on_osc01_23,#Want it small
             detunning_qubit01)#Want it big

# def wrapper_with_index(args):
#     index, (EJ, EC, EL, Er) = args
#     result = get_estimations(EJ, EC, EL, Er)
#     with open(f"result_{index}.pkl", "wb") as f:
#         pickle.dump(result, f)

def batch_wrapper(args, result_dict):
    batch_index, batch_args = args
    batch_results = []
    for i, (EJ, EC, EL, Er) in enumerate(batch_args):
        result = get_estimations(EJ, EC, EL, Er)
        index = batch_index * len(batch_args) + i
        batch_results.append((index, result))
    result_dict[batch_index] = batch_results

    
def main(idx):
    with open(f'{idx}.pkl', 'rb') as f:
        job = pickle.load(f)
    job.run()
    with gzip.open(f'{idx}_output.pkl.gz', 'wb') as f_out:
        pickle.dump(job, f_out)


if __name__ == "__main__":
    idx = int(sys.argv[1])
    main(idx)
