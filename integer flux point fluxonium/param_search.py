import numpy as np
import sys
import pickle
import scqubits
import math
import gzip

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
    def __init__(self, EJ, EC_values, EL_values):
        self.EJ = EJ
        self.EC_values = EC_values
        self.EL_values = EL_values
        self.results = None

    def run(self):
        self.results = np.vectorize(get_estimations)(self.EJ, self.EC_values, self.EL_values)

def get_estimations(EJ, EC, EL):
    g_strength = 0.3
    qubit_level = 9
    osc_level =16

    qbt = scqubits.Fluxonium(EJ=EJ,EC=EC,EL=EL,flux=0,cutoff=110,truncated_dim=qubit_level)
    q_evals = qbt.eigenvals()
    one_two_transition = q_evals[2]-q_evals[1]
    zero_three_transition = q_evals[3] - q_evals[0]
    E_osc = zero_three_transition-0.01
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
    differential_stark_on_qubit_12_from_osc12 = abs(stark(1,2,2)-stark(1,2,0))
    qubit_zero_lamb_on_osc01_12 = abs(lamb(0,1,0)-lamb(1,2,0)) # For easy populating photons
    qubit_zero_lamb_on_osc01_23 = abs(lamb(0,1,0)-lamb(2,3,0))
    detunning_qubit01 = detuning(0,1) + detuning(0, 2)
    return  (one_two_transition,#Want it small
             zero_three_transition,
             differential_stark_on_qubit_12_from_osc01+differential_stark_on_qubit_12_from_osc12,#Want it small
             qubit_zero_lamb_on_osc01_12+qubit_zero_lamb_on_osc01_23,#Want it small
             detunning_qubit01)#Want it big


def main(idx):
    with open(f'{idx}.pkl', 'rb') as f:
        job = pickle.load(f)
    job.run()
    with gzip.GzipFile(fileobj=sys.stdout.buffer, mode='wb') as f_out:
        pickle.dump(job, f_out)

if __name__ == "__main__":
    idx = int(sys.argv[1])
    main(idx)
