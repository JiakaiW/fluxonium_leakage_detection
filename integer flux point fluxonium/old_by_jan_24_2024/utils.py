from bidict import bidict
import ipywidgets as widgets
from IPython.display import clear_output
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import qutip
import scqubits
from typing import List
from tqdm.notebook import tqdm
from dataclasses import dataclass

class fluxonium_oscillator_system:
    def __init__(self,
                EJ:float = 3,
                EC:float = 0.6,
                EL:float = 0.13,
                Er:float = 7.2622522,
                g_strength:float = 0.3,
                qubit_level:float = 30,
                osc_level:float = 30,
                
                products_to_keep: List[List[int]]= None,
                
                ):
        self.qbt = scqubits.Fluxonium(EJ=EJ,EC=EC,EL=EL,flux=0,cutoff=110,truncated_dim=qubit_level)
        self.osc = scqubits.Oscillator(E_osc=Er,truncated_dim=osc_level)
        self.hilbertspace = scqubits.HilbertSpace([self.qbt, self.osc])
        self.hilbertspace.add_interaction(g_strength=g_strength,op1=self.qbt.n_operator,op2=self.osc.creation_operator,add_hc=True)
        self.hilbertspace.generate_lookup()
        self.product_to_dressed = generate_single_mapping(self.hilbertspace.hamiltonian())
        
        if products_to_keep != None:
            self.products_to_keep = products_to_keep
        else:
            self.products_to_keep = [[ql, ol] for ql in range(3) for ol in range(10) ] \
                                + [[ql, ol] for ql in [0] for ol in range(30) ]

        def truncate_function(self,qobj):
            return truncate_custom(qobj, self.products_to_keep, self.product_to_dressed)

        def pad_back_function(self,qobj):
            return pad_back_custom(qobj, self.products_to_keep, self.product_to_dressed)
        
        self.a = qutip.Qobj(self.hilbertspace.op_in_dressed_eigenbasis(op=self.osc.annihilation_operator)[:, :])
        self.a_trunc = truncate_function(self.a)

        (evals,) = self.hilbertspace["evals"]
        diag_dressed_hamiltonian = (
                2 * np.pi * qutip.Qobj(np.diag(evals),
                dims=[self.hilbertspace.subsystem_dims] * 2)
        )
        diag_dressed_hamiltonian = qutip.Qobj(diag_dressed_hamiltonian[:, :])
        diag_dressed_hamiltonian = truncate_function(diag_dressed_hamiltonian)

        w_d = transition_frequency(self.hilbertspace,self.product_to_dressed[(0,0)],self.product_to_dressed[(0,1)] ) 

    def run_mesolve(self,
                    intial_states,
                    ):
        return result


############################################################################
#
#
# Functions about manipulating qutip/scqubit objects
#
#
############################################################################

def truncate_custom(qobj: qutip.Qobj, products_to_keep: list, product_to_dressed: dict) -> qutip.Qobj:
    indices_to_keep = [dressed_level for (qubit_level, oscillator_level), dressed_level in product_to_dressed.items() if [qubit_level, oscillator_level] in products_to_keep]
    indices_to_keep.sort()

    if qobj.shape[1] == 1:  # is ket
        truncated_vector = qobj.full()[indices_to_keep, :]
        return qutip.Qobj(truncated_vector)
    else:  # is operator or density matrix
        truncated_matrix = qobj.full()[np.ix_(indices_to_keep, indices_to_keep)]
        return qutip.Qobj(truncated_matrix)

def pad_back_custom(qobj: qutip.Qobj, products_to_keep: list, product_to_dressed: dict) -> qutip.Qobj:
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
    Maps product of bare states to dressed state
    Returns a dictionary like {(0,0,0):0,(0,0,1):1}
    Use this function instead of scqubit's because I can change the overlap threshold here
    """
    evals, evecs = H_with_interaction_no_drive.eigenstates()
    overlap_matrix = scqubits.utils.spectrum_utils.convert_evecs_to_ndarray(evecs)
    OVERLAP_THRESHOLD = 0.02
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
        if (max_overlap**2 < OVERLAP_THRESHOLD):
            print(f'max overlap^2 {max_overlap**2} below threshold for dressed state {dressed_index} with eval {evals[dressed_index]}')
    product_to_dressed = {}
    for product, dressed in zip(product_state_names,dressed_indices):
        product_to_dressed[product] = dressed
    return product_to_dressed

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

def transition_frequency(hilbertspace,s0: int, s1: int) -> float:
    return (hilbertspace.energy_by_dressed_index(s1)- hilbertspace.energy_by_dressed_index(s0))

def find_dominant_frequency(expectation,tlist,dominant_frequency_already_found = None,plot = False):
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
    else:
        return dominant_freq



def dressed_to_2_level_dm(dressed_dm,product_to_dressed, qubit_level, osc_level,computational_0,computational_1,products_to_keep=None):
    dressed_dm_data = pad_back_custom(dressed_dm, products_to_keep, product_to_dressed)
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


def write_mesolve(w_d,amp,product_to_dressed,products_to_keep,t_stop=None):
    '''
    This function write a function to file, which can then be imported from a jupyter notebook and call multi-processing upon.
    (function defined in a notebook cannot be used by multiprocessing)
    '''
    if os.path.exists('temp_functions.py'):
        os.remove('temp_functions.py')
    with open('temp_functions.py', 'w') as f:
        f.write(f"""
from utils import *
            
w_d = {w_d}
amp = {amp}
t_stop = {t_stop}

product_to_dressed = {product_to_dressed}
products_to_keep = {products_to_keep}
def square_cos(t, *args):
    if t_stop != None:
        if t <= t_stop:
            cos = np.cos(w_d * 2 * np.pi * t)
            return 2 * np.pi * amp * cos
        else:
            return 0
    else:
        cos = np.cos(w_d * 2 * np.pi * t)
        return 2 * np.pi * amp * cos
    

def pad_back_custom(qobj: qutip.Qobj, products_to_keep: list, product_to_dressed: dict) -> qutip.Qobj:
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

def mesolve_and_pad(rho0,
            H_with_drive,
             tlist, 
            full_dim,
            c_ops = None
            ):

    temp = qutip.mesolve(
        H=H_with_drive,
        rho0=rho0,
        tlist=tlist,
        c_ops=c_ops,
        options=qutip.Options(store_states=True, nsteps=20000, num_cpus=1),
        progress_bar = qutip.ui.progressbar.EnhancedTextProgressBar(),
    )

    # padded_states = [pad_back(state, full_dim) for state in temp.states]
    
    padded_states = [pad_back_custom(state, products_to_keep, product_to_dressed) for state in temp.states]
    result = qutip.solver.Result()
    result.times = temp.times
    result.states = padded_states
    return result
""")