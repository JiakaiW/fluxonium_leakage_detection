import multiprocessing
import pickle
from tqdm import tqdm
import sys
sys.path.append('../utils/')
from utils_models import *

max_ql = 30
max_ol = 100
EJ = 3
EC = EJ/4
EL = EJ/20.5
Er = 8.46111172

g = 0.2
w_d = 8.460155465243822
amp = 0.003

tot_time =920
tlist = np.linspace(0, tot_time, tot_time)
system  =  FluxoniumOscillatorSystem(
                EJ = EJ,
                EC = EC,
                EL = EL,
                Er = Er,
                g_strength = g,
                qubit_level = max_ql,
                osc_level = max_ol,
                products_to_keep=[[ql, ol] for ql in range(15) for ol in range(max_ol) ],
                computaional_states = '1,2',
                )


def loop_function(ql):
    pops_list = [[] for _ in range(15)]
    for t_idx in tqdm(range(len(tlist))[::10], desc=f"t loop for ql={ql}"):
        dm = results[ql].states[t_idx]
        for q in range(15):
            sum_at_this_t_for_this_q = 0
            for o in range(system.osc.truncated_dim):
                product_state = (q, o)
                dressed_state = system.product_to_dressed[product_state]
                basis_state = system.truncate_function(qutip.basis(max_ql*max_ol, dressed_state))
                expectation_value = qutip.expect(basis_state * basis_state.dag(), dm)
                sum_at_this_t_for_this_q += expectation_value
            pops_list[q].append(sum_at_this_t_for_this_q)
    return pops_list

if __name__ == "__main__":
    with open('sesolve_large_dim_a003.pkl', 'rb') as file:
        results = pickle.load(file)
    lists = []
    # Define the number of processes
    num_processes = multiprocessing.cpu_count()
    
    # Create a pool of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the loop function to different ql values
        results = pool.map(loop_function, [0, 1, 2])

    # Combine results
    lists.extend(results)

    # Save the lists to a pickle file
    with open('sesolve_large_dim_pop_list_g0.2a003.pkl', 'wb') as file:
        pickle.dump(lists, file)
