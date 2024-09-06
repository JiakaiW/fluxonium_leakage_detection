import multiprocessing
import pickle
from tqdm import tqdm
import sys
sys.path.append('../CoupledQuantumSystems/')
from CoupledQuantumSystems.systems import *

max_ql = 13
max_ol = 15
EJ = 4
EC = EJ/2.7
EL = EJ/45
Er = 10.38695526

g = 0.2
w_d = 10.389507326769158
amp = 0.005
tot_time =100
tlist = np.linspace(0, tot_time, tot_time)
system  =  FluxoniumOscillatorSystem(
                EJ = EJ,
                EC = EC,
                EL = EL,
                Er = Er,
                g_strength = g,
                qubit_level = max_ql,
                osc_level = max_ol,
                products_to_keep=[[ql, ol] for ql in range(max_ql) for ol in range(max_ol) ],
                computaional_states = '1,2',
                )


def loop_function(ql):
    pops_list = [[] for _ in range(max_ql)]
    for t_idx in tqdm(range(len(tlist))[::10], desc=f"t loop for ql={ql}"):
        dm = results[ql].states[t_idx]
        for q in range(max_ql):
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
    with open('sesolve_large_dim_a005.pkl', 'rb') as file:
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
    with open('sesolve_large_dim_pop_list_g0.2a005.pkl', 'wb') as file:
        pickle.dump(lists, file)
