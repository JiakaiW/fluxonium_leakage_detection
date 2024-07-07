import sys
sys.path.append('../utils/')
from utils_models import *
import os
import gzip
import pickle
import qutip
from IPython.display import clear_output
from tqdm import tqdm
from multiprocessing import Pool

max_ql = 30
max_ol = 75
EJ = 3
EC = EJ/4
EL = EJ/20.5
Er = 8.46111172

g = 0.2
w_d = 8.460155465243822
amp = 0.003

tot_time =660
tlist = np.linspace(0, tot_time, tot_time)[::5]
kappa = 1e-3

system =  FluxoniumOscillatorSystem(
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
leakage_products_to_keep = [[ql, ol] for ql in [0,7] for ol in range(max_ol) ]

with open('mcsolve_results_with_product_basis_5000traj.pkl', 'rb') as f:
    results = pickle.load(f)

system.set_new_product_to_keep(leakage_products_to_keep)
system.set_new_operators_after_setting_new_product_to_keep()

basis_ops = []
for o in range(max_ol):
    product_state = (0, o)
    dressed_state = system.product_to_dressed[product_state]
    basis_state = system.truncate_function(qutip.basis(max_ql * max_ol, dressed_state))
    basis_ops.append(basis_state * basis_state.dag())

def process_trajectory(traj):
    leakage_traj = []
    for state in tqdm(traj[::10]):
        pop_sum = 1
        for o in range(max_ol):
            expectation_value = qutip.expect(basis_ops[o], state)
            pop_sum -= expectation_value
        leakage_traj.append(pop_sum)
    return leakage_traj

def process_file(zip_file):
    if not os.path.exists(zip_file):
        print(f"File {zip_file} does not exist. Skipping...")
        return []
    with gzip.GzipFile(zip_file, "rb") as f:
        result = pickle.load(f)
    # result.states is shaped like [ntraj, ntimes]
    with Pool() as pool:
        leakage_pops = pool.starmap(process_trajectory, [(traj) for traj in result.states])
    return leakage_pops

n_parts = 7
zip_files = [f"zipped_results/result_{i}.zip" for i in range(1750)]
part_length = len(zip_files) // n_parts
zip_file_parts = [zip_files[i * part_length : (i + 1) * part_length] for i in range(n_parts)]

zip_files = [f"zipped_results_first1000/result_{i}.zip" for i in range(1750)]
part_length = len(zip_files) // n_parts
for i in range(n_parts):
    zip_file_parts[i].extend(zip_files[i * part_length : (i + 1) * part_length])

filenames_for_zero = zip_file_parts[0]

leakage_pop = []
for zip_file in filenames_for_zero:
    print(zip_file)
    leakage_pop.extend(process_file(zip_file))

with open('indi_leakage_pop.pkl', 'wb') as f:
    pickle.dump(leakage_pop,f)