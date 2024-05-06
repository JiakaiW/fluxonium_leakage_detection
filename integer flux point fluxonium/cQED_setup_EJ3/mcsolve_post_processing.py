import multiprocessing
import pickle
from tqdm import tqdm
import sys
sys.path.append('../utils/')
from utils_models import *
from functools import partial

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
computational_products_to_keep = [[ql, ol] for ql in [1,2] for ol in range(max_ol) ]
list_of_products_to_keep = [
    leakage_products_to_keep,
    computational_products_to_keep,
    computational_products_to_keep,
    computational_products_to_keep,
    computational_products_to_keep,
    computational_products_to_keep,
    computational_products_to_keep
]

def get_product(dressed_dm,pad_back_custom,product_to_dressed,sign_multiplier):
    dressed_dm_data =    pad_back_custom(dressed_dm)
    # if dressed_dm_data.shape[1] == 1:
    #     dressed_dm_data = qutip.ket2dm(dressed_dm_data)
    dressed_dm_data = dressed_dm_data.full()

    # Infer subsystem dimensions
    subsystem_dims = [max(indexes) + 1 for indexes in zip(*product_to_dressed.keys())]
    rho_product = np.zeros((subsystem_dims*2), dtype=complex) # Here rho_product is shaped like (dim1,dim2,dim1,dim2)
    for product_state, dressed_index1 in product_to_dressed.items():
        for product_state2, dressed_index2 in product_to_dressed.items():
            element = dressed_dm_data[dressed_index1, dressed_index2] * sign_multiplier[dressed_index1] * sign_multiplier[dressed_index2]
            rho_product[product_state+product_state2] += element # Using index like (lvl1, lvl2, lvl1, lvl2) to access of of the entries

    two_lvl_qbt_dm_size = np.prod(subsystem_dims)
    rho_product = rho_product.reshape((two_lvl_qbt_dm_size,two_lvl_qbt_dm_size))
    rho_product = qutip.Qobj(rho_product, dims=[subsystem_dims, subsystem_dims])
    return rho_product

if __name__ == "__main__":
    with open('averaged.pkl', 'rb') as f:
        results = pickle.load(f)

    num_processes = 8#multiprocessing.cpu_count()


    for i, (result, products_to_keep) in enumerate(zip(results[1:], list_of_products_to_keep[1:])):
        system.set_new_product_to_keep(products_to_keep)
        system.set_new_operators_after_setting_new_product_to_keep()
        

        partial_function = partial(get_product,
                                pad_back_custom = system.pad_back_function,
                                product_to_dressed = system.product_to_dressed,
                                sign_multiplier = system.sign_multiplier)

        with multiprocessing.Pool(processes=num_processes) as pool:
            product_states = pool.map(partial_function,result.states)
        result.states_in_product_basis = product_states
        print(f'{i} done')

    with open('mcsolve_results_with_product_basis_5000traj.pkl', 'wb') as f:
        pickle.dump(results,f)
