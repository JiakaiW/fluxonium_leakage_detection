
import sys
original_sys_path = sys.path.copy()
sys.path.append('../')
from backup.utils import *

# Step-1: average over kets

files = [
    f'../pickles/mcsolve_2.65_g0.13_a0.003_10level_qbt_state{0}.pkl',
    f'../pickles/mcsolve_2.65_g0.13_a0.003_10level_qbt_state{1}.pkl',
    f'../pickles/mcsolve_2.65_g0.13_a0.003_10level_qbt_state{2}.pkl',
    f'../pickles/mcsolve_2.65_g0.13_a0.003_10level_qbt_state{3}.pkl',
    f'../pickles/mcsolve_2.65_g0.13_a0.003_10level_qbt_state{4}.pkl',
]
for i, file in enumerate(files):
    with open(file, 'rb') as file:
        result = pickle.load(file)

    ntraj = len(result.states)
    print(f'ntraj: {ntraj}')
    states_array = np.array([[state.full() for state in traj] for traj in result.states])
    # The following line averages over n trajectories of kets
    # n is traj index, t is time index, r is row index, c is column index, i and j are the row and column index of the conjugated ket
    summed_dm_array = np.einsum('ntrc,ntij->tri', states_array, states_array.conj()) 
    averaged_dm_array = summed_dm_array/ntraj
    result.states = [qutip.Qobj(dm) for dm in averaged_dm_array]
    with open(f'../pickles/mcsolve_2.65_g0.13_a0.003_10level_qbt_state{i}_summed.pkl', 'wb') as file:
        pickle.dump(result, file)
    


# Step-2: padd back to original shape
qubit_level = 25
osc_level = 40
kappa = 1e-3
system_leakage = fluxonium_oscillator_system(
    EJ = 2.65,
    EC = 0.6,
    EL = 0.13,
    Er = 7.17391479,
    g_strength = 0.13,
    qubit_level = qubit_level,
    osc_level = osc_level,
    kappa =kappa,
    products_to_keep=[[ql, ol] for ql in [0,7] for ol in range(osc_level) ],
    computaional_states = '1,2',
    )

system_computational = fluxonium_oscillator_system(
    EJ = 2.65,
    EC = 0.6,
    EL = 0.13,
    Er = 7.17391479,
    g_strength = 0.13,
    qubit_level = qubit_level,
    osc_level = osc_level,
    kappa =kappa,
    products_to_keep=[[ql, ol] for ql in [1,10,  2,9,11] for ol in range(20) ],
    computaional_states = '1,2',
    )

for i, system in zip([0,1,2,3,4],[system_leakage,system_computational,system_computational,system_computational,system_computational]):
    with open(f'../pickles/mcsolve_2.65_g0.13_a0.003_10level_qbt_state{i}_summed.pkl', 'rb') as file:
        result = pickle.load(file)
    states = []
    for state in tqdm(result.states,desc=f'padding states from result {i}'):
        states.append(system.pad_back_function(state))
    result.states = states
    with open(f'../pickles/mcsolve_2.65_g0.13_a0.003_10level_qbt_state{i}_summed_padd_back.pkl', 'wb') as file:
        pickle.dump(result, file)
        
# Step-3: truncate to two level
for i, system in zip([0,1,2,3,4],[system_leakage,system_computational,system_computational,system_computational,system_computational]):
    with open(f'../pickles/mcsolve_2.65_g0.13_a0.003_10level_qbt_state{i}_summed_padd_back.pkl', 'rb') as file:
        result = pickle.load(file)
    states = []
    for state in tqdm(result.states,desc=f'truncating states from result {i}'):
        states.append(
            dressed_to_2_level_dm(state,
                                  system.product_to_dressed,
                                    system.qubit_level,
                                       system.osc_level,
                                      system.computaional_states[0],
                                      system.computaional_states[1],
                                      products_to_keep = None
                                      )
            )
    result.states = states
    with open(f'../pickles/mcsolve_2.65_g0.13_a0.003_10level_qbt_state{i}_summed_two_level.pkl', 'wb') as file:
        pickle.dump(result, file)