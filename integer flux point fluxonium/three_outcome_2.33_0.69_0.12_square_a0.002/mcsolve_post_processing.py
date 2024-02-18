
import sys
original_sys_path = sys.path.copy()
sys.path.append('../')
from utils import *

# Step-1: average over kets

files = [
    f'../pickles/mcsolve_three_outcome_state{4}_233.pkl',
    f'../pickles/mcsolve_three_outcome_state{5}_233.pkl'
]
for i, file in zip([4,5],files):
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
    with open(f'../pickles/mcsolve_three_outcome_state{i}_233_summed.pkl', 'wb') as file:
        pickle.dump(result, file)
    


# Step-2: padd back to original shape
qubit_level = 25
osc_level = 50
kappa = 1e-3
# system_leak0 = fluxonium_oscillator_system(
#         EJ = 2.5,
#         EC = 0.65,
#         EL = 0.12,
#         Er = 7.05877808,
#         g_strength = 0.2,
#         qubit_level = qubit_level,
#         osc_level = osc_level,
#         kappa =kappa,
#         products_to_keep=[[ql, ol] for ql in [0] for ol in range(45) ] + [[ql, ol] for ql in [7] for ol in range(30)],
#         computaional_states = '1,2',
#         w_d = 7.0585
#         )

# system_leak3 = fluxonium_oscillator_system(
#     EJ = 2.5,
#     EC = 0.65,
#     EL = 0.12,
#     Er = 7.05877808,
#     g_strength = 0.2,
#     qubit_level = qubit_level,
#     osc_level = osc_level,
#     kappa =kappa,
#     products_to_keep=[[ql, ol] for ql in [3] for ol in range(45) ] + [[ql, ol] for ql in [12] for ol in range(30)],
#     computaional_states = '1,2',
#     w_d = 7.0585
#     )

qubit_level = 25
osc_level = 30
kappa = 1e-3
system_computational = fluxonium_oscillator_system(
    EJ = 2.33,
    EC = 0.69,
    EL = 0.12,
    Er = 7.16518677,
    g_strength = 0.18,
    qubit_level = qubit_level,
    osc_level = osc_level,
    kappa =kappa,
    products_to_keep=[[ql, ol] for ql in [1,2] for ol in range(20) ]+ \
                    [[ql, ol] for ql in [9,10,11] for ol in range(10) ],
    computaional_states = '1,2',
    w_d = 7.16475
    )


# for i, system in zip([0,1, 2,3,4,5,6,7],[system_leak0, system_leak3, system_computational,system_computational,system_computational,system_computational,system_computational,system_computational]):
for i, system in zip([4,5],[system_computational,system_computational]):
    with open(f'../pickles/mcsolve_three_outcome_state{i}_233_summed.pkl', 'rb') as file:
        result = pickle.load(file)
    states = []
    for state in tqdm(result.states,desc=f'padding states from result {i}'):
        states.append(system.pad_back_function(state))
    result.states = states
    with open(f'../pickles/mcsolve_three_outcome_state{i}_233_summed_padd_back.pkl', 'wb') as file:
        pickle.dump(result, file)
        
# Step-3: truncate to two level
# for i, system in zip([0,1,2,3,4,5,6,7],[system_leak0, system_leak3, system_computational,system_computational,system_computational,system_computational,system_computational,system_computational]):
for i, system in zip([4,5],[system_computational,system_computational]):
    with open(f'../pickles/mcsolve_three_outcome_state{i}_233_summed_padd_back.pkl', 'rb') as file:
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
    with open(f'../pickles/mcsolve_three_outcome_state{i}_233_summed_two_level.pkl', 'wb') as file:
        pickle.dump(result, file)