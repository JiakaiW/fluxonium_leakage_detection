
import sys
original_sys_path = sys.path.copy()
sys.path.append('../')
from utils_visualization import *
import concurrent.futures
from loky import get_reusable_executor


EJoverEL_list = np.linspace(3.5,4.5,30)
EJoverEC_list = np.linspace(20,40,30)

def get_transition(EJoverEL,
                   EJoverEC,
                   ):
    EJ = 4
    EL = EJ/EJoverEL
    EC = EJ/EJoverEC
    n_evals = 25
    qbt = scqubits.Fluxonium(EJ = EJ,EC = EC,EL = EL, cutoff = 110,flux = 0,truncated_dim=n_evals)
    evals = qbt.eigenvals(n_evals)
    elements =  qbt.matrixelement_table(operator = "n_operator",evals_count=n_evals)

    def find_closest_transition(Er):
        arr = np.array([evals[7]-evals[0],
                        evals[9]-evals[0],
                        evals[11]-evals[0],
                        evals[13]-evals[0]])
        names = ['07','09','011','013']
        differences = np.abs(arr - Er)
        closest_index = np.argmin(differences)
        return names[closest_index]

    for Er in np.linspace(evals[6]-evals[0],   evals[13]-evals[0],   int(1e4)):
        chi1 = sum([get_shift_accurate(elements[1,ql2], evals[ql2], evals[1], Er) for ql2 in range(n_evals)] )
        chi2 = sum([get_shift_accurate(elements[2,ql2], evals[ql2], evals[2], Er) for ql2 in range(n_evals)] )
        if abs(chi1-chi2) < 1e-4:
            chi0 = sum([get_shift_accurate(elements[0,ql2], evals[ql2], evals[0], Er) for ql2 in range(n_evals)] )
            if abs(chi0-chi1)>1e-1:
                return Er/EJ, find_closest_transition(Er)
    return None, None

def process_grid_element(args):
    i, j, EJoverEL, EJoverEC = args
    Er_over_EJ, closest_transition = get_transition(EJoverEL, EJoverEC)
    return i, j, Er_over_EJ, closest_transition

Er_over_EJ_grid = np.zeros((len(EJoverEL_list), len(EJoverEC_list)))
closest_transition_grid = np.empty((len(EJoverEL_list), len(EJoverEC_list)), dtype=object)

args_list = [(i, j, EJoverEL, EJoverEC) for i, EJoverEL in enumerate(EJoverEL_list) for j, EJoverEC in enumerate(EJoverEC_list)]

with get_reusable_executor(max_workers=None, context='loky') as executor:
    futures = {executor.submit(process_grid_element, arg): i for i, arg in enumerate(args_list)}
    
    for future in concurrent.futures.as_completed(futures):
        original_index = futures[future]
        i, j, Er_over_EJ, closest_transition = future.result()
        Er_over_EJ_grid[i, j] = Er_over_EJ
        closest_transition_grid[i, j] = closest_transition

# Print the results
print("Er/EJ grid:")
print(Er_over_EJ_grid)
print("\nClosest transition grid:")
print(closest_transition_grid)