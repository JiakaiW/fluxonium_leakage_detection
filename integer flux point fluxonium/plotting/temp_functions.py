
from scqubits import Fluxonium
qubit_level = 25
def compute_z(params):
    EC, EL, EJ = params
    qbt = Fluxonium(EJ=EJ, EC=EC, EL=EL, flux=0, cutoff=110, truncated_dim=qubit_level)
    num_evals = 3
    evals = qbt.eigenvals(num_evals)
    return evals[2] - evals[1]
