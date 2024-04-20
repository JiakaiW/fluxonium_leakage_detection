import qutip
import pickle
import numpy as np
import gzip
from dataclasses import dataclass
from typing import List,Union
from utils_drive import *


@dataclass
class packed_mcsolve_problem:
    y0: qutip.Qobj
    tlist: np.array
    static_hamiltonian: qutip.Qobj
    drive_terms: List[DriveTerm]
    c_ops: Union[None,List[qutip.Qobj]] = None
    e_ops:Union[None,List[qutip.Qobj]] = None
    chunk_seeds: List

    def run_mcsolve(self):
        H_with_drives =  [self.static_hamiltonian] + \
            [[drive_term.driven_op, drive_term.pulse_shape_func] for drive_term in self.drive_terms]
    
        result = qutip.mcsolve(
            H=H_with_drives,
            psi0=self.y0,
            tlist=self.tlist,
            options=qutip.Options(store_states=True, num_cpus=len(self.chunk_seeds),seeds=self.chunk_seeds),
            c_ops=self.c_ops,
            e_ops=self.e_ops,
            ntraj=len(self.chunk_seeds),
            progress_bar=False
        )

        return result

def main(idx):
    with open(f"{idx}.pkl", "rb") as f:
        problem = pickle.load(f)
    result = problem.run_mcsolve()

    # dumping to stdout may seem strange but condor nodes sometimes don't pickup the files that I store.
    with gzip.GzipFile(fileobj=sys.stdout.buffer, mode='wb') as f_out:
        pickle.dump(result, f_out)

if __name__ == "__main__":
    import sys
    idx = int(sys.argv[1])
    main(idx)