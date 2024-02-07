import qutip
import pickle
import numpy as np
import zipfile
import gzip
from dataclasses import dataclass
from typing import List

amp = 0.004

w_d = 7.2647876791261305


def square_cos(t,*args):
    cos = np.cos(w_d * 2*np.pi * t)
    return  2*np.pi * amp * cos


@dataclass
class packed_mcsolve_problem:
    H: List
    psi0: qutip.qobj.Qobj
    tlist: np.ndarray
    chunk_seeds: List
    c_ops: List
    e_ops: List
    keep_dms: bool = False

    def run_mcsolve(self):
        result = qutip.mcsolve(
            H=self.H,
            psi0=self.psi0,
            tlist=self.tlist,
            options=qutip.Options(store_states=self.keep_dms, num_cpus=len(self.chunk_seeds),seeds=self.chunk_seeds),
            c_ops=self.c_ops,
            e_ops=self.e_ops,
            ntraj=len(self.chunk_seeds),
            progress_bar=False
        )

        return result

def main(idx):
    # Load the packed problem from the .pkl file
    with open(f"{idx}.pkl", "rb") as f:
        problem = pickle.load(f)

    # Run mcsolve
    result = problem.run_mcsolve()


    with gzip.GzipFile(fileobj=sys.stdout.buffer, mode='wb') as f_out:
        pickle.dump(result, f_out)

if __name__ == "__main__":
    import sys
    idx = int(sys.argv[1])  # Get the chunk index from the command line
    main(idx)


