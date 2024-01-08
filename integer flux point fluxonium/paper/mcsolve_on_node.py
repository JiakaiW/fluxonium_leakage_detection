import qutip
import pickle
import numpy as np
import zipfile
import gzip

amp = 0.002

w_d = 2.9951027102366083


def square_cos(t,*args):
    cos = np.cos(w_d * 2*np.pi * t)
    return  2*np.pi * amp * cos

class packed_mcsolve_problem:
    def __init__(self, H, state0, tlist, chunk_seeds, c_ops):
        self.H = H
        self.state0 = state0
        self.tlist = tlist
        self.chunk_seeds = chunk_seeds
        self.c_ops = c_ops
    def run_mcsolve(self):
        result = qutip.mcsolve(
            H=self.H,
            psi0=self.state0,
            tlist=self.tlist,
            options=qutip.Options(store_states=True, nsteps=2000, num_cpus=len(self.chunk_seeds),seeds=self.chunk_seeds),
            c_ops=self.c_ops,
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


