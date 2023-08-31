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

        # Add an attribute to store the number of trajectories before averaging
        result.num_trajectories_before_averaging = len(self.chunk_seeds)

        # Convert the list of Qobj states to a NumPy array for efficient averaging
        states_array = np.array([[state.full() for state in traj] for traj in result.states])

        # Average over the trajectories (axis 0 is the trajectory axis)
        averaged_states_array = np.mean(states_array, axis=0)

        # Convert the averaged states back to Qobj
        averaged_states = [qutip.Qobj(state) for state in averaged_states_array]

        # Replace the states in the result object with the averaged states
        result.states = averaged_states


        return result

def main(idx):
    # Load the packed problem from the .pkl file
    with open(f"{idx}.pkl", "rb") as f:
        problem = pickle.load(f)

    # Run mcsolve
    result = problem.run_mcsolve()

    # This is essentially keeping only one out of four timeslices to reduce size
    result.states = result.states[::4]
    result.times = result.times[::4]

    with gzip.GzipFile(fileobj=sys.stdout.buffer, mode='wb') as f_out:
        pickle.dump(result, f_out)

if __name__ == "__main__":
    import sys
    idx = int(sys.argv[1])  # Get the chunk index from the command line
    main(idx)


