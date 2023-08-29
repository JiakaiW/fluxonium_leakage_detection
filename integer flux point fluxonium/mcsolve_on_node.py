import qutip
import pickle


def square_cos(t,*args):
    cos = np.cos(w_d * 2*np.pi * t)
    return  2*np.pi *0.002 * cos

class packed_mcsolve_problem:
    def __init__(self, H, state0, tlist, options, c_ops,ntraj):
        self.H = H
        self.state0 = state0
        self.tlist = tlist
        self.options = options
        self.c_ops = c_ops
        self.ntraj = ntraj
    def run_mcsolve(self):
        result = qutip.mcsolve(
            H=self.H,
            psi0=self.state0,
            tlist=self.tlist,
            options=self.options,
            c_ops=self.c_ops,
            ntraj=self.ntraj
        )
        return result

def main(idx):
    # Load the packed problem from the .pkl file
    with open(f"{idx}.pkl", "rb") as f:
        problem = pickle.load(f)

    # Run mcsolve
    result = problem.run_mcsolve()

    # Save the result to another .pkl file
    with open(f"result_{idx}.pkl", "wb") as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    import sys
    idx = int(sys.argv[1])  # Get the chunk index from the command line
    main(idx)


