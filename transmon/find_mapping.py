import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

def match_eigenvectors_with_confidence(vecs_old, vecs_new):
    """
    Match 'vecs_new' to 'vecs_old' via maximum overlap and compute a confidence score.
    
    Parameters
    ----------
    vecs_old : np.ndarray, shape (dim, dim)
        The eigenvectors from the previous step (columns = eigenvectors).
    vecs_new : np.ndarray, shape (dim, dim)
        The eigenvectors from the new step.
    
    Returns
    -------
    perm : np.ndarray of shape (dim,)
        The permutation that best matches new vectors to old vectors.
    confidence : float
        A measure of how well the new vectors match the old ones,
        e.g. the average or minimum matched overlap.
    overlap_matrix : np.ndarray of shape (dim, dim)
        Absolute overlaps between old and new vectors.
    """
    # Compute absolute overlap matrix
    overlap_matrix = cupy.abs(vecs_old.T.conj() @ vecs_new).get()
    
    # We want to MAXIMIZE total overlap -> linear_sum_assignment does MINIMIZE cost
    cost = -overlap_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind should be [0,1,2,...,dim-1], col_ind is the permutation
    
    perm = col_ind
    
    # Extract the matched overlaps on the diagonal (old i matched to new perm[i])
    matched_overlaps = overlap_matrix[row_ind, col_ind]
    
    # Confidence metric: here we use the minimum matched overlap,
    # but you could also use np.mean(matched_overlaps).
    
    
    return perm, matched_overlaps, overlap_matrix

def adaptive_g_sweep(
    g_start=0.0,
    g_end=10.0,
    initial_step=1e-3,
    min_conf_threshold=0.98,
    enlarge_step_factor=1.2,
    reduce_step_factor=0.5,
    output_folder="eigen_data",
    store_data=False
):
    """
    Adiabatically sweep 'g' from g_start to g_end, adaptively adjusting steps based on
    eigenvector matching confidence. Stores all eigenvalues/eigenvectors on disk,
    but does NOT keep them all in memory.

    We only keep in memory:
      - The last (old) eigenvectors (for matching)
      - The final accepted eigenvectors
    Everything else is written to disk and not retained in memory.

    Parameters
    ----------
    H0 : np.ndarray
        Base Hamiltonian.
    H_coupling : np.ndarray
        Interaction term to be scaled by g.
    g_start, g_end : float
        Sweep range for g.
    initial_step : float
        Initial guess for the step size.
    min_conf_threshold : float
        If matched overlap is below this, reduce step size.
    max_steps : int
        Max number of accepted steps.
    enlarge_step_factor : float
        Factor by which to multiply step size if confidence is well above threshold.
    reduce_step_factor : float
        Factor by which to multiply step size if confidence is below threshold.
    output_folder : str
        Where to store eigenvalues/vectors. Created if doesn't exist.
    store_data : bool
        If True, write NPZ files for each accepted step.

    Returns
    -------
    g_trajectory : list of float
        The accepted g-values in ascending order.
    final_eigvals : np.ndarray, shape (dim,)
        The eigenvalues for the final g.
    final_eigvecs : np.ndarray, shape (dim, dim)
        The eigenvectors for the final g, aligned to the second-to-last step.
    """
    def get_esys(g):
        hilbertspace = scqubits.HilbertSpace([qbt, osc])
        hilbertspace.add_interaction(g_strength=g, op1=qbt.n_operator, op2=osc.n_operator, add_hc=False)
        hamiltonian_np = hilbertspace.hamiltonian().full()
        hamiltonian_cupy = cupy.array(hamiltonian_np)
        cupy_evals, cupy_evecs = cupy.linalg.eigh(hamiltonian_cupy)
        return cupy_evals, cupy_evecs
    if store_data:
        import os
        os.makedirs(output_folder, exist_ok=True)

    # 1) Diagonalize at g_start
    g_current = g_start
    step_size = initial_step
    
    vals_init, vecs_init =  get_esys(g_current)
    idx_init = np.argsort(vals_init)
    vals_old = vals_init[idx_init]
    vecs_old = vecs_init[:, idx_init]

    # Save the trajectory of accepted g-values
    g_trajectory = [g_current]

    # Optionally store to disk
    if store_data:
        fname = os.path.join(output_folder, f"g_{g_current:.3e}.npz")
        np.savez(fname, vals=vals_old, vecs=vecs_old)
    import os
    os.makedirs('figures', exist_ok=True)

    # We'll keep track only of the latest accepted eigenvals/vecs in memory
    final_eigvals = vals_old
    final_eigvecs = vecs_old

    # 2) Adiabatic stepping loop
    while g_current < g_end:
        g_candidate = g_current + step_size
        if g_candidate > g_end:
            g_candidate = g_end

        # clear_output(wait=True)
        print(f'g_current: {g_current:.3e}, g_candidate: {g_candidate:.3e}, step_size: {step_size:.3e}')

        # Diagonalize at g_candidate
        vals_cand, vecs_cand = get_esys(g_candidate)
        idx_cand = np.argsort(vals_cand)
        vals_cand = vals_cand[idx_cand]
        vecs_cand = vecs_cand[:, idx_cand]

        # Match new eigenvectors to old
        perm, matched_overlaps, overlap_matrix = match_eigenvectors_with_confidence(vecs_old, vecs_cand)

        # Reorder
        vals_cand = vals_cand[perm]
        vecs_cand = vecs_cand[:, perm]

        # Check confidence
        
        clear_output(wait=True)
        fig = plt.figure(figsize=(4,3))
        plt.plot(1- matched_overlaps)
        plt.yscale('log')
        plt.ylim(1e-8,1)
        plt.title(f'''Loss (1-overlap) of matched eigenvectors \n
g_current: {g_current:.3e}, g_candidate: {g_candidate:.3e}, 
step_size: {step_size:.3e}
''',fontsize=10)
        plt.tight_layout()
         # store the plot to figure folder
        plt.savefig(f'figures/g_{g_current:.3e}.png')    
        plt.show()
        

        confidence = np.min(matched_overlaps)
        if confidence < min_conf_threshold:
            print(f"Confidence {confidence} fell below {min_conf_threshold}, reducing step size.")
            # Overlap too small -> reduce step
            step_size *= reduce_step_factor
            if step_size < 1e-12:
                print(f"Step size {step_size} fell below 1e-12, aborting.")
                break
        else:
            # Accept step
            g_current = g_candidate
            g_trajectory.append(g_current)

            # Update old references
            vecs_old = vecs_cand
            final_eigvals = vals_cand
            final_eigvecs = vecs_cand

            # Store to disk
            if store_data:
                fname = os.path.join(output_folder, f"g_{g_current:.3e}.npz")
                np.savez(fname, vals=vals_cand, vecs=vecs_cand)

            # Possibly enlarge step if confidence is well above threshold
            if 1- confidence < (1-min_conf_threshold)*0.2:
                step_size *= enlarge_step_factor

        if np.isclose(g_current, g_end, atol=1e-12):
            break

    # Return minimal in-memory data: the list of g-values, plus the final (vals, vecs)
    return g_trajectory, final_eigvals, final_eigvecs