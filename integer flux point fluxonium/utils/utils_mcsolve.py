

existing_chunk_num = 0
for i in range(4):
    existing_chunk_num = pack_mcsolve_chunks(H = H_with_drive,
                    state0 = qutip.basis(hilbertspace.dimension, product_to_dressed[(i,0)]),
                    tlist = tlist,
                    c_ops  = [decay_term],
                    ntraj = 500,
                    existing_chunk_num = existing_chunk_num,
                    chunk_size = 4)

def pack_pkl_files_to_zip(zip_filename="mcsolve_input.zip"):
    # Create a new ZIP file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Loop through all files in the current directory
        for filename in os.listdir('.'):
            # Check if the file is a .pkl file with an integer name
            name, ext = os.path.splitext(filename)
            if ext == '.pkl' and name.isdigit():
                # Add the file to the ZIP
                zipf.write(filename)
                # Delete the .pkl file
                os.remove(filename)
                
pack_pkl_files_to_zip()

from mcsolve_on_node import *
import pickle
import zipfile
import os
from IPython.display import clear_output

def pack_mcsolve_chunks(
                    y0: qutip.Qobj,
                    tlist: np.array,
                    static_hamiltonian: qutip.Qobj,
                    drive_terms: List[DriveTerm],
                    c_ops: Union[None,List[qutip.Qobj]] = None,
                    e_ops:Union[None,List[qutip.Qobj]] = None,
                        ntraj = 500,
                        existing_chunk_num: int = 0,
                        chunk_size = 4):

    seeds = list(np.random.randint(0, 2**32,
                        size=ntraj,
                        dtype=np.uint32))

    chunk_id = existing_chunk_num

    for i in range(0, ntraj, chunk_size):
        chunk_seeds = seeds[i:i + chunk_size]
        problem = packed_mcsolve_problem(
            y0=y0,
            tlist=tlist,
            static_hamiltonian = static_hamiltonian,
            drive_terms = drive_terms,
            c_ops=c_ops,
            e_ops = e_ops,
            chunk_seeds=chunk_seeds,
        )

        with open(f"{chunk_id}.pkl", "wb") as f:
            pickle.dump(problem, f)
        chunk_id += 1
    existing_chunk_num = chunk_id
    return existing_chunk_num


def pack_pkl_files_to_zip(zip_filename="mcsolve_input.zip"):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir('.'):
            name, ext = os.path.splitext(filename)
            if ext == '.pkl' and name.isdigit():
                zipf.write(filename)
                os.remove(filename)


def merge_results(zip_files):
    # Used to merge mcsolve results on HTC condor
    num_total_states = 0
    averaged_dm_array = None
    tlist = None

    num_files_tot = len(zip_files)
    num_files_done = 0

    for zip_file in zip_files:
        if not os.path.exists(zip_file):
            print(f"File {zip_file} does not exist. Skipping...")
            continue

        with gzip.GzipFile(zip_file, "rb") as f:
            result = pickle.load(f)

        if tlist is None:
            tlist = result.times

        num_new_states = len(result.seeds)
        num_total_states += num_new_states

        # Convert states to density matrices and sum them up
        states_array = np.array([[state.full() for state in traj] for traj in result.states])
        summed_dm_array = np.einsum('ntrc,ntij->tri', states_array, states_array.conj()) 
        if averaged_dm_array is None:
            averaged_dm_array = summed_dm_array
        else:
            averaged_dm_array += summed_dm_array

        num_files_done += 1
        clear_output()
        print(f"done:{num_files_done}/{num_files_tot}")
    averaged_dm_array /= num_total_states
    
    # Convert the final averaged density matrices back to Qobj
    averaged_dms = [qutip.Qobj(dm) for dm in averaged_dm_array]

    final_result = qutip.solver.Result()
    final_result.states = averaged_dms
    final_result.times = tlist

    return final_result


def merge_results_and_slice(zip_files):
    # Used to merge mcsolve results on HTC condor
    num_total_states = 0
    averaged_dm_array = None
    tlist = None

    num_files_tot = len(zip_files)
    num_files_done = 0

    for zip_file in zip_files:
        if not os.path.exists(zip_file):
            print(f"File {zip_file} does not exist. Skipping...")
            continue

        with gzip.GzipFile(zip_file, "rb") as f:
            result = pickle.load(f)

        if tlist is None:
            tlist = result.times[::8]

        num_new_states = len(result.seeds)
        num_total_states += num_new_states

        # Convert states to density matrices and sum them up
        states_array = np.array([[state.full() for state in traj] for traj in result.states[::8]])
        summed_dm_array = np.einsum('ntrc,ntij->tri', states_array, states_array.conj()) 
        if averaged_dm_array is None:
            averaged_dm_array = summed_dm_array
        else:
            averaged_dm_array += summed_dm_array

        num_files_done += 1
        clear_output()
        print(f"done:{num_files_done}/{num_files_tot}")
    averaged_dm_array /= num_total_states
    
    # Convert the final averaged density matrices back to Qobj
    averaged_dms = [qutip.Qobj(dm) for dm in averaged_dm_array]

    final_result = qutip.solver.Result()
    final_result.states = averaged_dms
    final_result.times = tlist

    return final_result


def aggregate_results(num_chunks):
    # Used for time chunks in GPU solver
    aggregated_states = []
    aggregated_seeds = []

    for idx in range(num_chunks):
        with open(f"result_{idx}.pkl", "rb") as f:
            chunk_result = pickle.load(f)
        aggregated_states.extend(chunk_result.states)
        aggregated_seeds.extend(chunk_result.seeds)

    aggregated_result = qutip.solver.Result()
    aggregated_result.states = aggregated_states
    aggregated_result.seeds = aggregated_seeds

    return aggregated_result



