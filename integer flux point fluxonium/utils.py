import qutip
import numpy as np
from matplotlib import pyplot as plt
import scqubits
from typing import List
import math
from IPython.display import clear_output
from jax import vmap
import jax.numpy as jnp
from jax import jit
from mcsolve_on_node import packed_mcsolve_problem
import pickle
import zipfile
import os
def generate_single_mapping(H_with_interaction_no_drive) -> np.ndarray:
    """
    Maps product of bare states to dressed state
    Returns a dictionary like {(0,0,0):0,(0,0,1):1}
    Use this function instead of scqubit's because I can change the overlap threshold here
    """
    evals, evecs = H_with_interaction_no_drive.eigenstates()
    overlap_matrix = scqubits.utils.spectrum_utils.convert_evecs_to_ndarray(evecs)
    OVERLAP_THRESHOLD = 0.01
    product_state_names = []
    dims = H_with_interaction_no_drive.dims[0]
    system_size = len(dims)
    def generate_product_states(current_state, ele_index):
        if ele_index == system_size:
            product_state_names.append(tuple(current_state))
            return
        
        for l in range(dims[ele_index]):
            current_state[ele_index] = l
            generate_product_states(current_state.copy(), ele_index + 1)

    current_state = [0] * system_size
    generate_product_states(current_state, 0)

    total_dim = math.prod(dims)
    dressed_indices = [None] * total_dim
    for dressed_index in range(len(evals)):
        max_position = (np.abs(overlap_matrix[dressed_index, :])).argmax()
        max_overlap = np.abs(overlap_matrix[dressed_index, max_position])
        if (max_overlap**2 > OVERLAP_THRESHOLD):
            overlap_matrix[:, max_position] = 0
            dressed_indices[int(max_position)] = dressed_index
        else:
            print(f'max overlap^2 {max_overlap**2} below threshold for dressed state {dressed_index} with eval {evals[dressed_index]}')
    product_to_dressed = {}
    for product, dressed in zip(product_state_names,dressed_indices):
        product_to_dressed[product] = dressed
    return product_to_dressed


def plot_specturum(qubit, resonator, hilbertspace, max_qubit_level = 4,max_resonator_level=3,
                    flagged_transitions = [[[0,0],[0,1]],[[1,0],[1,1]],[[2,0],[2,1]],[[3,0],[3,1]]]):
    product_to_dressed = generate_single_mapping(hilbertspace.hamiltonian())
    energy_text_size = 8
    # clear_output(wait=True)
    qubit_ori_energies = qubit.eigenvals(max_qubit_level)
    resonator_ori_energies = resonator.eigenvals(max_resonator_level)
    fig, old_ax = qubit.plot_wavefunction(which = [0,1,2,3,4,5,6,7])
    left, bottom, width, height = 1, 0, 1, 1  
    ax = fig.add_axes([left, bottom, width, height])
    fig.set_size_inches(4, 4)
    for ql in range(0,max_qubit_level):
        for rl in range(0,max_resonator_level):
            original = (qubit_ori_energies[ql] + resonator_ori_energies[rl])* 2 * np.pi
            x1,x2 = ql-0.25,ql+0.25
            ax.plot([x1, x2], [original, original], linewidth=1, color='red')
            ax.text(ql, original, f"{original:.3f}", fontsize=energy_text_size, ha='center', va='bottom')

            dressed_state_index = product_to_dressed[(ql,rl)]
            dressed = hilbertspace.energy_by_dressed_index(dressed_state_index)* 2 * np.pi
            ax.plot([x1, x2], [dressed, dressed], linewidth=1, color='green')
            ax.text(ql, dressed, f"{dressed:.3f}", fontsize=energy_text_size, ha='center', va='top')

    for transition in flagged_transitions:
        state1, state2 = transition[0],transition[1]
        dressed_index1 = product_to_dressed[(state1[0],state1[1])]
        dressed_index2 = product_to_dressed[(state2[0],state2[1])]
        if dressed_index1!= None and dressed_index2!= None:
            energy1 = hilbertspace.energy_by_dressed_index(dressed_index1)* 2 * np.pi
            energy2 = hilbertspace.energy_by_dressed_index(dressed_index2)* 2 * np.pi
            ax.plot([state1[0], state2[0]], [energy1, energy2], linewidth=1, color='green')
            ax.text((state1[0]+ state2[0])/2, (energy1+ energy2)/2, f"{energy2-energy1:.3f}", fontsize=energy_text_size, ha='center', va='top')
        else:
            print("dressed_state_index contain None")
    plt.show()


def dressed_transition_frequency_over_2pi(hilbertspace,s0, s1) -> float:
    return (hilbertspace.energy_by_dressed_index(s1) - hilbertspace.energy_by_dressed_index(s0))

def replace_non_float64_with_none(lst):
    for i in range(len(lst)):
        if type(lst[i]) is not np.float64:
            lst[i] = None
    return lst


def sweep_resonator_frequency_for_gf_ef_detunning(EJ=8.9,
                                        EC=2.5,
                                        EL=0.5,
                                        flux = 0,
                                        g_strength = 0.3):
    # for erasure detection, we want g0g1 detunned from e0e1 and f0f1
    # for measurement, we want one e0e1 detuned from the rest two.
    E_vals = np.linspace(2, 20, 200)
    g0g1_vals = []
    e0e1_vals = []
    f0f1_vals = []
    h0h1_vals = []

    qbt = scqubits.Fluxonium(
            EJ=EJ,
            EC=EC,
            EL=EL,
            flux=flux,
            cutoff=30,
            truncated_dim=6
        )
    num_done = 0
    num_tot = len(E_vals)
    for e in E_vals:
        osc = scqubits.Oscillator(
            E_osc=e,
            truncated_dim=10
        )
        hilbertspace = scqubits.HilbertSpace([qbt, osc])
        g_strength = 0.5
        hilbertspace.add_interaction(
            g_strength=g_strength,
            op1=qbt.n_operator,
            op2=osc.creation_operator,
            add_hc=True
        )
        hilbertspace.generate_lookup()
        product_to_dressed = generate_single_mapping(hilbertspace.hamiltonian())

        g0g1 = dressed_transition_frequency_over_2pi(hilbertspace,product_to_dressed[(0,0)],product_to_dressed[(0,1)])
        e0e1 = dressed_transition_frequency_over_2pi(hilbertspace,product_to_dressed[(1,0)],product_to_dressed[(1,1)])
        f0f1 = dressed_transition_frequency_over_2pi(hilbertspace,product_to_dressed[(2,0)],product_to_dressed[(2,1)])
        h0h1 = dressed_transition_frequency_over_2pi(hilbertspace,product_to_dressed[(3,0)],product_to_dressed[(3,1)])

        g0g1_vals.append(g0g1)
        e0e1_vals.append(e0e1)
        f0f1_vals.append(f0f1)
        h0h1_vals.append(h0h1)
        g0g1_vals = replace_non_float64_with_none(g0g1_vals)
        e0e1_vals = replace_non_float64_with_none(e0e1_vals)
        f0f1_vals = replace_non_float64_with_none(f0f1_vals)
        h0h1_vals = replace_non_float64_with_none(h0h1_vals)
        num_done+=1
        if num_done%10 == 0:
            clear_output()
            print(f"done:{num_done}/{num_tot}")
    chi_fg_MHz = []
    for a, b in zip(f0f1_vals, g0g1_vals):
        if a is None or b is None:
            chi_fg_MHz.append(None)
        else:
            chi_fg_MHz.append((a - b)*1000)
    chi_fe_MHz = []
    for a, b in zip(f0f1_vals, e0e1_vals):
        if a is None or b is None:
            chi_fe_MHz.append(None)
        else:
            chi_fe_MHz.append((a - b)*1000)

    plt.plot(E_vals, chi_fg_MHz, label=r'$\chi_{\mathrm{fg}}$')
    plt.plot(E_vals, chi_fe_MHz, label=r'$\chi_{\mathrm{fe}}$')
    plt.yscale('symlog')
    plt.legend()
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(True)
    plt.xlabel("resonator frequency (GHz)")
    plt.ylabel("detunning (MHz)")
    plt.show()


def sweep_resonator_frequency_for_ge_gf_gh_detunning(EJ=8.9,
                                        EC=2.5,
                                        EL=0.5,
                                        flux = 0,
                                        g_strength = 0.3):

    E_vals = np.linspace(2, 10, 200)
    transition_name_to_data = {}
    def add_transition_to_list(name:str,hilbertspace,product_to_dressed,product_state1,product_state2):
        freq = dressed_transition_frequency_over_2pi(hilbertspace,product_to_dressed[product_state1],product_to_dressed[product_state2])
        if type(freq) is not np.float64:
            freq = None
        transition_name_to_data.setdefault(name, []).append(freq)

    qubit_levels = 6
    qbt = scqubits.Fluxonium(
            EJ=EJ,
            EC=EC,
            EL=EL,
            flux=flux,
            cutoff=30,
            truncated_dim=qubit_levels
        )
    num_done = 0
    num_tot = len(E_vals)
    ql_names= ['g','e','f','h','i','j']
    for e in E_vals:
        osc = scqubits.Oscillator(E_osc=e, truncated_dim=20)
        hilbertspace = scqubits.HilbertSpace([qbt, osc])
        hilbertspace.add_interaction(g_strength=g_strength,op1=qbt.n_operator,op2=osc.creation_operator,add_hc=True)
        hilbertspace.generate_lookup()
        product_to_dressed = generate_single_mapping(hilbertspace.hamiltonian())

        # Oscillator transition
        for ql in [0,1,2,3]:
            add_transition_to_list(name = f"{ql_names[ql]}0{ql_names[ql]}1",
                                hilbertspace = hilbertspace,
                                product_to_dressed=product_to_dressed,
                                product_state1 = (ql,0),
                                product_state2= (ql,1))
        # Qubit transition
        for q1 in range(qubit_levels-1):
            for q2 in range(q1+1,qubit_levels):
                if q1 in [0,1,2] or q2 in [0,1,2]: # Only plot qubit transitions that's to or from g/e/f
                    add_transition_to_list(name = f"{ql_names[q1]}{ql_names[q2]}",
                                hilbertspace = hilbertspace,
                                product_to_dressed=product_to_dressed,
                                product_state1 = (q1,0),
                                product_state2= (q2,0))

        num_done+=1
        if num_done%10 == 0:
            clear_output()
            print(f"done:{num_done}/{num_tot}")

    # Loop over all transitions in the dictionary
    for name, freq_list in transition_name_to_data.items():
        if name == "g0g1":
            continue
        
        detunning = []
        for a, b in zip(freq_list, transition_name_to_data["g0g1"]):
            if a is None or b is None:
                detunning.append(None)
            else:
                detunning.append((a - b)*1000)
        if name in ["e0e1","f0f1","h0h1"]:# Thick line
            plt.plot(E_vals, detunning, label=r'$\chi_{\mathrm{'+f'{name}'+'-g0g1}}$')
        else:# Thin line
            plt.plot(E_vals, detunning, label=r'$\chi_{\mathrm{'+f'{name}'+'-g0g1}}$', linewidth=1, alpha=0.5)
    plt.yscale('symlog')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(True)
    plt.xlabel("resonator frequency (GHz)")
    plt.ylabel("detunning (MHz)")
    plt.tight_layout()
    plt.show()


# In case alpha oscillates not at drive frequency 
def find_dominant_frequency(expectation,tlist,dominant_frequency_already_found = None,plot = False):
    if dominant_frequency_already_found != None:
        expectation = expectation * np.exp(-1j*2*np.pi*dominant_frequency_already_found*tlist)

    expectation_fft = np.fft.fft(expectation)
    frequencies = np.fft.fftfreq(len(tlist), d=(tlist[1] - tlist[0]))  # assuming tlist is uniformly spaced

    # Identify the dominant frequency: 
    # (we exclude the zero frequency, which usually has the DC offset)
    dominant_freq_idx = np.argmax(np.abs(expectation_fft[1:])) + 1
    dominant_freq = frequencies[dominant_freq_idx]

    if plot:
        # Print the dominant frequency
        print(f"The dominant oscillation frequency is: {dominant_freq:.3f} (in the same units as 1/timestep)")

        fft_shifted = np.fft.fftshift(expectation_fft)
        frequencies_shifted = np.fft.fftshift(frequencies)
        plt.plot(frequencies_shifted, np.abs(fft_shifted))
        plt.xlabel('Frequency (arbitrary units)')
        plt.ylabel('Magnitude')
        plt.title('FFT of the Expectation Value')
        plt.grid(True)
        plt.show()
    else:
        return dominant_freq

def transition_frequency(hilbertspace,s0: int, s1: int) -> float:
    return (hilbertspace.energy_by_dressed_index(s1)- hilbertspace.energy_by_dressed_index(s0))



class CustomOdeResult:
    def __init__(self, t = [], y=[]):
        self.t = t
        self.y = y

def solve_with_mesolve(H,state0,tlist,options = None,c_ops = None):
    return qutip.mesolve(
        H = H,
        rho0=  state0,
        tlist = tlist,
        options=options,
        progress_bar = True,
        c_ops= c_ops
    )


def solve_with_mcsolve(H,state0,tlist,options = None,c_ops = None,ntraj= 50):
    # Also does averaging so it returns a result in the same form as mesolve
    result =  qutip.mcsolve(
        H = H,
        psi0 = state0,
        tlist = tlist,
        options=options,
        progress_bar = True,
        c_ops= c_ops,
        ntraj= 50
    )
    # Averaging over the trajectories
    num_times = len(result.times)
    averaged_states = []

    for t in range(num_times):
        state_t = sum(result.states[traj][t] for traj in range(result.ntraj))
        state_t = state_t / result.ntraj
        averaged_states.append(state_t)

    # Replace the original result's states with the averaged states
    result.states = averaged_states
    return result

def pack_mcsolve_chunks(H,state0,tlist,c_ops,ntraj = 1000,existing_chunk_num: int = 0):
    # Pack chunks that can be sent to htc_condor

    seeds = list(np.random.randint(0, 2**32,
                        size=ntraj,
                        dtype=np.uint32))
    chunk_size = 1

    chunk_id = existing_chunk_num
    # Pack problems
    for i in range(0, ntraj, chunk_size):
        chunk_seeds = seeds[i:i + chunk_size]
        problem = packed_mcsolve_problem(
            H=H,
            state0=state0,
            tlist=tlist,
            options=qutip.Options(store_states=True, nsteps=2000, num_cpus=1,seeds=chunk_seeds),
            c_ops=c_ops,
            ntraj= len(chunk_seeds)
        )

        with open(f"{chunk_id}.pkl", "wb") as f:
            pickle.dump(problem, f)
        chunk_id += 1
    existing_chunk_num = chunk_id
    return existing_chunk_num


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



def aggregate_results(num_chunks):
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



def solve_with_jax_ode_in_chunks(ham_solver, rho0, tot_time, square, num_chunks = 50, num_points_per_chunk = 2):
    pass


def solve_with_jax_gpu(ham_solver, y0, tlist, signals, max_dt=1, chunk_size=10):
    tlist_chunks = []
    for i in range(0, len(tlist) - 1, chunk_size - 1):
        chunk = tlist[i:i + chunk_size]
        tlist_chunks.append(chunk)

    current_state = y0
    chunk_results = []

    for i, chunk in enumerate(tlist_chunks):
        result = ham_solver.solve(
            y0=current_state,
            t_span=[chunk[0], chunk[-1]],
            signals=signals,
            method='jax_expm_parallel',
            t_eval=jnp.linspace(chunk[0], chunk[-1], len(chunk)),
            max_dt=max_dt
        )

        # Append the result, excluding the overlapping state if not the first chunk
        if i == 0:
            chunk_results.extend(result.y)
        else:
            chunk_results.extend(result.y[1:])

        current_state = result.y[-1]
        clear_output(wait=True)
        print(f"Progress: Chunk {i + 1}/{len(tlist_chunks)} solved.")

    ode_result = CustomOdeResult(t=tlist, y=chunk_results)
    return ode_result

def solve_with_jax_gpu_lindbladian(ham_solver, y0, tlist, signals, max_dt=1, chunk_size=10):
    ham_solver.model.evaluation_mode = "dense_vectorized"
    if y0.shape[0] != ham_solver.model.dim**2:
        y0 = y0 @ y0.conj().T
        y0 = y0.flatten(order='F')

    ode_result = CustomOdeResult()
    if chunk_size >= 1:
        ode_result =  solve_with_jax_gpu(ham_solver, y0, tlist, signals, max_dt, chunk_size)

    else:
        current_state = y0
        chunk_results = []
        t_results = []
        total_time = tlist[-1] - tlist[0]
        num_intervals = int(len(tlist) / chunk_size)
        interval_length = total_time / num_intervals

        for i in range(num_intervals):
            t_start = tlist[0] + i * interval_length
            t_end = t_start + interval_length
            t_eval_interval = jnp.array([t_end])

            result = ham_solver.solve(
                y0=current_state,
                t_span=[t_start, t_end],
                signals=signals,
                method='jax_expm_parallel',
                t_eval=t_eval_interval,
                max_dt=max_dt
            )

            chunk_results.append(result.y[-1])
            t_results.append(t_eval_interval[-1])

            current_state = result.y[-1]
            clear_output(wait=True)
            print(f"Progress: Interval {i + 1}/{num_intervals} solved.")
        ode_result = CustomOdeResult(t=t_results, y=chunk_results)
    
    return ode_result


def plot_population(results,qubit_level,osc_level,product_to_dressed,a,w_d,tlist,fourier=False):
    product_states = [(ql,ol) for ql in range(qubit_level) for ol in range(osc_level)]
    idxs = [product_to_dressed[(s1, s2)] for (s1, s2) in product_states]
    tot_dims = qubit_level*osc_level

    nlevels = len(results)


    a_op = jnp.array(a.full())
    pn_op = jnp.array((a.dag()*a).full())


    def compute_expectation(ket_or_dm, operator):
        # Check if the input is a ket or a density matrix
        if ket_or_dm.shape[-1] == 1:  # Input is a ket
            return (jnp.linalg.multi_dot([jnp.conj(ket_or_dm).T, operator, ket_or_dm]))[0][0]
        else:  # Input is a density matrix
            return jnp.trace(jnp.dot(operator, ket_or_dm))

    # Vectorize the function over the kets
    vectorized_compute_expectation = vmap(compute_expectation, in_axes=(0, None))
    vectorized_compute_expectation = jit(vectorized_compute_expectation)

    for i in range(nlevels):
        if hasattr(results[i], 'y'):
            states = jnp.array(results[i].y)  # assuming y contains JAX arrays or density matrices
        elif hasattr(results[i], 'states'):
            states = jnp.stack([jnp.array(q.full()) for q in results[i].states])  # assuming states contains QObj or density matrices

        results[i].expect = []
        for idx in idxs:
            dressed_state = jnp.zeros(tot_dims).at[idx].set(1).reshape(-1, 1)
            dressed_state_op = jnp.outer(dressed_state, jnp.conj(dressed_state).T)
            expectations = vectorized_compute_expectation(states, dressed_state_op)
            results[i].expect.append(expectations)
        alpha_expect = vectorized_compute_expectation(states, a_op)
        pns_expect = vectorized_compute_expectation(states, pn_op)
        results[i].expect.append(alpha_expect)
        results[i].expect.append(pns_expect)

    if fourier == True:
        first_dominant_freq =find_dominant_frequency(results[0].expect[-2],tlist)
    else:
        first_dominant_freq = w_d


    fig, axes = plt.subplots(4,nlevels, figsize=(9, 6))

    for i in range(nlevels):
        qubit_state_population = [np.zeros(shape=len(tlist))]*qubit_level
        for idx, product_state in enumerate(product_states):
            ql = product_state[0]
            qubit_state_population[ql] += results[i].expect[idx]
        for ql in range(nlevels):
            axes[0][i].plot(tlist, qubit_state_population[ql], label=r"$\overline{|%s\rangle}$" % (f"{ql}"))
        

        #*np.exp(-1j * 2 * np.pi * first_dominant_freq * tlist) # *np.exp(-1j * 2 * np.pi * dominant_freq * tlist)  

        alpha = results[i].expect[-2]*np.exp(-1j * 2 * np.pi * first_dominant_freq * tlist)

        # Coherent state eigenval
        real = alpha.real
        imag = alpha.imag
        axes[1][i].plot(tlist,imag , label=r"imag alpha")
        axes[2][i].plot(tlist, real, label=r"real alpha")
        axes[3][i].plot(-imag, real, label=r"imag alpha VS real alpha")
        
        # Photon number
        axes[0][i].plot(tlist, results[i].expect[-1], label=r"photon number")


    axes[0][nlevels-1].legend(loc='center', ncol=1, bbox_to_anchor=(1.5, 0.5))
    axes[1][nlevels-1].legend(loc='center', ncol=1, bbox_to_anchor=(1.3, 0.5))
    axes[2][nlevels-1].legend(loc='center', ncol=1, bbox_to_anchor=(1.3, 0.5))
    axes[3][nlevels-1].legend(loc='center', ncol=1, bbox_to_anchor=(1.4, 0.5))
    plt.ylabel("population")
    plt.xlabel("t (ns)")
    for row in [0,1,2,3]:
        max_x_range,min_x_range,max_y_range,min_y_range = 0,0,0,0
        for col in range(nlevels):
            ymin, ymax = axes[row][col].get_ylim()
            xmin, xmax = axes[row][col].get_xlim()
            if ymax > max_y_range:
                max_y_range = ymax
            if ymin < min_y_range:
                min_y_range = ymin
            if xmax > max_x_range:
                max_x_range = xmax
            if xmin < min_x_range:
                min_x_range = xmin
        for col in range(nlevels):
            axes[row][col].set_ylim(min_y_range, max_y_range)
            axes[row][col].set_xlim(min_x_range,max_x_range)
    # plt.yscale('log')
    plt.show()

