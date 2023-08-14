# Built-in modules
import os
import math
import time
from multiprocessing import Pool, cpu_count
from scipy.interpolate import griddata
import uuid 
# Third-party modules
import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import qutip
from qutip import *
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import clear_output, display, update_display, HTML
import PIL
from typing import List
# Specific functions/classes from third-party library 
from scqubits import *  # consider importing specific functions/classes
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import clear_output,  HTML
import uuid
import os
import imageio
import qutip as qt

import math
import scqubits


def generate_single_mapping(H_with_interaction_no_drive) -> np.ndarray:
    """
    Returns a dictionary like {(0,0,0):0,(0,0,1):1}
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


def find_dominant_frequency(expectation,tlist,dominant_frequency_already_found = None,plot = False):
    if dominant_frequency_already_found != None:
        expectation = expectation * np.exp(-1j*2*np.pi*dominant_freq*tlist)

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


def gaussian(t, A, T, t0 = None, sigma = None):
    if sigma == None:
        sigma = T / 6.0
    if t0 == None:
        t0 = T / 2
    return A * np.exp(-(t - t0)**2 / (2 * sigma**2))

def drive_strength_for_gaussian_pulse_given_total_time( T = 100,  # total length of the pulse,
                                                        t0 = None,  # center of the pulse
                                                        sigma = None  # width of the Gaussian
                                                        ):
    # need to multiply by 2pi in qutip
    # Need to divide the amplitude by abs(matrix element)
    if sigma == None:
        sigma = T / 6.0
    if t0 == None:
        t0 = T / 2.2

    # Objective function: the difference between the integral and pi
    def objective(A):
        integral, upper_bound_on_error = quad(gaussian, t0 - T/2, t0 + T/2, args=(A,T))# args contain additional args other than t
        return np.abs(integral - np.pi)

    # Find the amplitude that makes the integral of the Gaussian pulse equal to pi
    def find_amplitude(t0, T, sigma):
        res = minimize_scalar(objective, args=(), bounds=(0,10), method='bounded')
        return res.x

    # Find the amplitude
    A = find_amplitude(t0, T, None)
    # print(f"Amplitude A for a pi pulse: {A:.4f}")
    return A




cmap = LinearSegmentedColormap.from_list(
    'custom', 
    [(0, 'black'), (0.333, 'darkred'),(0.666, 'orange'), (1, 'white')], 
    N=256
)

'''
Zhenyiqi_version qfunction
'''
def Husimi_Q(alpha:complex,
             t:float,
            reduced_rho: qutip.Qobj ,
            osc_levels: int ,
            w_d: float ):
    sum_part_of_the_coherent_state_Alpha  = qutip.Qobj(dims = [[osc_levels],[1]])
    for n in range(osc_levels):
        sum_part_of_the_coherent_state_Alpha += alpha**n * np.exp(-1j*w_d*t) / np.sqrt(math.factorial(n)) * qutip.basis(osc_levels,n)
        # sum_part_of_the_coherent_state_Alpha +=alpha**n*np.exp(1j*w_d*t)/np.sqrt(math.factorial(n))*qutip.basis(osc_levels,n)
        # sum_part_of_the_coherent_state_Alpha +=alpha**n/np.sqrt(math.factorial(n))*qutip.basis(osc_levels,n)
    Alpha = np.exp(-abs(alpha)**2/2)*sum_part_of_the_coherent_state_Alpha
    return (1/np.pi*Alpha.dag() * reduced_rho * Alpha)[0].real

def Husimi_helper_function(args):
    return Husimi_Q(*args)

def parallel_Husimi_Q(alpha, t,reduced_rho, osc_levels, w_d):
    with Pool(os.cpu_count()) as p:
        R = np.array(p.map(Husimi_helper_function, [(alpha, t,reduced_rho, osc_levels, w_d) for alpha in alpha.ravel()])).reshape(alpha.shape)
    return R

def plot_t_dep_Husimi(t,tlist,results,qubit_levels,osc_levels,hilbertspace,w_d, file_name = None):
    t = min(enumerate(tlist), key=lambda x:abs(x[1]-t))[0]
    fig, axes = plt.subplots(1, 4, figsize=(9, 3))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = np.linspace(-2*np.pi, 2*np.pi, 100)
    X, Y = np.meshgrid(x, y)
    alpha = X + 1j*Y
    if type(w_d) == float:
        w_d = [w_d]*4
    for initial_state in [0,1,2,3]: # for initial state g,e,f
        state = results[initial_state].states[t]
        
        # Rewrite the state in QuTip as a product-like  state
        if state.isket:
            qutip_product_state = qutip.Qobj(np.zeros((qubit_levels*osc_levels, 1)), dims = [[qubit_levels,osc_levels],[1,1]])
            for level in range(hilbertspace.dimension):
                (ql,ol) = hilbertspace.bare_index(level)
                qutip_product_state += complex(state[level]) * qutip.tensor(qutip.basis(qubit_levels, ql), qutip.basis(osc_levels, ol))
            rho = qutip.ket2dm(qutip_product_state)

        else:
            rho = qutip.Qobj(dims = [[qubit_levels,osc_levels],[qubit_levels,osc_levels]])
            for l1 in range(hilbertspace.dimension):
                (ql1,ol1) = hilbertspace.bare_index(l1)
                for l2 in range(hilbertspace.dimension):
                    (ql2,ol2) = hilbertspace.bare_index(l2)
                    rho += complex(state[l1,l2]) \
                                        * qutip.tensor(qutip.basis(qubit_levels, ql1), qutip.basis(osc_levels, ol1))\
                                        * qutip.tensor(qutip.basis(qubit_levels, ql2), qutip.basis(osc_levels, ol2)).dag()        
        reduced_rho = qutip.ptrace(rho, 1)
        reduced_rho = reduced_rho / reduced_rho.norm()
        # if np.any(reduced_rho.eigenenergies() < 0):
        #     raise ValueError(f"{reduced_rho.eigenenergies()}")

        R = parallel_Husimi_Q(alpha,t, reduced_rho, osc_levels, w_d[initial_state])
        # R = qutip.qfunc(state = reduced_rho,xvec=x,yvec=y)
        im = axes[initial_state].imshow(R, extent=(-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi), 
                                        origin='lower', 
                                        cmap=cmap,
                                        vmin=0, 
                                        vmax=0.4)

    fig.colorbar(im, cax=cbar_ax)
    if file_name == None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.clf()
        plt.close()

def plot_t_dep_Husimi_coarse_grid_then_refine(t, tlist, results, qubit_levels, osc_levels, hilbertspace, w_d):
    ## This version uses a coarse grid first and then selectively compute the non-zero places
    ## But it turns out this actually runs slower than just use a fine grid
    t = min(enumerate(tlist), key=lambda x:abs(x[1]-t))[0]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    x_coarse = np.linspace(-2*np.pi, 2*np.pi, 20)
    y_coarse = np.linspace(-2*np.pi, 2*np.pi, 20)
    alpha_coarse = x_coarse[:, None] + 1j*y_coarse

    x_fine = np.linspace(-2*np.pi, 2*np.pi, 100)
    y_fine = np.linspace(-2*np.pi, 2*np.pi, 100)
    alpha_fine = x_fine[:, None] + 1j*y_fine

    for initial_state in [0,1,2]: # for initial state g,e,f
        state = results[initial_state].states[t]
        qutip_product_state = qutip.Qobj(dims = [[qubit_levels, osc_levels], [1, 1]])
        for level in range(hilbertspace.dimension):
            (ql,ol) = hilbertspace.bare_index(level)
            qutip_product_state += complex(state[level]) * qutip.tensor(qutip.basis(qubit_levels, ql), qutip.basis(osc_levels, ol))
        rho = qutip.ket2dm(qutip_product_state)
        reduced_rho = qutip.ptrace(rho, 1)



        R_coarse = parallel_Husimi_Q(alpha_coarse,t, reduced_rho, osc_levels, w_d).reshape(Z_coarse.shape)

        # create the mask
        threshold = 0
        mask_coarse = np.abs(R_coarse) > threshold

        # Create a mask of the same shape as alpha_fine from alpha_coarse
        mask_fine = griddata((alpha_coarse.real.ravel(), alpha_coarse.imag.ravel()), 
                            mask_coarse.ravel(), 
                            (alpha_fine.real.ravel(), alpha_fine.imag.ravel()), 
                            method='nearest').reshape(alpha_fine.shape)

        # Use the mask to find significant regions
        alpha_significant = alpha_fine[mask_fine > 0]

        # Compute Husimi Q function on finer grid at significant regions only
        R_fine_significant = parallel_Husimi_Q(alpha_significant, t,reduced_rho, osc_levels, w_d)

        # Initialize R_fine as all zeros and then fill in the significant regions
        R_fine = np.zeros_like(alpha_fine, dtype=np.float64)
        R_fine[mask_fine > 0] = R_fine_significant

        im = axes[initial_state].imshow(np.abs(R_fine), extent=(-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi), origin='lower', cmap=cmap )
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

def make_husimi_widget(tlist,results,qubit_levels,osc_levels,hilbertspace,w_d):
    inter =  interactive(plot_t_dep_Husimi,
                t = (tlist[0], tlist[-1], (tlist[-1]-tlist[0])/len(tlist)),
                tlist= fixed(tlist),
                results = fixed(results),
                qubit_levels=fixed(qubit_levels),
                osc_levels=fixed(osc_levels),
                hilbertspace=fixed(hilbertspace),
                w_d=fixed(w_d),
                file_name = fixed(None))
    return inter

def make_husimi_gif_zhen_yi_qi_version(tlist,results,qubit_levels,osc_levels,hilbertspace,w_d):
    folder = str(uuid.uuid4())
    os.makedirs(folder)
    num_steps = 40
    step = len(tlist) // num_steps
    chosen_t_indices = [i * step for i in range(num_steps)]
    chosen_t_indices.append(len(tlist)-1)
    file_name_list = []
    num_finished = 0
    for t_idx in chosen_t_indices:
        file_name = folder+'/'+str(t_idx) + '.png'
        plot_t_dep_Husimi(tlist[t_idx],tlist,results,qubit_levels,osc_levels,hilbertspace,w_d, file_name = file_name)
        file_name_list.append(file_name)
        clear_output(wait=True)
        num_finished += 1
        print(f'finished {num_finished} out of {num_steps}')
    with imageio.get_writer(f'{folder}/Husimi.gif', mode='I') as writer:
        for file_name in file_name_list:
            image = imageio.imread(file_name)
            writer.append_data(image)
    clear_output(wait=True)
    HTML(f'<img src="{folder}/Husimi.gif">')



'''
qutip.qfunc
'''

def convert_dressed_state_to_LC_basis_product_state(dressed_state:qutip.Qobj,qubit_levels,osc_levels,product_to_dressed):
    # This conversion is an approximation for easy partial trace later
    dressed_state = dressed_state.data.toarray()
    product_state = np.zeros((qubit_levels*osc_levels,1), dtype=np.complex128)
    for ql in range(qubit_levels):
        for ol in range(osc_levels):
            index_in_product_state = ql*osc_levels + ol
            corresponding_dressed_index = product_to_dressed[(ql,ol)]
            product_state[index_in_product_state][0] = dressed_state[corresponding_dressed_index][0]

    product_state = qutip.Qobj(product_state,dims = [[qubit_levels ,osc_levels],[1,1]])
    return product_state

def plot_t_dep_Husimi_qt_version(t,tlist,results,qubit_levels,osc_levels,product_to_dressed,dominant_freq = 0.0,file_name = None):
    t = min(enumerate(tlist), key=lambda x:abs(x[1]-t))[0]
    fig, axes = plt.subplots(1, 4, figsize=(8, 3))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    
    a_bare = qt.destroy(osc_levels)
    rotation_operator = []

    if type(dominant_freq) == float:
        dominant_freq = [dominant_freq]*4


    for initial_state in [0,1,2,3]: 
        state = results[initial_state].states[t]
        product_state = convert_dressed_state_to_LC_basis_product_state(state,qubit_levels,osc_levels,product_to_dressed)
        reduced_rho = qt.ptrace(product_state, 1)

        theta = dominant_freq[initial_state] * t
        rotation_operator = (1j * a_bare.dag() * a_bare * theta).expm()
        reduced_rho_rotated = rotation_operator * reduced_rho * rotation_operator.dag()
        reduced_rho_rotated = (reduced_rho_rotated + reduced_rho_rotated.dag()) / 2
        reduced_rho_rotated = reduced_rho_rotated / reduced_rho_rotated.tr()
        reduced_rho = reduced_rho_rotated

        R = qt.qfunc(state = reduced_rho,xvec=x,yvec=x)
        im = axes[initial_state].imshow(R, extent=(-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi), 
                                        origin='lower',  cmap=cmap,  vmin=0, vmax=1/np.pi)
    fig.colorbar(im, cax=cbar_ax)
    if file_name == None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.clf()
        plt.close()

def make_husimi_gif_qt_version(tlist,results,qubit_levels,osc_levels,product_to_dressed,dominant_freq=0.0):
    folder = str(uuid.uuid4())
    os.makedirs(folder)
    num_steps = 40
    step = len(tlist) / num_steps
    chosen_t_indices = [i * step for i in range(num_steps)]
    chosen_t_indices.append(len(tlist)-1)
    file_name_list = []
    num_finished = 0
    for t_idx in chosen_t_indices:
        file_name = folder+'/'+str(t_idx) + '.png'
        plot_t_dep_Husimi_qt_version(tlist[int(t_idx)],tlist,results,qubit_levels,osc_levels,product_to_dressed,dominant_freq,file_name = file_name)
        file_name_list.append(file_name)
        clear_output(wait=True)
        num_finished += 1
        print(f'finished {num_finished} out of {num_steps}')
    with imageio.get_writer(f'{folder}/Husimi.gif', mode='I') as writer:
        for file_name in file_name_list:
            image = imageio.imread(file_name)
            writer.append_data(image)
    clear_output(wait=True)
    HTML(f'<img src="{folder}/Husimi.gif">')




def plot_specturum(qubit, 
                   resonator, 
                   hilbertspace, 
                   max_qubit_level = 4,
                   max_resonator_level=3,
                    flagged_transitions = [[[0,0],[0,1]],[[1,0],[1,1]],[[2,0],[2,1]],[[3,0],[3,1]]],message = ''):
    bare_color = 'black'
    dressed_color = 'red'
    energy_text_size = 8
    clear_output(wait=True)
    product_to_dressed = generate_single_mapping(hilbertspace.hamiltonian())

    qubit_ori_energies = qubit.eigenvals(max_qubit_level)
    resonator_ori_energies = resonator.eigenvals(max_resonator_level)
    fig, old_ax = qubit.plot_wavefunction(which = [0,1,2,3,4])
    left, bottom, width, height = 1, 0, 1, 1  
    ax = fig.add_axes([left, bottom, width, height])
    fig.set_size_inches(4, 4)
    for ql in range(0,max_qubit_level):
        for rl in range(0,max_resonator_level):
            original = (qubit_ori_energies[ql] + resonator_ori_energies[rl])* 2 * np.pi
            x1,x2 = ql-0.25,ql+0.25
            ax.plot([x1, x2], [original, original], linewidth=1, color=bare_color)
            ax.text(ql, original, f"{original:.3f}", fontsize=energy_text_size, ha='center', va='bottom')

            dressed_state_index = product_to_dressed[(ql,rl)]
            if dressed_state_index != None:
                dressed = hilbertspace.energy_by_dressed_index(dressed_state_index)* 2 * np.pi
                ax.plot([x1, x2], [dressed, dressed], linewidth=1, color=dressed_color)
                ax.text(ql, dressed, f"{dressed:.3f}", fontsize=energy_text_size, ha='center', va='top')
            else:
                print("dressed_state_index contain None")

    for transition in flagged_transitions:
        state1, state2 = transition[0],transition[1]
        dressed_index1 = product_to_dressed[(state1[0],state1[1])]
        dressed_index2 = product_to_dressed[(state2[0],state2[1])]
        if dressed_index1!= None and dressed_index2!= None:
            energy1 = hilbertspace.energy_by_dressed_index(dressed_index1)* 2 * np.pi
            energy2 = hilbertspace.energy_by_dressed_index(dressed_index2)* 2 * np.pi
            ax.plot([state1[0], state2[0]], [energy1, energy2], linewidth=1, color=dressed_color)
            ax.text((state1[0]+ state2[0])/2, (energy1+ energy2)/2, f"{energy2-energy1:.3f}", fontsize=energy_text_size, ha='center', va='top')
        else:
            print("dressed_state_index contain None")
    plt.show()

def plot_population_and_alpha(results,idxs,product_states,tlist,dominant_freq,nlevels = 4):
    dictionary = {0: 'g', 1: 'e', 2: 'f', 3: 'h'}
    fig, axes = plt.subplots(4, 4, figsize=(9, 6))
    if type(dominant_freq) == float:
        dominant_freq = [dominant_freq]*nlevels
    for i in range(nlevels):
        for idx, res in zip(idxs, results[i].expect):
            product_state = product_states[idxs.index(idx)]
            qubit_state = dictionary[product_state[0]]
            resonator_state = product_state[1]
            axes[0][i].plot(tlist, res, label=r"$\overline{|%s\rangle}$" % (str(",".join([qubit_state,str(resonator_state)]))))
        
        alpha = results[i].expect[-2]*np.exp(-1j * 2 * np.pi * dominant_freq[i] * tlist)  
        real = alpha.real
        imag = alpha.imag
        axes[0][i].plot(tlist, results[i].expect[-1], label=r"photon number")
        axes[1][i].plot(tlist,imag , label=r"imag alpha")
        axes[2][i].plot(tlist, real, label=r"real alpha")
        axes[3][i].plot(-imag, real, label=r"imag alpha VS real alpha")

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