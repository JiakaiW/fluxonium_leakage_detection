import concurrent
from itertools import product

from loky import get_reusable_executor
import numpy as np
import qutip
import scqubits
from typing import List, Union, Tuple
from utils_v2_for_gfIFQ.utils_basic_funcs import *
from utils_v2_for_gfIFQ.utils_DriveTerm import *
from utils_v2_for_gfIFQ.utils_evo import *
from utils_v2_for_gfIFQ.utils_noise import *

'''
The numerical study of single-qubit gates for gf-IFQ is composed of these tasks.

Gates to be studied: X (STIRAP), Y (STIRAP with a pi/2 phase) and H (fSTIRAP),
    Z gates will be done virtually. (https://arxiv.org/pdf/1612.00858)

Gate simulation takes these inputs: hamiltonian, with decay/dephasing rates evaluated from numerical estimation of lifetime.
    Decay from state $\ket{b}$ to $\ket{a}$ is $\sqrt{\Gamma_{\ket{b}\rightarrow \ket{a}}} \ket{a}\bra{b}$
    Pure dephasing is $diag(\sqrt{\Gamma_i}, \sqrt{\Gamma_j}, \sqrt{\Gamma_k})$
'''


class gfIFQ:
    def __init__(self,
                 EJ,
                 EC,
                 EL,
                 flux=0, truncated_dim=8) -> None:
        self.fluxonium = scqubits.Fluxonium(EJ=EJ,
                                            EC=EC,
                                            EL=EL,
                                            flux=flux, cutoff=110,
                                            truncated_dim=truncated_dim)
        self.truncated_dim = truncated_dim
        self.evals = self.fluxonium.eigenvals(evals_count=truncated_dim)
        self.phi_tabel = self.fluxonium.matrixelement_table(
            'phi_operator', evals_count=truncated_dim)
        self.n_tabel = self.fluxonium.matrixelement_table(
            'n_operator', evals_count=truncated_dim)
    def get_c_ops(self,
                  temp_in_mK,
                  loss_tangent_ref,
                  one_over_f_flux_noise_amplitude) -> None:
        # array element [i,j] means transition rate from i to j
        dielectric_T1_array = np.zeros(
            shape=(self.truncated_dim, self.truncated_dim))
        one_over_f_T1_array = np.zeros(
            shape=(self.truncated_dim, self.truncated_dim))
        T1_array = np.zeros(shape=(self.truncated_dim, self.truncated_dim))
        Tphi_array = np.zeros(shape=(self.truncated_dim,))
        EL = self.fluxonium.EL
        EC = self.fluxonium.EC
        # T1
        for i in range(self.truncated_dim):
            for j in range(self.truncated_dim):
                if i == j:
                    continue
                freq = (self.evals[i]-self.evals[j]) * 2 * np.pi
                phi_ele = self.phi_tabel[i, j]
                dielectric_T1_array[i, j] = 1 / (np.abs(phi_ele)**2 * diel_spectral_density(
                    freq, EC, temp_in_mK, loss_tangent_ref))
                one_over_f_T1_array[i, j] = 1 / (np.abs(phi_ele)**2 * one_over_f_spectral_density(
                    freq, EL, one_over_f_flux_noise_amplitude))
        T1_array = 2/(1/dielectric_T1_array + 1/one_over_f_T1_array)

        # Tphi
        for i in range(self.truncated_dim):
            Tphi_array[i] = T_phi(
                second_order_derivative=get_second_order_derivative_of_eigenenergy(
                    EJ=self.fluxonium.EJ,
                    EC=self.fluxonium.EC,
                    EL=self.fluxonium.EL,
                    i=i,
                    flux0=0),
                one_over_f_flux_noise_amplitude=one_over_f_flux_noise_amplitude
            )

        c_ops = qutip.Qobj(T1_array) + qutip.Qobj(np.diag(Tphi_array))
        return c_ops

    def get_DDP_STIRAP_drive_terms(self,
                               i,
                               j,
                               k,
                               t_stop,
                               Rabi_freq0 = 1e-1,
                               t_start = 0,
                               phi = 0
                               ):
        amp_ij = Rabi_freq0 / np.abs(self.n_tabel[i,j])
        amp_jk = Rabi_freq0 / np.abs(self.n_tabel[j,k])
        drive_terms = [
            DriveTerm( 
                driven_op=self.fluxonium.n_operator,
                pulse_shape_func=masked_optimized_STIRAP_with_modulation,
                pulse_shape_args={
                    'w_d':self.evals[j]-self.evals[i],
                    'amp': amp_ij, # Without 2pi !
                    't_stop': t_stop,
                    'stoke': True,
                    't_start': t_start,
                    'phi':phi
                  },
                ),
            DriveTerm( 
                driven_op=self.fluxonium.n_operator,
                pulse_shape_func=masked_optimized_STIRAP_with_modulation,
                pulse_shape_args={
                    'w_d':self.evals[k]-self.evals[j],
                    'amp': amp_jk, # Without 2pi !
                    't_stop': t_stop,
                    'stoke': False,
                    't_start': t_start,
                    'phi':phi
                  },
                ),
        ]
        return drive_terms

    def run_qutip_mesolve_parrallel(self,
                                    initial_states: qutip.Qobj,
                                    tlist: np.array,
                                    drive_terms: List[DriveTerm],
                                    c_ops: Union[None,
                                                 List[qutip.Qobj]] = None,
                                    e_ops: Union[None,
                                                 List[qutip.Qobj]] = None,
                                    post_processing=None,  # Currently I have no post_processing written
                                    ) -> None:

        post_processing_funcs = []
        post_processing_args = []
        results = [None] * len(initial_states)
        with get_reusable_executor(max_workers=None, context='loky') as executor:
            futures = {executor.submit(ODEsolve_and_post_process,
                                       y0=initial_states[i],
                                       tlist=tlist,
                                       static_hamiltonian=self.diag_dressed_hamiltonian,
                                       drive_terms=drive_terms,
                                       c_ops=c_ops,
                                       e_ops=e_ops,
                                       post_processing_funcs=post_processing_funcs,
                                       post_processing_args=post_processing_args,
                                       ): i for i in range(len(initial_states))}

            for future in concurrent.futures.as_completed(futures):
                original_index = futures[future]
                results[original_index] = future.result()

        return results


