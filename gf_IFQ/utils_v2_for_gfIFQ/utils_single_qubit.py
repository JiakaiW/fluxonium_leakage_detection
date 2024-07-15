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

        fluxonium1 = scqubits.Fluxonium(EJ=EJ,
                                        EC=EC,
                                        EL=EL,
                                        flux=flux, cutoff=110,
                                        truncated_dim=truncated_dim)
        pass

    def get_c_ops(self) -> None:

        pass

    def run_STIRAP(self,
                   initial_states: qutip.Qobj,
                   tlist: np.array,
                   drive_terms: List[DriveTerm],
                   c_ops: Union[None, List[qutip.Qobj]] = None,
                   e_ops: Union[None, List[qutip.Qobj]] = None,
                   fractional=False,
                   ) -> None:
        pass

    # def run_fSTIRAP(self) -> None:
    #     pass
