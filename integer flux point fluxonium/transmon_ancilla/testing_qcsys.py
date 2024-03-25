import jax
import jax.numpy as jnp
import qcsys as qs
import scqubits
import numpy as np
import qutip

jax.config.update('jax_platform_name', 'cpu')
# jax.config.update('jax_enable_x64', True)
def transform_op_into_dressed_basis_jax(op_matrix, dressed_evecs):
    """
    Transform an operator into the dressed basis using JAX.

    Parameters:
    - op_matrix: A 2D JAX array representing the operator's matrix.
    - dressed_evecs: A 2D JAX array representing the dressed eigenvectors.

    Returns:
    - A 2D JAX array representing the transformed operator.
    """
    S = jnp.conj(dressed_evecs)
    data = jnp.dot(S, jnp.dot(op_matrix, S.T.conj()))
    return data

import jax
import jax.numpy as jnp
import qcsys as qs
import scqubits
import numpy as np
import qutip

jax.config.update('jax_platform_name', 'cpu')
# jax.config.update('jax_enable_x64', True)
def transform_op_into_dressed_basis_jax(op_matrix, dressed_evecs):
    """
    Transform an operator into the dressed basis using JAX.

    Parameters:
    - op_matrix: A 2D JAX array representing the operator's matrix.
    - dressed_evecs: A 2D JAX array representing the dressed eigenvectors.

    Returns:
    - A 2D JAX array representing the transformed operator.
    """
    S = jnp.conj(dressed_evecs)
    data = jnp.dot(S, jnp.dot(op_matrix, S.T.conj()))
    return data



qsf = qs.Fluxonium.create(
    25,
    {"Ej": 2.7, "Ec": 0.6, "El": 0.13, "phi_ext": 0.0},
    N_pre_diag=100,
    use_linear = False
)

scf = scqubits.Fluxonium(EJ=2.7,
                        EC=0.6,
                        EL=0.13,
                        flux=0,cutoff=100,
                        truncated_dim=25)


np.array_equal(np.array(scf.n_operator())[:25,:25] , np.array(qsf.ops['n'].data,dtype = 'complex128')), \
    np.allclose(np.array(qutip.Qobj(scf.hamiltonian()).tidyup()), np.array(qsf.get_H_full().data,dtype = 'complex128'),atol=1e-08)


qst_sc = qs.SingleChargeTransmon.create(
    N = 4,
    params = {"Ej": 40, "Ec": 0.5,"ng":0.0},
    N_max_charge=30
)

sct = scqubits.Transmon(
    EJ=40,
    EC=0.5,
    ng=0.0,
    ncut=30,
    truncated_dim = 4
    )

np.array_equal(np.array(sct.n_operator()),np.array(qst_sc.linear_ops['n'].data)), \
    np.array_equal(np.array(sct.hamiltonian()),np.array(qst_sc.get_H_full().data))