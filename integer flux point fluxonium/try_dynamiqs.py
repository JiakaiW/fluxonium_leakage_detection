import dynamiqs as dq
import jax.numpy as jnp

# parameters
n = 80      # Hilbert space dimension
omega = 1.0  # frequency
kappa = 0.1  # decay rate
alpha_list  = [i for i in range(10)]  # initial coherent state amplitude

# initialize operators, initial state and saving times
a = dq.destroy(n)
H = omega * dq.dag(a) @ a
jump_ops = [jnp.sqrt(kappa) * a]
psi0 = [dq.coherent(n, alpha) for alpha in alpha_list]
tsave = jnp.linspace(0, 1000, 1000)

# run simulation  
result = dq.mesolve( H,
                     jump_ops, 
                    psi0, 
                    tsave,
                    options=dq.Options(verbose=True))
print(result)

# import dynamiqs as dq
# import jax.numpy as jnp
# import jax

# # parameters
# n = 128      # Hilbert space dimension
# omega = 1.0  # frequency
# kappa = 0.1  # decay rate
# alpha = 1.0  # initial coherent state amplitude

# def population(omega, kappa, alpha):
#     """Return the oscillator population after time evolution."""
#     # initialize operators, initial state and saving times
#     a = dq.destroy(n)
#     H = omega * dq.dag(a) @ a
#     jump_ops = [jnp.sqrt(kappa) * a]
#     psi0 = dq.coherent(n, alpha)
#     tsave = jnp.linspace(0, 500, 500)

#     # run simulation
#     result = dq.mesolve(H, jump_ops, psi0, tsave,options=dq.Options(verbose=True))

#     return dq.expect(dq.number(n), result.states[-1]).real

# # compute gradient with respect to omega, kappa and alpha
# grad_population = jax.grad(population, argnums=(0, 1, 2))
# grads = grad_population(omega, kappa, alpha)
# print(f'Gradient w.r.t. omega={grads[0]:.2f}')
# print(f'Gradient w.r.t. kappa={grads[1]:.2f}')
# print(f'Gradient w.r.t. alpha={grads[2]:.2f}')