import dynamiqs as dq
import jax.numpy as jnp
import numpy as np

def square_pulse_with_rise_fall(t,
                                w_d,
                                amp,
                                t_start = 0,
                                t_rise = 0,
                                t_square = 0):
    def cos_modulation():
        return 2 * np.pi * amp * jnp.cos(w_d * 2 * np.pi * t)
    
    t_fall_start = t_start + t_rise + t_square  # Start of fall
    t_end = t_fall_start + t_rise  # End of the pulse

    before_pulse_start = jnp.less(t, t_start)
    during_rise_segment = jnp.logical_and(jnp.greater_equal(t_rise, 0), jnp.logical_and(jnp.greater_equal(t, t_start), jnp.less_equal(t, t_start + t_rise)))
    constant_amplitude_segment = jnp.logical_and(jnp.greater(t, t_start + t_rise), jnp.less_equal(t, t_fall_start))
    during_fall_segment = jnp.logical_and(jnp.greater_equal(t_rise, 0), jnp.logical_and(jnp.greater(t, t_fall_start), jnp.less_equal(t, t_end)))

    return jnp.where(before_pulse_start, 0,
                     jnp.where(during_rise_segment, jnp.sin(jnp.pi * (t - t_start) / (2 * t_rise)) ** 2 * cos_modulation(),
                               jnp.where(constant_amplitude_segment, cos_modulation(),
                                         jnp.where(during_fall_segment, jnp.sin(jnp.pi * (t_end - t) / (2 * t_rise)) ** 2 * cos_modulation(), 0))))

n = 5      # Hilbert space dimension
omega = 1.0  # frequency
amp = 5e-3
kappa = 0.1  # decay rate
alpha_list  = [i for i in range(10)]  # initial coherent state amplitude
w_d = omega
# initialize operators, initial state and saving times
a = dq.destroy(n)

def _H(t, w_d,amp ):
   return omega * dq.dag(a) @ a + (dq.dag(a) + a) * square_pulse_with_rise_fall(t,w_d,amp,t_square = 100)

H =  dq.timecallable(_H, args=(w_d,amp))


jump_ops = [jnp.sqrt(kappa) * a]
psi0 = [dq.coherent(n, alpha) for alpha in alpha_list]
tsave = jnp.linspace(0, 5, 5)

# run simulation  
result = dq.mesolve( H,
                     jump_ops, 
                    psi0, 
                    tsave,)
print(result)
