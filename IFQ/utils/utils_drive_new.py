############################################################################
#
#
# Ancilliary functions about pulse shaping and time dynamics
#
#
############################################################################
from dataclasses import dataclass, field
from typing import  Callable, Dict
import numpy as np
import qutip

@dataclass
class DriveTerm:
    '''
    This class provides a wrapper around pulse_shape_func since qutip doesn't accept duplicate keys in args.
    '''
    driven_op: qutip.Qobj
    pulse_shape_func: Callable
    pulse_id: str
    pulse_shape_args_without_id: Dict[str, float]
    pulse_shape_func_with_id: Callable = field(init=False)
    pulse_shape_args_with_id: Dict[str, float] = field(init=False)
    
    def __post_init__(self):
        self.pulse_shape_func_with_id = self.generate_pulse_shape_func_with_id()
        self.pulse_shape_args_with_id = self.modify_args_with_id(self.pulse_shape_args_without_id)

    def modify_args_with_id(self, pulse_shape_args: Dict[str, float]) -> Dict[str, float]:
        return {f"{key}{self.pulse_id}": value for key, value in pulse_shape_args.items()}

    def generate_pulse_shape_func_with_id(self) -> Callable:
        # Create a wrapper to handle the modified args
        def wrapper(t, args={}):
            try:
                unmodified_args = {key[:-len(self.pulse_id)]: value for key, value in args.items() if key.endswith(self.pulse_id)}
                return self._pulse_shape_func(t, unmodified_args)
            except KeyError as e:
                raise KeyError(f"Missing argument key for pulse_id {self.pulse_id}: {e}")
            except Exception as e:
                raise ValueError(f"Error processing pulse function for pulse_id {self.pulse_id}: {e}")
        return wrapper

    def get_driven_op(self) -> qutip.Qobj:
        return self.driven_op

    def get_pulse_shape_func_with_id(self) -> Callable:
        return self.pulse_shape_func_with_id

    def get_pulse_shape_args_with_id(self) -> Dict[str, float]:
        return self.pulse_shape_args_with_id

def square_pulse_with_rise_fall(t,
                                args = {}):
    
    w_d = args['w_d']
    amp = args['amp']
    t_start = args.get('t_start', 0)  # Default start time is 0
    t_rise = args.get('t_rise', 0)  # Default rise time is 0 for no rise
    t_square = args.get('t_square', 0)  # Duration of constant amplitude

    def cos_modulation():
        return 2 * np.pi * amp * np.cos(w_d * 2 * np.pi * t)
    
    t_fall_start = t_start + t_rise + t_square  # Start of fall
    t_end = t_fall_start + t_rise  # End of the pulse
    
    if t < t_start:
        return 0
    elif t_start <= t <= t_start + t_rise:
        return np.sin(np.pi * (t - t_start) / (2 * t_rise)) ** 2 * cos_modulation()
    elif t_start + t_rise < t <= t_fall_start:
        return cos_modulation()
    elif t_fall_start < t <= t_end:
        return np.sin(np.pi * (t_end - t) / (2 * t_rise)) ** 2 * cos_modulation()
    else:
        return 0
    
