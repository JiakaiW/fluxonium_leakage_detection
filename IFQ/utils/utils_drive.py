############################################################################
#
#
# Ancilliary functions about pulse shaping and time dynamics
#
#
############################################################################
from dataclasses import dataclass
from typing import  Callable, Dict
import numpy as np
import qutip

@dataclass
class DriveTerm:
    driven_op: qutip.Qobj
    pulse_shape_func: Callable
    pulse_shape_args: Dict

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
    

def second_square_pulse_with_rise_fall(t,
                                args = {}):
    
    w_d = args['w_d1']
    amp = args['amp1']
    t_start = args.get('t_start1', 0)  # Default start time is 0
    t_rise = args.get('t_rise1', 0)  # Default rise time is 0 for no rise
    t_square = args.get('t_square1', 0)  # Duration of constant amplitude

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
    
def third_square_pulse_with_rise_fall(t,
                                args = {}):
    
    w_d = args['w_d2']
    amp = args['amp2']
    t_start = args.get('t_start2', 0)  # Default start time is 0
    t_rise = args.get('t_rise2', 0)  # Default rise time is 0 for no rise
    t_square = args.get('t_square2', 0)  # Duration of constant amplitude

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
    
    
def forth_square_pulse_with_rise_fall(t,
                                args = {}):
    
    w_d = args['w_d3']
    amp = args['amp3']
    t_start = args.get('t_start3', 0)  # Default start time is 0
    t_rise = args.get('t_rise3', 0)  # Default rise time is 0 for no rise
    t_square = args.get('t_square3', 0)  # Duration of constant amplitude

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