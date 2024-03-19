import matplotlib.pyplot as plt
import numpy as np

def square_pulse_with_rise_fall(t, w_d, amp, t_start=0, t_rise=0, t_square=0):
    def cos_modulation(t):
        return 2 * np.pi * amp * np.cos(w_d * 2 * np.pi * t)

    t_fall_start = t_start + t_rise + t_square  # Start of fall
    t_end = t_fall_start + t_rise  # End of the pulse

    before_pulse_start = np.less(t, t_start)
    during_rise_segment = np.logical_and(np.greater_equal(t_rise, 0), np.logical_and(np.greater_equal(t, t_start), np.less_equal(t, t_start + t_rise)))
    constant_amplitude_segment = np.logical_and(np.greater(t, t_start + t_rise), np.less_equal(t, t_fall_start))
    during_fall_segment = np.logical_and(np.greater_equal(t_rise, 0), np.logical_and(np.greater(t, t_fall_start), np.less_equal(t, t_end)))

    return np.where(before_pulse_start, 0,
                    np.where(during_rise_segment, np.sin(np.pi * (t - t_start) / (2 * t_rise)) ** 2 * cos_modulation(t),
                             np.where(constant_amplitude_segment, cos_modulation(t),
                                      np.where(during_fall_segment, np.sin(np.pi * (t_end - t) / (2 * t_rise)) ** 2 * cos_modulation(t), 0))))

# Define parameters
t_values = np.linspace(0, 10, 1000)
w_d = 1.0
amp = 1.0
t_start = 0.0
t_rise = 2.0
t_square = 3.0

# Compute the function values
y_values = square_pulse_with_rise_fall(t_values, w_d, amp, t_start, t_rise, t_square)

# Plot
plt.plot(t_values, y_values)
plt.xlabel('Time')
plt.ylabel('Function Value')
plt.title('Square Pulse with Rise and Fall')
plt.grid(True)
plt.show()
