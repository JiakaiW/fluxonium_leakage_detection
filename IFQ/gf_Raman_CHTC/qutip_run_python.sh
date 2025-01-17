# Define the number of detuning and t_duration values
num_detuning=16
num_t_duration=4

# Get the task number from the command line argument
task_number=$1

# Calculate detuning_idx and t_duration_idx
detuning_idx=$((task_number / num_t_duration))
t_duration_idx=$((task_number % num_t_duration))

# Run the Python script with the calculated indices
python opt_gf_raman.py $detuning_idx $t_duration_idx