# #!/bin/bash

# # Get process number from argument
# process=$1

# # Calculate chunk size (how many parameter combinations per job)
# total_combinations=8040000  # 201 * 200 * 200
# n_jobs=1000                 # Split into 1000 jobs
# chunk_size=$((total_combinations / n_jobs))

# # Calculate start and end indices for this process
# start_idx=$((process * chunk_size))
# end_idx=$(((process + 1) * chunk_size))

# # For the last process, make sure we get all remaining combinations
# if [ "$process" -eq $((n_jobs - 1)) ]; then
#     end_idx=$total_combinations
# fi

# # Run the Python script with calculated indices
# python sin^2_sweep_condor.py $start_idx $end_idx 


#!/bin/bash

# Get process number from argument
process=$1

# Calculate chunk size (how many parameter combinations per job)
total_combinations=16000  # 201 * 200 * 200
n_jobs=1000                 # Split into 1000 jobs
chunk_size=$((total_combinations / n_jobs))

# Calculate start and end indices for this process
start_idx=$((process * chunk_size))
end_idx=$(((process + 1) * chunk_size))

# For the last process, make sure we get all remaining combinations
if [ "$process" -eq $((n_jobs - 1)) ]; then
    end_idx=$total_combinations
fi

# Run the Python script with calculated indices
python sin^2_sweep_condor.py $start_idx $end_idx 