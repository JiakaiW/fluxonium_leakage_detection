# Get process number from argument
process=$1

# Calculate indices
n_amp=6
n_wd=5
n_t_tot=29

# Integer division and modulo to get indices
t_tot_idx=$((process % n_t_tot))
temp=$((process / n_t_tot))
wd_regime_idx=$((temp % n_wd))
amp_regime_idx=$((temp / n_wd))

# Run the Python script with calculated indices
python opt_ef_X.py $t_tot_idx $amp_regime_idx $wd_regime_idx