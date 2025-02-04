import numpy as np
import qutip
from CoupledQuantumSystems.drive import DriveTerm, sin_squared_pulse_with_modulation
from CoupledQuantumSystems.IFQ import gfIFQ
from CoupledQuantumSystems.evo import ODEsolve_and_post_process
import argparse


def main(start_idx, end_idx):
    EJ = 3
    EJoverEC = 6
    EJoverEL = 25
    EC = EJ / EJoverEC
    EL = EJ / EJoverEL
    
    def get_results(n_lvls):
        # Create quantum objects once for this n_lvls
        qbt = gfIFQ(EJ = EJ, EC = EC, EL = EL, flux = 0, truncated_dim=n_lvls)
        state_g_dressed = qutip.basis(qbt.truncated_dim, 0)
        state_0_dressed = qutip.basis(qbt.truncated_dim, 1)
        state_1_dressed = qutip.basis(qbt.truncated_dim, 2)
        e_ops = [qutip.ket2dm(state_g_dressed), qutip.ket2dm(state_0_dressed), qutip.ket2dm(state_1_dressed)]
        n_op = qutip.Qobj(qbt.fluxonium.n_operator(energy_esys=True))

        # Pre-compute parameter combinations first
        t_tot_arr = np.linspace(100, 400, 201)
        w_d_arr = np.linspace(0.003252043953428929 - 0.001, 0.003252043953428929 + 0.001, 200)
        amp_arr = np.linspace(0, 5, 200)
        
        param_combinations = []
        for t_tot in t_tot_arr:
            for w_d in w_d_arr:
                for amp in amp_arr:
                    param_combinations.append((t_tot, w_d, amp))
        param_combinations = param_combinations[start_idx:end_idx]
        param_combinations = np.array(param_combinations)

        ave_transfer_prob_list = []
        batch_size = 8

        for i in range(0, len(param_combinations), batch_size):
            batch_params = param_combinations[i:i + batch_size]
            batch_tlist = []
            batch_drive_terms = []
            batch_e_ops = []
            
            for t_tot, w_d, amp in batch_params:
                batch_tlist.append(np.linspace(0, t_tot, 11))
                batch_drive_terms.append([DriveTerm(driven_op=n_op,
                                        pulse_shape_func=sin_squared_pulse_with_modulation,
                                        pulse_id='pi',
                                        pulse_shape_args={"w_d": w_d,
                                        "amp": amp,
                                        "t_duration": t_tot,
                                        })])
                batch_e_ops.append(e_ops)

            # Replace parallel solver with single-threaded solver
            results_list = []
            for j in range(len(batch_tlist)):
                result_sublist = []
                for y0 in [state_0_dressed, state_1_dressed]:
                    result = ODEsolve_and_post_process(
                        y0=y0,
                        tlist=batch_tlist[j],
                        static_hamiltonian=qbt.diag_hamiltonian,
                        drive_terms=batch_drive_terms[j],
                        e_ops=batch_e_ops[j],
                        print_progress=False,
                    )
                    result_sublist.append(result)
                results_list.append(result_sublist)
            batch_results = results_list
            
            for results_of_the_same_evo in batch_results:
                one_minus_pop2 = abs(1 - results_of_the_same_evo[0].expect[2][-1])
                one_minus_pop1 = abs(1 - results_of_the_same_evo[1].expect[1][-1])
                ave_transfer_prob_list.append((one_minus_pop2 + one_minus_pop1) / 2)
            
            # Only clear batch objects
            del batch_results, batch_tlist, batch_drive_terms, batch_e_ops

        return {
            'parameters': param_combinations,
            'transfer_probs': np.array(ave_transfer_prob_list)
        }

    # Save results with parameter information
    output_filename = f"sweep_start_idx_{start_idx}_end_idx_{end_idx}"
    for n_lvls in [20, 23, 26]:
        results = get_results(n_lvls)
        np.savez_compressed(
            f"{output_filename}_{n_lvls}lvls.npz",
            parameters=results['parameters'],
            transfer_probs=results['transfer_probs']
        )
        del results  # Clean up results after saving

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run ef X gate sweep')
    parser.add_argument('start_idx', type=int, help='start index')
    parser.add_argument('end_idx', type=int, help='end index')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args.start_idx, args.end_idx) 