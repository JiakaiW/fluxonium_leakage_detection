import os
import sys
import time
import signal
import logging
import traceback
import functools
import pickle
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
import queue

# Set up logging first
log_filename = "optimization_debug.log"
with open(log_filename, 'w') as f:
    f.write(f"=== New optimization run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_debug(msg):
    """Helper function to log debug messages to file"""
    logging.debug(msg)

# Configure JAX environment variables before importing JAX
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# Set up JAX compilation cache with absolute path
cache_dir = os.path.abspath('./.jax_cache')
os.environ['JAX_COMPILATION_CACHE_DIR'] = cache_dir
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'  # Limit memory usage
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Disable memory preallocation

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
print(f"Using JAX cache directory: {cache_dir}")

# Now import JAX and configure it
import jax
print(f"JAX version: {jax.__version__}")
jax.config.update('jax_enable_compilation_cache', True)
print("JAX compilation cache enabled")

import jax.numpy as jnp
import optax

# Only after JAX is configured, import other dependencies
import numpy as np
import qutip
import warnings
warnings.filterwarnings("ignore")

# Import dynamiqs after JAX is configured
import dynamiqs as dq
dq.set_device('cpu')

# Rich imports for display
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich import box

# Import quantum system modules last
from CoupledQuantumSystems.drive import DriveTerm
from CoupledQuantumSystems.IFQ import gfIFQ

def precompile_jax_functions():
    """Pre-compile JAX functions with timing information"""
    print(f"[Process {os.getpid()}] Pre-compiling JAX functions...")
    start_time = time.time()
    
    # Dummy values for pre-compilation
    print(f"[Process {os.getpid()}] Setting up dummy values...")
    dummy_t = 0.0
    dummy_params = jnp.array([0.1, 0.1, 0.1])
    dummy_t_tot = 50.0
    dummy_args = {"w_d": 0.1, "amp": 0.1, "t_square": 10.0, "t_rise": 1.0}
    
    print(f"[Process {os.getpid()}] Starting Hamiltonian compilation (cache_dir={cache_dir})...")
    t0 = time.time()
    _ = _hamiltonian_fn(dummy_t, dummy_args["w_d"], dummy_args["amp"], dummy_args["t_square"], dummy_args["t_rise"])
    print(f"[Process {os.getpid()}] Hamiltonian compilation completed in {time.time()-t0:.2f}s")
    
    print(f"[Process {os.getpid()}] Starting objective function compilation...")
    t0 = time.time()
    _ = _objective_wrapper(dummy_params, dummy_t_tot)
    print(f"[Process {os.getpid()}] Objective compilation completed in {time.time()-t0:.2f}s")
    
    print(f"[Process {os.getpid()}] Starting gradient function compilation (this may take a while)...")
    t0 = time.time()
    _ = _grad_objective(dummy_params, dummy_t_tot)
    print(f"[Process {os.getpid()}] Gradient compilation completed in {time.time()-t0:.2f}s")
    
    print(f"[Process {os.getpid()}] All compilations complete. Total time: {time.time()-start_time:.2f}s")

def init_worker():
    """Initialize each worker process with minimal overhead"""
    print(f"\n[Process {os.getpid()}] Worker process starting initialization...")
    
    print(f"[Process {os.getpid()}] Setting up logging...")
    # Disable JAX logging in worker processes
    logging.getLogger('jax').setLevel(logging.WARNING)
    # Disable all other logging except errors
    logging.getLogger('dynamiqs').setLevel(logging.ERROR)
    logging.getLogger('qutip').setLevel(logging.ERROR)
    
    print(f"[Process {os.getpid()}] Verifying JAX cache directory: {cache_dir}")
    if not os.path.exists(cache_dir):
        print(f"[Process {os.getpid()}] Warning: Cache directory not found!")
    
    print(f"[Process {os.getpid()}] Starting JAX pre-compilation (should use cache)...")
    # Pre-compile in worker
    precompile_jax_functions()
    print(f"[Process {os.getpid()}] Worker initialization complete")

def square_pulse_with_rise_fall_jnp(t,
                                args = {}):
    
    w_d = args['w_d']
    amp = args['amp']
    t_start = args.get('t_start', 0)  # Default start time is 0
    t_rise = args.get('t_rise', 1e-10)  # Default rise time is 0 for no rise
    t_square = args.get('t_square', 0)  # Duration of constant amplitude

    def cos_modulation():
        return 2 * jnp.pi * amp * jnp.cos(w_d * 2 * jnp.pi * t)
    
    t_fall_start = t_start + t_rise + t_square  # Start of fall
    t_end = t_fall_start + t_rise  # End of the pulse

    before_pulse_start = jnp.less(t, t_start)
    during_rise_segment = jnp.logical_and(jnp.greater(t_rise, 0), jnp.logical_and(jnp.greater_equal(t, t_start), jnp.less_equal(t, t_start + t_rise)))
    constant_amplitude_segment = jnp.logical_and(jnp.greater(t, t_start + t_rise), jnp.less_equal(t, t_fall_start))
    during_fall_segment = jnp.logical_and(jnp.greater(t_rise, 0), jnp.logical_and(jnp.greater(t, t_fall_start), jnp.less_equal(t, t_end)))

    return jnp.where(before_pulse_start, 0,
                    jnp.where(during_rise_segment, jnp.sin(jnp.pi * (t - t_start) / (2 * t_rise)) ** 2 * cos_modulation(),
                            jnp.where(constant_amplitude_segment, cos_modulation(),
                                        jnp.where(during_fall_segment, jnp.sin(jnp.pi * (t_end - t) / (2 * t_rise)) ** 2 * cos_modulation(), 0))))

solver = dq.solver.Dopri8(
                    rtol= 1e-06,
                    atol= 1e-06,
                    safety_factor= 0.6,
                    min_factor= 0.1,
                    max_factor = 4.0,
                    max_steps = int(1e4*1000),
                )


# -------------------------------
# Define your parameters and objects
# -------------------------------
EJ = 3
EJoverEC = 6
EJoverEL = 25
EC = EJ / EJoverEC
EL = EJ / EJoverEL

qbt = gfIFQ(EJ=EJ, EC=EC, EL=EL, flux=0, truncated_dim=20)

driven_op=qutip.Qobj(qbt.fluxonium.n_operator(energy_esys=True))

e_ops = [
    qutip.basis(qbt.truncated_dim, i) * qutip.basis(qbt.truncated_dim, i).dag()
    for i in range(10)
]

element = np.abs(
    qbt.fluxonium.matrixelement_table("n_operator", evals_count=3)[1, 2]
)
freq = (qbt.fluxonium.eigenvals()[2] - qbt.fluxonium.eigenvals()[1]) * 2 * np.pi
init_wd = qbt.fluxonium.eigenvals()[2] - qbt.fluxonium.eigenvals()[1]

# -------------------------------
# Define static Hamiltonian function
# -------------------------------
def make_hamiltonian(t, args=None):
    """Static Hamiltonian function that can be reused"""
    if args is None:
        args = {}
    
    _H = qbt.diag_hamiltonian.full()
    if not args:  # If no args provided, return bare Hamiltonian
        return _H
        
    pulse = square_pulse_with_rise_fall_jnp(t, args)
    _H += driven_op.full() * pulse
    return _H

# Create the time-dependent Hamiltonian
log_debug("Creating static Hamiltonian...")
static_H = dq.timecallable(make_hamiltonian)
log_debug("Static Hamiltonian created successfully")

# -------------------------------
# Define static functions for JAX transformations
# -------------------------------
@jax.jit
def _hamiltonian_fn(t, w_d, amp, t_square, t_rise):
    """Static Hamiltonian function that avoids recreating args dictionary"""
    args = {
        "w_d": w_d,
        "amp": amp,
        "t_square": t_square,
        "t_rise": t_rise,
    }
    return make_hamiltonian(t, args)

@jax.jit
def _objective_wrapper(params, t_tot):
    """Static wrapper for the objective function"""
    amp, w_d, ramp = params
    
    # Create time points
    tlist = jnp.linspace(0, t_tot * (1+ramp), 100)
    
    # Create initial states (these are static and don't need JAX)
    initial_states = [
        qutip.basis(qbt.truncated_dim, 1),
        qutip.basis(qbt.truncated_dim, 2),
    ]
    
    # Create Hamiltonian with current parameters
    H = dq.timecallable(
        lambda t: _hamiltonian_fn(t, w_d, amp, t_tot * (1-ramp), t_tot * ramp)
    )
    
    # Solve Schrödinger equation
    results = dq.sesolve(
        H=H,
        psi0=initial_states,
        tsave=tlist,
        exp_ops=e_ops,
        solver=solver,
        options=dq.Options(progress_meter=None),
    )
    
    # Calculate objective value
    one_minus_pop2 = abs((1 - results.expects[0][2][-1]).real)
    one_minus_pop1 = abs((1 - results.expects[1][1][-1]).real)
    return one_minus_pop2 + one_minus_pop1

# -------------------------------
# Create static gradient function
# -------------------------------
@functools.partial(jax.jit, static_argnums=(1,))
def _grad_objective(params, t_tot):
    """Static function for gradient computation with t_tot as static argument"""
    return _objective_wrapper(params, t_tot)

def make_grad_fn_static(t_tot):
    """Create a gradient function for a specific t_tot without using lambda"""
    return jax.grad(functools.partial(_grad_objective, t_tot=t_tot))

def optimize_for_t_tot(args):
    t_tot, progress_queue = args
    
    try:
        # Send initial status IMMEDIATELY
        progress_queue.put_nowait(('status', (t_tot, 'Started')))
        
        start_time = time.time()
        print(f"\n=== Starting optimization for t_tot={t_tot} at {datetime.now().strftime('%H:%M:%S')} ===")
        log_debug(f"Process for t_tot={t_tot} starting...")
        
        # Load initial parameters first
        t0 = time.time()
        print(f"[{t_tot}] Loading initial parameters...")
        possible_filenames = [
            f"dq_optimized_results_{t_tot}.pkl",
            f"nevergrad_optimized_results_{int(t_tot)}.pkl",
            f"nevergrad_optimized_results_{int(t_tot)}.0.pkl"
        ]
        
        for filename in possible_filenames:
            if os.path.exists(filename):
                file_name = filename
                break
        else:
            raise FileNotFoundError(f"No initial parameters found for t_tot={t_tot}")
        
        results_dict = pickle.load(open(file_name, "rb"))
        params = jnp.array([results_dict["best_amp"], results_dict["best_w_d"], results_dict["best_ramp"]])
        print(f"[{t_tot}] Loaded params in {time.time()-t0:.2f}s: amp={params[0]:.6f}, w_d={params[1]:.6f}, ramp={params[2]:.6f}")
        
        # Try direct function evaluation first
        t0 = time.time()
        print(f"[{t_tot}] Attempting direct function evaluation...")
        try:
            direct_val = _objective_wrapper(params, t_tot)
            print(f"[{t_tot}] Direct evaluation completed in {time.time()-t0:.2f}s, result: {direct_val:.6f}")
        except Exception as e:
            print(f"[{t_tot}] Error in direct evaluation: {str(e)}")
            raise e
        
        # Create gradient function using the static approach
        t0 = time.time()
        print(f"[{t_tot}] Creating gradient function...")
        grad_fn = make_grad_fn_static(t_tot)
        
        # Now try gradient computation
        print(f"[{t_tot}] Testing gradient computation...")
        try:
            initial_grads = grad_fn(params)
            print(f"[{t_tot}] Gradient computation completed in {time.time()-t0:.2f}s")
            print(f"[{t_tot}] Initial gradients: {initial_grads}")
        except Exception as e:
            print(f"[{t_tot}] Error in gradient computation: {str(e)}")
            raise e
        
        print(f"[{t_tot}] Total initialization time: {time.time()-start_time:.2f}s")
        
        # If we got here, both function and gradient work, set up the optimizer
        print("Setting up optimization...")
        log_debug(f"Process t_tot={t_tot}: Setting up optimization...")
        optimizer = optax.nadam(learning_rate=jnp.array([1e-2, 1e-4, 1e-3]))
        opt_state = optimizer.init(params)
        
        best_val = float('inf')
        best_params = None
        no_improvement_count = 0
        num_steps = 1500
        current_val = direct_val  # Initialize with the direct evaluation result
        
        # Start optimization loop
        print(f"Starting optimization loop (max {num_steps} steps)...")
        log_debug(f"Process t_tot={t_tot}: Starting optimization loop...")
        for step in range(num_steps):
            try:
                if step % 20 == 0:
                    print(f"Step {step}: Current value = {current_val:.6f}")
                    log_debug(f"Process t_tot={t_tot}: At step {step}")
                
                # Evaluate function and gradients
                current_val = _objective_wrapper(params, t_tot)
                grads = grad_fn(params)
                
                # Update best values
                if current_val < best_val:
                    best_val = current_val
                    best_params = jnp.array(params)
                    no_improvement_count = 0
                    print(f"New best value: {best_val:.6f}")
                else:
                    no_improvement_count += 1
                
                # Send updates every step
                progress_queue.put_nowait(('update', (t_tot, step, float(current_val), params)))
                
                # Early stopping if no improvement for 20 steps
                if no_improvement_count > 20:
                    print(f"Early stopping triggered at step {step}")
                    log_debug(f"Process t_tot={t_tot}: Early stopping at step {step}")
                    progress_queue.put_nowait(('status', (t_tot, f'Early stop at step {step}')))
                    break
                
                # Update parameters
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                
            except Exception as e:
                print(f"Error in step {step}: {str(e)}")
                log_debug(f"Process t_tot={t_tot}: Error in step {step}: {str(e)}\n{traceback.format_exc()}")
                progress_queue.put_nowait(('status', (t_tot, f'Step error: {str(e)}')))
                continue
        
        print(f"Optimization loop complete for t_tot={t_tot}")
        log_debug(f"Process t_tot={t_tot}: Optimization loop complete")
        
        # Use the best parameters found
        if best_params is not None:
            params = best_params
            current_val = best_val
            print(f"Final best value: {best_val:.6f}")
        
        # Final update
        progress_queue.put_nowait(('update', (t_tot, num_steps-1, current_val, params)))
        progress_queue.put_nowait(('status', (t_tot, 'Completed')))
        
        # Store results
        results_dict = {
            "best_cost": current_val,
            "best_amp": params[0],
            "best_w_d": params[1],
            "best_ramp": params[2],
        }
        
        print(f"Saving results to dq_optimized_results_{t_tot}.pkl")
        with open(f"dq_optimized_results_{t_tot}.pkl", "wb") as f:
            pickle.dump(results_dict, f)
        
        return t_tot, results_dict
    
    except Exception as e:
        print(f"Fatal error in optimization for t_tot={t_tot}: {str(e)}")
        # Send error status
        progress_queue.put_nowait(('status', (t_tot, f'Error: {str(e)}')))
        raise e

class OptimizationProgress:
    def __init__(self, t_tot_list):
        self.console = Console()
        self.t_tot_list = t_tot_list
        self.current_results = {t_tot: {
            "step": 0,
            "val": float('inf'),
            "params": None,
            "best_val": float('inf'),
            "best_params": None,
            "status": "Pending"
        } for t_tot in t_tot_list}
        
    def create_table(self):
        table = Table(title="Optimization Progress", box=box.ROUNDED)
        table.add_column("t_tot", justify="right", style="cyan")
        table.add_column("Status", justify="center", style="bold")
        table.add_column("Step", justify="right", style="green")
        table.add_column("Current Cost", justify="right", style="magenta")
        table.add_column("Current Params", justify="left", style="yellow")
        table.add_column("Best Cost", justify="right", style="bright_green")
        table.add_column("Best Params", justify="left", style="bright_yellow")
        
        for t_tot in sorted(self.t_tot_list):
            result = self.current_results[t_tot]
            current_params_str = f"[dim]{str(result['params'])}[/dim]" if result['params'] is not None else "-"
            best_params_str = f"[dim]{str(result['best_params'])}[/dim]" if result['best_params'] is not None else "-"
            status_style = {
                "Pending": "white",
                "Started": "yellow",
                "Completed": "green"
            }.get(result['status'], "red")
            
            table.add_row(
                f"{t_tot:.1f}",
                f"[{status_style}]{result['status']}[/{status_style}]",
                str(result['step']),
                f"{result['val']:.6f}" if result['val'] != float('inf') else "-",
                current_params_str,
                f"{result['best_val']:.6f}" if result['best_val'] != float('inf') else "-",
                best_params_str
            )
        return table
    
    def update_progress(self, t_tot, step, val, params):
        self.current_results[t_tot].update({
            "step": step,
            "val": val,
            "params": params
        })
        # Update best values if current is better
        if val < self.current_results[t_tot]["best_val"]:
            self.current_results[t_tot].update({
                "best_val": val,
                "best_params": params
            })
    
    def update_status(self, t_tot, status):
        self.current_results[t_tot]['status'] = status

# Pre-compile functions at module level
def precompile_jax_functions():
    """Pre-compile JAX functions with timing information"""
    print(f"[Process {os.getpid()}] Pre-compiling JAX functions...")
    start_time = time.time()
    
    # Dummy values for pre-compilation
    print(f"[Process {os.getpid()}] Setting up dummy values...")
    dummy_t = 0.0
    dummy_params = jnp.array([0.1, 0.1, 0.1])
    dummy_t_tot = 50.0
    dummy_args = {"w_d": 0.1, "amp": 0.1, "t_square": 10.0, "t_rise": 1.0}
    
    print(f"[Process {os.getpid()}] Starting Hamiltonian compilation (cache_dir={cache_dir})...")
    t0 = time.time()
    _ = _hamiltonian_fn(dummy_t, dummy_args["w_d"], dummy_args["amp"], dummy_args["t_square"], dummy_args["t_rise"])
    print(f"[Process {os.getpid()}] Hamiltonian compilation completed in {time.time()-t0:.2f}s")
    
    print(f"[Process {os.getpid()}] Starting objective function compilation...")
    t0 = time.time()
    _ = _objective_wrapper(dummy_params, dummy_t_tot)
    print(f"[Process {os.getpid()}] Objective compilation completed in {time.time()-t0:.2f}s")
    
    print(f"[Process {os.getpid()}] Starting gradient function compilation (this may take a while)...")
    t0 = time.time()
    _ = _grad_objective(dummy_params, dummy_t_tot)
    print(f"[Process {os.getpid()}] Gradient compilation completed in {time.time()-t0:.2f}s")
    
    print(f"[Process {os.getpid()}] All compilations complete. Total time: {time.time()-start_time:.2f}s")

def init_worker():
    """Initialize each worker process with minimal overhead"""
    print(f"\n[Process {os.getpid()}] Worker process starting initialization...")
    
    print(f"[Process {os.getpid()}] Setting up logging...")
    # Disable JAX logging in worker processes
    logging.getLogger('jax').setLevel(logging.WARNING)
    # Disable all other logging except errors
    logging.getLogger('dynamiqs').setLevel(logging.ERROR)
    logging.getLogger('qutip').setLevel(logging.ERROR)
    
    print(f"[Process {os.getpid()}] Verifying JAX cache directory: {cache_dir}")
    if not os.path.exists(cache_dir):
        print(f"[Process {os.getpid()}] Warning: Cache directory not found!")
    
    print(f"[Process {os.getpid()}] Starting JAX pre-compilation (should use cache)...")
    # Pre-compile in worker
    precompile_jax_functions()
    print(f"[Process {os.getpid()}] Worker initialization complete")

if __name__ == '__main__':
    # Force spawn method for cleaner process creation
    print("Setting up multiprocessing spawn method...")
    mp.set_start_method('spawn')
    
    # Pre-compile in main process first
    print("\nPre-compiling in main process...")
    precompile_jax_functions()
    
    # NOTE: Below is a single optimization test that can be uncommented for debugging
    # It runs one optimization with detailed output to verify the optimization process
    # works correctly before running the parallel version
    """
    # Test single optimization first
    print("Testing single optimization...")
    t_tot = 50.0  # Test with first value
    
    # Create a simple queue for testing
    manager = Manager()
    test_queue = manager.Queue()
    
    try:
        # Run optimization directly
        print(f"Starting optimization for t_tot={t_tot}")
        result = optimize_for_t_tot((t_tot, test_queue))
        print(f"Optimization complete. Result: {result}")
        
        # Print all queue messages
        print("\nQueue messages:")
        while True:
            try:
                msg_type, msg_data = test_queue.get_nowait()
                print(f"{msg_type}: {msg_data}")
            except:
                break
                
    except Exception as e:
        print(f"Error in optimization: {e}")
        print(traceback.format_exc())
    """
    
    # Get the path to the current Python interpreter
    current_python = sys.executable
    # Set the Python executable for multiprocessing
    print(f"\nConfiguring multiprocessing with Python: {current_python}")
    mp.set_executable(current_python)
    
    # Initialize signal handler for graceful shutdown
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    t_tot_list = np.linspace(50, 250, 21)
    results = []
    pool = None
    
    try:
        # Initialize progress tracker
        progress_tracker = OptimizationProgress(t_tot_list)
        
        # Create a process-safe queue for progress updates
        manager = Manager()
        progress_queue = manager.Queue(maxsize=100)
        
        # Use specified number of processes
        num_processes = min(18, len(t_tot_list))
        print(f"\nStarting optimization with {num_processes} parallel processes...")
        print("Creating process pool and initializing workers (this may take a while)...")
        
        # Create pool arguments with progress queue
        pool_args = [(t_tot, progress_queue) for t_tot in t_tot_list]
        
        # Display live progress table
        with Live(progress_tracker.create_table(), refresh_per_second=2) as live:
            # Start the pool with our initialization function
            print("\nStarting pool creation...")
            t0 = time.time()
            pool = Pool(processes=num_processes, initializer=init_worker)
            print(f"Pool creation took {time.time()-t0:.2f}s")
            
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            # Start the processes
            print("\nStarting map_async...")
            t0 = time.time()
            result_objects = pool.map_async(optimize_for_t_tot, pool_args)
            print(f"map_async call took {time.time()-t0:.2f}s")
            
            # Monitor progress queue
            print("\nStarting progress monitoring...")
            t0 = time.time()
            last_print = time.time()
            waiting_for_first_message = True
            
            while not result_objects.ready():
                try:
                    current_time = time.time()
                    # Print waiting message every 5 seconds if we haven't received any messages
                    if waiting_for_first_message and (current_time - last_print) > 5:
                        print(f"Waiting for first worker to complete initialization... ({int(current_time - t0)}s elapsed)")
                        last_print = current_time
                    
                    # Process all available updates
                    while True:
                        try:
                            msg_type, msg_data = progress_queue.get_nowait()
                            waiting_for_first_message = False  # We got a message
                            if msg_type == 'update':
                                t_tot, step, val, params = msg_data
                                progress_tracker.update_progress(t_tot, step, val, params)
                            elif msg_type == 'status':
                                t_tot, status = msg_data
                                progress_tracker.update_status(t_tot, status)
                                print(f"Status update t_tot={t_tot}: {status} ({int(time.time() - t0)}s elapsed)")
                        except (queue.Empty, mp.queues.Empty):
                            break
                        except Exception as e:
                            print(f"Error processing message: {str(e)}")
                            continue
                    
                    # Update the live display
                    live.update(progress_tracker.create_table())
                except KeyboardInterrupt:
                    print("\nReceived keyboard interrupt. Cleaning up...")
                    if pool:
                        pool.terminate()
                        pool.join()
                    sys.exit(1)
                except Exception as e:
                    print(f"Error updating display: {str(e)}")
                    continue
                
                time.sleep(0.1)
            
            # Get all results
            try:
                results = result_objects.get(timeout=1)
            except Exception as e:
                print(f"Error getting results: {e}")
                raise
            
            # Print final results in a nice table
            final_table = Table(title="Final Optimization Results", box=box.DOUBLE)
            final_table.add_column("t_tot", justify="right", style="cyan")
            final_table.add_column("Best Cost", justify="right", style="green")
            final_table.add_column("Best Parameters", justify="left", style="yellow")
            
            for t_tot, result_dict in sorted(results):
                final_table.add_row(
                    f"{t_tot:.1f}",
                    f"{result_dict['best_cost']:.6f}",
                    f"amp={result_dict['best_amp']:.6f}\nw_d={result_dict['best_w_d']:.6f}\nramp={result_dict['best_ramp']:.6f}"
                )
            
            Console().print("\n")
            Console().print(Panel.fit("Optimization Complete!", style="bold green"))
            Console().print(final_table)
            
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Cleaning up...")
        if pool:
            pool.terminate()
            pool.join()
        sys.exit(1)
    finally:
        # Cleanup
        if pool:
            pool.close()
            pool.join()
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint_handler)