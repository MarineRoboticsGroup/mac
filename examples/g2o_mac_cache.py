import sys
import random
import numpy as np
import networkx as nx
from timeit import default_timer as timer
from pose_graph_utils import read_g2o_file, plot_poses, rpm_to_mac, RelativePoseMeasurement

# MAC requirements
from mac import MAC
from mac.baseline import NaiveGreedy
from mac.utils import split_edges, Edge, round_madow

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 16})

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [.g2o file] [optional: --run-greedy]")
        sys.exit()
        pass

    run_greedy = False
    if len(sys.argv) > 2:
        if sys.argv[2] == "--run-greedy":
            run_greedy = True
            pass
        else:
            print(f"Unknown argument: {sys.argv[2]}")
            print(f"Usage: {sys.argv[0]} [.g2o file] [optional: --run-greedy]")
            sys.exit()
            pass
        pass

    dataset_name = sys.argv[1].split('/')[-1].split('.')[0]
    print(f"Loading dataset: {dataset_name}")

    # Load a g2o file
    print("Reading g2o file")
    start = timer()
    measurements, num_poses = read_g2o_file(sys.argv[1])
    end = timer()
    print("Success! elapsed time: ", (end - start))

    # Split measurements into odom and loop closures
    odom_measurements, lc_measurements = split_edges(measurements)

    # Convert measurements to MAC edge format
    odom_edges = rpm_to_mac(odom_measurements)
    lc_edges = rpm_to_mac(lc_measurements)

    # Print dataset stats
    print(f"Loaded {len(measurements)} total measurements with: ")
    print(f"\t {len(odom_measurements)} base (odometry) measurements and")
    print(f"\t {len(lc_measurements)} candidate (loop closure) measurements")

    # Make a naive solver for use as an initialization
    naive = NaiveGreedy(lc_edges)

    # Make a MAC Solver
    mac = MAC(odom_edges, lc_edges, num_poses, use_cache=False)
    mac_cache = MAC(odom_edges, lc_edges, num_poses, fiedler_method="tracemin_cholesky", use_cache=True)

    #############################
    # Running the tests!
    #############################

    # Test between 100% and 0% loop closures
    # NOTE: If running greedy, these must be in increasing order!
    percent_lc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Containers for MAC results
    results = []
    unrounded_results = []
    upper_bounds = []
    times = []

    cache_results = []
    cache_unrounded_results = []
    cache_upper_bounds = []
    cache_times = []

    for pct_lc in percent_lc:
        num_lc = int(pct_lc * len(lc_measurements))
        print("Num LC to accept: ", num_lc)

        # Compute a solution using the naive method. This serves both as a
        # baseline and as a sparse initializer for our method.
        naive_result = naive.subset(num_lc)

        w_init = naive_result

        # Solve the relaxed maximum algebraic connectivity augmentation problem.
        start = timer()
        result, unrounded, upper = mac.fw_subset(w_init, num_lc, max_iters=20, rounding="nearest")
        end = timer()
        solve_time = end - start
        times.append(solve_time)
        results.append(result)
        upper_bounds.append(upper)
        unrounded_results.append(unrounded)

        start = timer()
        result_cache, unrounded_cache, upper_cache = mac_cache.fw_subset(w_init, num_lc, max_iters=20, rounding="nearest", fallback=False)
        end = timer()
        solve_time = end - start
        cache_times.append(solve_time)
        cache_results.append(result_cache)
        cache_upper_bounds.append(upper_cache)
        cache_unrounded_results.append(unrounded_cache)

        # Verify that the results match
        # assert np.allclose(result, result_cache), "Results should be the same"
        # assert np.allclose(unrounded, unrounded_cache), "Unrounded results should be the same"
        # assert np.allclose(upper, upper_cache), "Upper bounds should be the same"
        pass

    # plot connectivity vs. percent_lc
    our_objective_vals = [mac.evaluate_objective(result) for result in results]
    unrounded_objective_vals = [mac.evaluate_objective(unrounded) for unrounded in unrounded_results]
    cache_objective_vals = [mac.evaluate_objective(cache_result) for cache_result in cache_results]
    cache_unrounded_objective_vals = [mac.evaluate_objective(cache_unrounded) for cache_unrounded in cache_unrounded_results]

    plt.plot(100.0*np.array(percent_lc), our_objective_vals, label='MAC (Ours) Nearest')
    plt.plot(100.0*np.array(percent_lc), upper_bounds, label='Dual Upper Bound', linestyle='--', color='C0')
    plt.fill_between(100.0*np.array(percent_lc), our_objective_vals, upper_bounds, alpha=0.2, label='Suboptimality Gap')
    plt.plot(100.0*np.array(percent_lc), unrounded_objective_vals, label='Unrounded', c='C2')

    plt.plot(100.0*np.array(percent_lc), cache_objective_vals, label='MAC (Cache)', marker='x', color='C4')
    plt.plot(100.0*np.array(percent_lc), cache_upper_bounds, label='Upper Bound (Cache)', linestyle='--', color='C4')
    plt.fill_between(100.0*np.array(percent_lc), cache_objective_vals, cache_upper_bounds, alpha=0.1, label='Suboptimality Gap (Cache)', color='C5')
    plt.plot(100.0*np.array(percent_lc), cache_unrounded_objective_vals, label='Unrounded (Cache)', c='C5')

    plt.ylabel(r'Algebraic Connectivity $\lambda_2$')
    plt.xlabel(r'\% Edges Added')
    plt.legend()
    plt.savefig(f"alg_conn_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(100.0*np.array(percent_lc), times, label='MAC (No Cache)')
    plt.plot(100.0*np.array(percent_lc), cache_times, label='MAC (Cache)', marker='x', color='C4')
    plt.xlim([0.0, 90.0])
    plt.ylabel(r'Time (s)')
    plt.xlabel(r'\% Edges Added')
    plt.savefig(f"comp_time_cache_compare_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.legend()
    plt.show()
