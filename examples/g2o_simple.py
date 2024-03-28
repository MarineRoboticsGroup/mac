import sys
import random
import os
from os.path import join
import numpy as np
import networkx as nx
from timeit import default_timer as timer
from pose_graph_utils import (
    read_g2o_file,
    plot_poses,
    rpm_to_mac,
    rpm_to_nx,
    RelativePoseMeasurement,
)
from typing import List, Dict, Tuple

# MAC requirements
from mac.mac import MAC
from mac.baseline import NaiveGreedy
from mac.greedy_esp import GreedyESP
from mac.utils import split_edges, Edge, round_madow

import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 16})


if __name__ == "__main__":

    ###### SETUP #######

    # load the experiment
    cur_file_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(cur_file_dir, "..", "data", "nclt")
    experiment_filepath = os.path.join(data_dir, "all_nclt_sessions_no_outliers.g2o")

    # file for the groundtruth values (can compare via `poses_ate_rot` or `poses_ate_tran`)
    groundtruth_filepath = os.path.join(data_dir, "gt_pose_all_nclt_sessions.tum")

    # whether we also want to run the GreedyESP algorithm
    run_greedy = True

    ####### RUN #########

    # load the NCLT data
    print(f"Reading NCLT data from {experiment_filepath}")
    start = timer()
    measurements, num_poses = read_g2o_file(experiment_filepath)
    end = timer()
    print(
        f"Read {len(measurements)} measurements from {num_poses} poses in {end-start:.2f} seconds"
    )

    measurements_as_nx_graph = rpm_to_nx(measurements)
    assert nx.is_connected(measurements_as_nx_graph), "Graph is not connected"

    # Split measurements into odom and loop closures
    odom_measurements, lc_measurements = split_edges(measurements)

    # Convert measurements to MAC edge format
    odom_edges = rpm_to_mac(odom_measurements)
    lc_edges = rpm_to_mac(lc_measurements)

    # Print dataset stats
    print(f"Loaded {len(measurements)} total measurements with: ")
    print(f"\t {len(odom_measurements)} base (odometry) measurements and")
    print(f"\t {len(lc_measurements)} candidate (loop closure) measurements")

    # Make a MAC Solver
    mac = MAC(
        odom_edges,
        lc_edges,
        num_poses,
        use_cache=True,
        fiedler_method="tracemin_cholesky",
    )

    # Make a Naive Solver
    naive = NaiveGreedy(lc_edges)

    # Make a GreedyEsp Solver
    if run_greedy:
        greedy_esp = GreedyESP(odom_edges, lc_edges, num_poses, lazy=True)

    #############################
    # Running the tests!
    #############################

    # Test between 100% and 0% loop closures
    # NOTE: If running greedy, these must be in increasing order!
    percent_lc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Container for Naive results
    naive_results = []

    # Containers for MAC results
    results = []
    unrounded_results = []
    upper_bounds = []
    times = []

    madow_results = []
    madow_times = []

    # Container for GreedyEig results
    greedy_eig_results = []
    greedy_eig_times = []

    # Container for GreedyESP results
    greedy_esp_results = []
    greedy_esp_times = []

    print("Running MAC!")
    for pct_lc in percent_lc:
        num_lc = int(pct_lc * len(lc_measurements))
        print("Num LC to accept: ", num_lc)

        # Compute a solution using the naive method. This serves both as a
        # baseline and as a sparse initializer for our method.
        naive_result = naive.subset(num_lc)
        naive_results.append(naive_result)

        w_init = naive_result

        # Solve the relaxed maximum algebraic connectivity augmentation problem.
        start = timer()
        result, unrounded, upper = mac.fw_subset(
            w_init, num_lc, max_iters=20, rounding="nearest"
        )
        end = timer()
        solve_time = end - start
        times.append(solve_time)
        results.append(result)
        upper_bounds.append(upper)
        unrounded_results.append(unrounded)

        start = timer()
        madow_rounded = round_madow(unrounded, num_lc, seed=np.random.RandomState(42))
        end = timer()
        madow_results.append(madow_rounded)
        madow_times.append(solve_time + (end - start))

    # Solve the relaxed maximum algebraic connectivity augmentation problem.
    if run_greedy:
        print("Running GreedyESP")
        num_lcs = [int(pct_lc * len(lc_measurements)) for pct_lc in percent_lc]
        greedy_esp_results, _, greedy_esp_times = greedy_esp.subsets_lazy(
            num_lcs, verbose=True
        )
        pass

    # Display the algebraic connectivity for each method
    for i in range(len(naive_results)):
        pct_lc = percent_lc[i]
        print(
            f"Naive AC at {pct_lc * 100.0} % loop closures: {mac.evaluate_objective(naive_results[i])}"
        )
        print(
            f"Our AC at {pct_lc * 100.0} % loop closures: {mac.evaluate_objective(results[i])}"
        )
        print(
            f"Our unrounded AC at {pct_lc * 100.0} % loop closures: {mac.evaluate_objective(unrounded_results[i])}"
        )
        print(f"Dual at {pct_lc * 100.0} % loop closures: {upper_bounds[i]}")
        if run_greedy:
            print(
                f"Greedy ESP AC at {pct_lc * 100.0} % loop closures: {mac.evaluate_objective(greedy_esp_results[i])}"
            )
            pass
        pass

    #############################
    # Plot the Results
    #############################

    # plot connectivity vs. percent_lc
    our_objective_vals = [mac.evaluate_objective(result) for result in results]
    naive_objective_vals = [
        mac.evaluate_objective(naive_result) for naive_result in naive_results
    ]
    unrounded_objective_vals = [
        mac.evaluate_objective(unrounded) for unrounded in unrounded_results
    ]
    madow_objective_vals = [mac.evaluate_objective(madow) for madow in madow_results]
    if run_greedy:
        # greedy_eig_objective_vals = [mac.evaluate_objective(ge_result) for ge_result in greedy_eig_results]
        greedy_esp_objective_vals = [
            mac.evaluate_objective(ge_result) for ge_result in greedy_esp_results
        ]

    plt.plot(
        100.0 * np.array(percent_lc), our_objective_vals, label="MAC Nearest (Ours)"
    )
    plt.plot(
        100.0 * np.array(percent_lc),
        madow_objective_vals,
        label="MAC Madow (Ours)",
        linestyle="-.",
        marker="x",
        color="C5",
    )

    plt.plot(
        100.0 * np.array(percent_lc),
        upper_bounds,
        label="Dual Upper Bound",
        linestyle="--",
        color="C0",
    )
    plt.fill_between(
        100.0 * np.array(percent_lc),
        our_objective_vals,
        upper_bounds,
        alpha=0.2,
        label="Suboptimality Gap",
    )
    plt.fill_between(
        100.0 * np.array(percent_lc),
        madow_objective_vals,
        upper_bounds,
        alpha=0.2,
        label="Suboptimality Gap",
    )
    plt.plot(
        100.0 * np.array(percent_lc),
        unrounded_objective_vals,
        label="Unrounded",
        c="C2",
    )
    plt.plot(
        100.0 * np.array(percent_lc),
        naive_objective_vals,
        label="Naive Method",
        color="red",
        linestyle="-.",
    )
    if run_greedy:
        # plt.plot(100.0*np.array(percent_lc), greedy_eig_objective_vals, label='Greedy E-Opt', color='orange')
        plt.plot(
            100.0 * np.array(percent_lc),
            greedy_esp_objective_vals,
            label="Greedy ESP",
            color="orange",
        )
        pass
    plt.ylabel(r"Algebraic Connectivity $\lambda_2$")
    plt.xlabel(r"\% Edges Added")
    plt.legend()
    plt.savefig(f"alg_conn_nclt.png", dpi=600, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.semilogy(100.0 * np.array(percent_lc), times, label="MAC Nearest (Ours)")
    plt.semilogy(
        100.0 * np.array(percent_lc),
        madow_times,
        label="MAC Madow (Ours)",
        linestyle="-.",
        marker="x",
        color="C5",
    )
    if run_greedy:
        # plt.plot(100.0*np.array(percent_lc), greedy_eig_times, label='Greedy E-Opt', color='orange')
        plt.semilogy(
            100.0 * np.array(percent_lc),
            greedy_esp_times,
            label="Greedy ESP",
            color="orange",
        )
    plt.xlim([0.0, 90.0])
    plt.ylabel(r"Time (s)")
    plt.xlabel(r"\% Edges Added")
    plt.savefig(f"comp_time_nclt.png", dpi=600, bbox_inches="tight")
    plt.legend()
    plt.show()
