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
    ate_tran,
    rpe_rot
)
from typing import List, Dict, Tuple

from evo.tools import file_interface
from evo.core import sync, metrics
from evo.core.trajectory import PoseTrajectory3D

# MAC requirements
from mac.mac import MAC
from mac.baseline import NaiveGreedy
from mac.greedy_esp import GreedyESP
from mac.utils import split_edges, Edge, round_madow

import matplotlib.pyplot as plt

from pose_graph_utils import read_g2o_file, plot_poses, rpm_to_mac, RelativePoseMeasurement

plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 16})

from SESyncFactor3d import RelativePose3dFactor
import gtsam

def select_measurements(measurements, w):
    assert(len(measurements) == len(w))
    meas_out = []
    for i, meas in enumerate(measurements):
        if w[i] == 1.0:
            meas_out.append(meas)
    return meas_out

def gtsam_values_to_traj(values, is3D, times=None):
    """Convert a set of gtsam.Values of type gtsam.pose{D} (D = 2 or 3) to a
    trajectory that can parsed by evo.

    `values`: a set of gtsam.Values of type gtsam.Pose{D} where D = 2 or 3
    `is3D`: true or false (0/1). Set to true of `values` contains gtsam.Pose3
    `times`: a list of timestamps for the poses in `values`. If None, the
        timestamps are set to the indices of the poses in `values`.

    returns PoseTrajectory3D object for processing with evo
    """
    traj = []  # a List[np.ndarray] where each item is a 4x4 SE(3) matrix
    if times is None:
        timestamps = np.array(range(len(values.keys())).astype(np.float64))
    else:
        timestamps = np.array(times).astype(np.float64)

    for k in values.keys():
        if is3D:
            pose = values.atPose3(k).matrix()
            traj.append(pose)
        else:
            pose2 = values.atPose2(k).matrix()
            traj.append(se2_to_se3(pose2))

    return PoseTrajectory3D(poses_se3 = traj,
                            timestamps = timestamps)

def run_gtsam_solve(odom_measurements, lc_measurements, initial, use_cauchy=False, use_frobenius=False):
    # Create a factor graph container for all the measurements
    graph = gtsam.NonlinearFactorGraph()

    # This is a hack so we can use writeG2o to write the graph to a file
    # graphNoKernel will use only BetweenFactors so we can keep the topology of the graph
    # and the resulting values, but not the precise model. The g2o file will be used
    # for visualizing the results.
    graphNoKernel = gtsam.NonlinearFactorGraph()
    for meas in odom_measurements:
        idx1 = meas.i
        idx2 = meas.j
        if use_frobenius:
            translation_precision = meas.tau
            rotation_precision = meas.kappa
            noise_model = gtsam.noiseModel.Diagonal.Precisions(
                2 * np.array([translation_precision] * 3 + [rotation_precision] * 9)
            )
            odom_factor = RelativePose3dFactor(idx1, idx2, meas.R, meas.t, noise_model)
        else:
            noise_model = gtsam.noiseModel.Diagonal.Precisions(np.array([10e06, 10e06, 10e06, 10e04, 10e04, 10e04]))
            odom_factor = gtsam.BetweenFactorPose3(idx1, idx2, gtsam.Pose3(gtsam.Rot3(meas.R), meas.t), noise_model)

        nokernel_factor = gtsam.BetweenFactorPose3(idx1, idx2, gtsam.Pose3(gtsam.Rot3(meas.R), meas.t), noise_model)
        graph.add(odom_factor)
        graphNoKernel.add(nokernel_factor)

    if use_frobenius:
        translation_precision = 0.5
        rotation_precision = 0.25
        base_loop_model = gtsam.noiseModel.Diagonal.Precisions(
            2 * np.array([translation_precision] * 3 + [rotation_precision] * 9)
        )
    else:
        base_loop_model = gtsam.noiseModel.Isotropic.Variance(6, 2.0)

    if use_cauchy:
        loop_model = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy(1.0), base_loop_model)
    else:
        loop_model = base_loop_model
    for meas in lc_measurements:
        idx1 = meas.i
        idx2 = meas.j
        if use_frobenius:
            loop_factor = RelativePose3dFactor(idx1, idx2, meas.R, meas.t, loop_model)
        else:
            loop_factor = gtsam.BetweenFactorPose3(idx1, idx2, gtsam.Pose3(gtsam.Rot3(meas.R), meas.t), loop_model)

        nokernel_betweenfactor_model = gtsam.noiseModel.Isotropic.Variance(6, 2.0)
        nokernel_factor = gtsam.BetweenFactorPose3(idx1, idx2, gtsam.Pose3(gtsam.Rot3(meas.R), meas.t), nokernel_betweenfactor_model)
        graph.add(loop_factor)
        graphNoKernel.add(nokernel_factor)

        pass

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    return result, graphNoKernel


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [path to g2o] [path to times] [path to ground truth] [optional: --run-greedy --use-cauchy --use-frobenius]")
        sys.exit()
        pass

    g2o_path = sys.argv[1]
    times_path = sys.argv[2]
    gt_path = sys.argv[3]
    run_greedy = "--run-greedy" in sys.argv
    use_cauchy = "--use-cauchy" in sys.argv
    use_frobenius = "--use-frobenius" in sys.argv

    # Get dataset name as last part of path before g2o extension
    dataset_name = sys.argv[1].split('/')[-1].split('.')[0]

    # load the NCLT data
    print(f"Reading dataset {dataset_name} from {g2o_path}...")
    start = timer()
    measurements, num_poses = read_g2o_file(g2o_path)
    end = timer()
    print(f"Read {len(measurements)} measurements from {num_poses} poses in {end-start:.2f} seconds")

    # Get auxiliary data: the initial guess (as gtsam.Values) and the keyframe timestamps
    gtsam_graph, initial = gtsam.readG2o(g2o_path, is3D=True)
    keyframe_times = np.loadtxt(times_path)

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
    mac = MAC(odom_edges, lc_edges, num_poses, use_cache=True, fiedler_method="tracemin_cholesky")

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
        result, unrounded, upper = mac.fw_subset(w_init, num_lc, max_iters=20, rounding="nearest")
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
        greedy_esp_results, _, greedy_esp_times = greedy_esp.subsets_lazy(num_lcs, verbose=True)
        pass

    # Display the algebraic connectivity for each method
    for i in range(len(naive_results)):
        pct_lc = percent_lc[i]
        print(f"Naive AC at {pct_lc * 100.0} % loop closures: {mac.evaluate_objective(naive_results[i])}")
        print(f"Our AC at {pct_lc * 100.0} % loop closures: {mac.evaluate_objective(results[i])}")
        print(f"Our unrounded AC at {pct_lc * 100.0} % loop closures: {mac.evaluate_objective(unrounded_results[i])}")
        print(f"Dual at {pct_lc * 100.0} % loop closures: {upper_bounds[i]}")
        if run_greedy:
            print(f"Greedy ESP AC at {pct_lc * 100.0} % loop closures: {mac.evaluate_objective(greedy_esp_results[i])}")
            pass
        pass

    #############################
    # Plot the Results
    #############################

    colors = {"MAC Nearest (Ours)": "C0",
              "MAC Madow (Ours)": "C4",
              "Unrounded": "C2",
              "Dual Upper Bound": "C0",
              "Greedy ESP": "C1",
              "Naive Method": "C3"}

    # plot connectivity vs. percent_lc
    our_objective_vals = [mac.evaluate_objective(result) for result in results]
    naive_objective_vals = [mac.evaluate_objective(naive_result) for naive_result in naive_results]
    unrounded_objective_vals = [mac.evaluate_objective(unrounded) for unrounded in unrounded_results]
    madow_objective_vals = [mac.evaluate_objective(madow) for madow in madow_results]
    if run_greedy:
        # greedy_eig_objective_vals = [mac.evaluate_objective(ge_result) for ge_result in greedy_eig_results]
        greedy_esp_objective_vals = [mac.evaluate_objective(ge_result) for ge_result in greedy_esp_results]

    plt.plot(100.0*np.array(percent_lc), our_objective_vals, label='MAC Nearest (Ours)', marker='s', color=colors["MAC Nearest (Ours)"])
    plt.plot(100.0*np.array(percent_lc), madow_objective_vals, label='MAC Madow (Ours)', marker='o', color=colors["MAC Madow (Ours)"])

    plt.plot(100.0*np.array(percent_lc), upper_bounds, label='Dual Upper Bound', linestyle='--', color=colors["Dual Upper Bound"])
    plt.fill_between(100.0*np.array(percent_lc), our_objective_vals, upper_bounds, alpha=0.1)
    plt.fill_between(100.0*np.array(percent_lc), madow_objective_vals, upper_bounds, alpha=0.1, color='C4')

    plt.plot(100.0*np.array(percent_lc), unrounded_objective_vals, label='Unrounded', color=colors["Unrounded"])

    if run_greedy:
        plt.plot(100.0*np.array(percent_lc), greedy_esp_objective_vals, label='Greedy ESP', marker='o', color=colors["Greedy ESP"])
        pass
    plt.plot(100.0*np.array(percent_lc), naive_objective_vals, label='Naive Method', marker='o', color=colors["Naive Method"])

    plt.ylabel(r'Algebraic Connectivity $\lambda_2$')
    plt.xlabel(r'\% Edges Added')
    plt.legend()
    plt.savefig(f"alg_conn_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"alg_conn_{dataset_name}_300.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"alg_conn_{dataset_name}.svg", transparent=True,bbox_inches='tight')
    plt.show()

    # Plot computation time vs. percent_lc
    plt.figure()
    plt.semilogy(100.0*np.array(percent_lc[:-1]), times[:-1], label='MAC Nearest (Ours)', marker='s', color=colors["MAC Nearest (Ours)"])
    plt.semilogy(100.0*np.array(percent_lc[:-1]), madow_times[:-1], label='MAC Madow (Ours)', marker='o', color=colors["MAC Madow (Ours)"])
    if run_greedy:
        # plt.plot(100.0*np.array(percent_lc), greedy_eig_times, label='Greedy E-Opt', color='orange')
        plt.semilogy(100.0*np.array(percent_lc[:-1]), greedy_esp_times[:-1], label='Greedy ESP', marker='o', color=colors["Greedy ESP"])
    plt.xlim([0.0, 100.0])
    plt.ylabel(r'Time (s)')
    plt.xlabel(r'\% Edges Added')
    plt.legend()
    plt.savefig(f"comp_time_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"comp_time_{dataset_name}_300.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"comp_time_{dataset_name}.svg", transparent=True, bbox_inches='tight')
    plt.show()

    #############################
    # Run GTSAM
    #############################

    print("Optimizing the pose graph (this may take a while)...")

    gtsam_ours = []
    gtsam_madow = []
    gtsam_naive = []
    gtsam_esp = []
    for i, pct in enumerate(percent_lc):
        our_selected_lc = select_measurements(lc_measurements, results[i])
        our_gtsam_result, our_gtsam_graph_nokernel = run_gtsam_solve(odom_measurements, our_selected_lc, initial, use_cauchy=use_cauchy, use_frobenius=use_frobenius)
        our_gtsam_traj = gtsam_values_to_traj(our_gtsam_result, is3D=True, times=keyframe_times)
        gtsam_ours.append(our_gtsam_traj)
        gtsam.writeG2o(our_gtsam_graph_nokernel, our_gtsam_result, f"gtsam_ours_{dataset_name}_{pct}.g2o")

        madow_selected_lc = select_measurements(lc_measurements, madow_results[i])
        madow_gtsam_result, madow_gtsam_graph_nokernel = run_gtsam_solve(odom_measurements, madow_selected_lc, initial, use_cauchy=use_cauchy, use_frobenius=use_frobenius)
        madow_gtsam_traj = gtsam_values_to_traj(madow_gtsam_result, is3D=True, times=keyframe_times)
        gtsam_madow.append(madow_gtsam_traj)
        gtsam.writeG2o(madow_gtsam_graph_nokernel, madow_gtsam_result, f"gtsam_madow_{dataset_name}_{pct}.g2o")

        naive_selected_lc = select_measurements(lc_measurements, naive_results[i])
        naive_gtsam_result, naive_gtsam_graph_nokernel = run_gtsam_solve(odom_measurements, naive_selected_lc, initial, use_cauchy=use_cauchy, use_frobenius=use_frobenius)
        naive_gtsam_traj = gtsam_values_to_traj(naive_gtsam_result, is3D=True, times=keyframe_times)
        gtsam_naive.append(naive_gtsam_traj)
        gtsam.writeG2o(naive_gtsam_graph_nokernel, naive_gtsam_result, f"gtsam_naive_{dataset_name}_{pct}.g2o")

        if run_greedy:
            esp_selected_lc = select_measurements(lc_measurements, greedy_esp_results[i])
            esp_gtsam_result, esp_gtsam_graph_nokernel = run_gtsam_solve(odom_measurements, esp_selected_lc, initial, use_cauchy=use_cauchy, use_frobenius=use_frobenius)
            esp_gtsam_traj = gtsam_values_to_traj(esp_gtsam_result, is3D=True, times=keyframe_times)
            gtsam_esp.append(esp_gtsam_traj)
            gtsam.writeG2o(esp_gtsam_graph_nokernel, esp_gtsam_result, f"gtsam_esp_{dataset_name}_{pct}.g2o")

    print("Finished pose-graph optimization")
    print("Loading ground-truth (this may take a while) ...")
    traj_ref = file_interface.read_tum_trajectory_file(gt_path)
    print(f"Successfully loaded ground truth with {traj_ref.num_poses} poses")
    print("Computing ATE and RPE...")

    our_ates = [ate_tran(traj_ref, gtsam_ours[i]) for i in range(len(gtsam_ours))]
    our_rpes = [rpe_rot(traj_ref, gtsam_ours[i]) for i in range(len(gtsam_ours))]
    madow_ates = [ate_tran(traj_ref, gtsam_madow[i]) for i in range(len(gtsam_madow))]
    madow_rpes = [rpe_rot(traj_ref, gtsam_madow[i]) for i in range(len(gtsam_madow))]
    naive_ates = [ate_tran(traj_ref, gtsam_naive[i]) for i in range(len(gtsam_naive))]
    naive_rpes = [rpe_rot(traj_ref, gtsam_naive[i]) for i in range(len(gtsam_naive))]
    if run_greedy:
        esp_ates = [ate_tran(traj_ref, gtsam_esp[i]) for i in range(len(gtsam_esp))]
        esp_rpes = [rpe_rot(traj_ref, gtsam_esp[i]) for i in range(len(gtsam_esp))]

    # traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    # umeyama_result = traj_est.align(traj_ref, correct_scale=False, correct_only_scale=False)
    # plot_evo_trajs([traj_est, traj_ref], ["estimate", "reference"], vertical_layout=False)



    plt.figure()
    plt.plot(100.0*np.array(percent_lc), our_ates, label='MAC Nearest (Ours)', marker='s', color=colors["MAC Nearest (Ours)"])
    plt.plot(100.0*np.array(percent_lc), madow_ates, label='MAC Madow (Ours)', marker='o', color=colors["MAC Madow (Ours)"])
    if run_greedy:
        plt.plot(100.0*np.array(percent_lc), esp_ates, label='Greedy ESP', marker='o', color=colors["Greedy ESP"])
    plt.plot(100.0*np.array(percent_lc), naive_ates, label='Naive Method', marker='o', color=colors["Naive Method"])
    plt.xlabel(r'\% Edges Added')
    plt.ylabel(r'ATE Translation vs. Ground Truth [m]')
    plt.legend()
    plt.savefig(f"ate_tran_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"ate_tran_{dataset_name}_300.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"ate_tran_{dataset_name}.svg", transparent=True, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.semilogy(100.0*np.array(percent_lc[:-1]), our_ates[:-1], label='MAC Nearest (Ours)', marker='s', color=colors["MAC Nearest (Ours)"])
    plt.semilogy(100.0*np.array(percent_lc[:-1]), madow_ates[:-1], label='MAC Madow (Ours)', marker='o', color=colors["MAC Madow (Ours)"])
    if run_greedy:
        plt.semilogy(100.0*np.array(percent_lc[:-1]), esp_ates[:-1], label='Greedy ESP', marker='o', color=colors["Greedy ESP"])
    plt.semilogy(100.0*np.array(percent_lc[:-1]), naive_ates[:-1], label='Naive Method', marker='o', color=colors["Naive Method"])
    plt.xlabel(r'\% Edges Added')
    plt.ylabel(r'ATE Translation vs. Ground Truth [m]')
    plt.xlim([0.0, 100.0])
    plt.legend()
    plt.savefig(f"ate_tran_{dataset_name}_log.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"ate_tran_{dataset_name}_log_300.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"ate_tran_{dataset_name}_log.svg", transparent=True, bbox_inches='tight')
    plt.show()


    plt.figure()
    plt.plot(100.0*np.array(percent_lc), our_rpes, label='MAC Nearest (Ours)', marker='s', color=colors["MAC Nearest (Ours)"])
    plt.plot(100.0*np.array(percent_lc), madow_rpes, label='MAC Madow (Ours)', marker='o', color=colors["MAC Madow (Ours)"])
    if run_greedy:
        plt.plot(100.0*np.array(percent_lc), esp_rpes, label='Greedy ESP', marker='o', color=colors["Greedy ESP"])
    plt.plot(100.0*np.array(percent_lc), naive_rpes, label='Naive Method', marker='o', color=colors["Naive Method"])
    plt.xlabel(r'\% Edges Added')
    plt.ylabel(r'RPE Rotation vs. Ground Truth [deg]')
    plt.legend()
    plt.savefig(f"rpe_rot_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"rpe_rot_{dataset_name}_300.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"rpe_rot_{dataset_name}.svg", transparent=True, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.semilogy(100.0*np.array(percent_lc[:-1]), our_rpes[:-1], label='MAC Nearest (Ours)', marker='s', color=colors["MAC Nearest (Ours)"])
    plt.semilogy(100.0*np.array(percent_lc[:-1]), madow_rpes[:-1], label='MAC Madow (Ours)', marker='o', color=colors["MAC Madow (Ours)"])
    if run_greedy:
        plt.semilogy(100.0*np.array(percent_lc[:-1]), esp_rpes[:-1], label='Greedy ESP', marker='o', color=colors["Greedy ESP"])
    plt.semilogy(100.0*np.array(percent_lc[:-1]), naive_rpes[:-1], label='Naive Method', marker='o', color=colors["Naive Method"])
    plt.xlabel(r'\% Edges Added')
    plt.ylabel(r'RPE Rotation vs. Ground Truth [deg]')
    plt.xlim([0.0, 100.0])
    plt.legend()
    plt.savefig(f"rpe_rot_{dataset_name}_log.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"rpe_rot_{dataset_name}_log_300.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"rpe_rot_{dataset_name}_log.svg", transparent=True, bbox_inches='tight')
    plt.show()

