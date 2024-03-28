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

from pose_graph_utils import read_g2o_file, plot_poses, rpm_to_mac, RelativePoseMeasurement

plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 16})

# SE-Sync setup
sesync_lib_path = "/Users/kevin/repos/SESync/C++/build/lib"
sys.path.insert(0, sesync_lib_path)

import PySESync

from g2o_experiment import *

def offset_keys_for_within_session_measurements(
    measurements: List[RelativePoseMeasurement], offset: int
) -> List[RelativePoseMeasurement]:
    """As the keys of within-session measurements do not account for the
    previous sessions, we need to offset them to avoid collisions with
    measurements from other sessions.

    RelativePoseMeasurement = namedtuple('RelativePoseMeasurement',
                                        ['i', 'j', 't', 'R', 'kappa', 'tau'])

    Args:
        measurements (List[RelativePoseMeasurement]): the original measurements
        from a single session
        offset (int): the offset to add to the keys

    Returns:
        List[RelativePoseMeasurement]: the measurements with the keys offset
    """
    new_measurements = []
    for m in measurements:
        new_measurements.append(
            RelativePoseMeasurement(
                m.i + offset, m.j + offset, m.t, m.R, m.kappa, m.tau
            )
        )
    return new_measurements


def get_char_and_index_from_key(key: int) -> Tuple[str, int]:
    """Given a GTSAM key, return the character and index. More details below:

        The GTSAM symbol syntax is (character, index). For example, ('A', 1) or
        ('B', 292). The symbol is then mapped to a key, which is a uint64_t integer, according to the following code:

        #### GTSAM symbol mapping  (C++) ####
        static const size_t keyBits = sizeof(Key) * 8;
        static const size_t chrBits = sizeof(unsigned char) * 8;
        static const size_t indexBits = keyBits - chrBits;
        static const Key chrMask = Key(UCHAR_MAX)  << indexBits; // For some reason, std::numeric_limits<unsigned char>::max() fails
        static const Key indexMask = ~chrMask;

        Key Symbol::symbol(unsigned char c, std::uint64_t j) { return (Key)Symbol(c,j); }

        Symbol::Symbol(Key key) :
            c_((unsigned char) ((key & chrMask) >> indexBits)),
            j_ (key & indexMask)
            { }

        Key Symbol::key() const {
            if (j_ > indexMask) {
                boost::format msg("Symbol index is too large, j=%d, indexMask=%d");
                msg % j_ % indexMask;
                throw std::invalid_argument(msg.str());
            }
            Key key = (Key(c_) << indexBits) | j_;
            return key;
        }

    Args:
        key (int): the key, which is a uint64_t integer

    Returns:
        Tuple[str, int]: the character and index
    """
    assert key >= 0, "Key must be non-negative"
    key_bits = 64
    chr_bits = 8
    index_bits = key_bits - chr_bits
    chr_mask = 255 << index_bits

    # flip the bits of chr_mask to get indexMask
    chr_mask_bin = bin(chr_mask)[2:].zfill(64)
    index_mask_bin = "".join(["0" if c == "1" else "1" for c in chr_mask_bin])
    index_mask = int(index_mask_bin, 2)

    c = chr((key & chr_mask) >> index_bits)
    j = key & index_mask
    return c, j


def print_bit_mask(mask: int) -> None:
    """Print a bit mask.

    Args:
        mask (int): the bit mask
    """
    print(bin(mask)[2:].zfill(64))


def rekey_inter_session_measurements(
    measurements: List[RelativePoseMeasurement], offsets: Dict[str, int]
) -> List[RelativePoseMeasurement]:
    """The keys of inter-session measurements are based on GTSAM symbols, with
    letters to indicate the session and numbers to indicate the key within the
    session. We need to invert the GTSAM symbol mapping and then perform the
    rekeying by properly offsetting the indices.
    Args:
        measurements (List[RelativePoseMeasurement]): the original measurements
        offsets (Dict[str, int]): a dictionary mapping session names to offsets

    Returns:
        List[RelativePoseMeasurement]: the measurements with the keys rekeyed
    """
    new_measurements = []
    for m in measurements:
        char_i, index_i = get_char_and_index_from_key(m.i)
        char_j, index_j = get_char_and_index_from_key(m.j)
        assert char_i in offsets, "Session {} not found in offsets".format(char_i)
        assert char_j in offsets, "Session {} not found in offsets".format(char_j)
        new_measurements.append(
            RelativePoseMeasurement(
                index_i + offsets[char_i],
                index_j + offsets[char_j],
                m.t,
                m.R,
                m.kappa,
                m.tau,
            )
        )
    return new_measurements


def read_nclt_experiments(
    base_nclt_data_dir: str = os.path.expanduser(
        "~/mac-mission-control/nclt-processing"
    ),
    run_sanity_tests: bool = False,
) -> Tuple[List[RelativePoseMeasurement], int]:
    """Reads the multiple sessions of the NCLT dataset and returns the
    measurements.

    Args:
        base_nclt_data_dir (str, optional): The path to the root directory of
            the NCLT dataset. Defaults to
            os.path.expanduser("~/mac-mission-control/nclt-processing").
        run_sanity_tests (bool, optional): Whether to run sanity tests on the
            data. Useful for debugging. Defaults to False.

    Returns:
        List[RelativePoseMeasurement]: the measurements from all sessions and
            inter-session loop closures
        int: the number of poses in the dataset
    """

    # get the directories holding each session
    session_subdirs = [join(data_dir, f"data-{i}") for i in range(1, 4)]
    assert all(
        [os.path.isdir(d) for d in session_subdirs]
    ), f"Missing data directories: {session_subdirs}"
    assert len(session_subdirs) == 3, "Expected 3 sessions"

    measurements = []
    num_poses = 0
    session_offsets = {}
    for i, session_dir in enumerate(session_subdirs):
        print(f"Processing session {i+1}...")

        # add the offsets we will need to apply to indices to rekey the measurements
        session_offset_char = chr(ord("a") + i)
        assert session_offset_char in [
            "a",
            "b",
            "c",
        ], "Invalid session offset character"
        session_offsets[session_offset_char] = num_poses

        # get the g2o file from just the data in this session
        within_session_file = join(session_dir, "graph.g2o")
        assert os.path.isfile(
            within_session_file
        ), f"Missing within-session file: {within_session_file}"
        print(f"Reading within-session measurements from {within_session_file}")
        within_session_measurements, within_session_num_poses = read_g2o_file(
            within_session_file
        )
        rekeyed_within_session_measurements = (
            offset_keys_for_within_session_measurements(
                within_session_measurements, num_poses
            )
        )
        num_poses += within_session_num_poses
        measurements.extend(rekeyed_within_session_measurements)

        # get the g2o file for any loop closures found between this session and
        # previous sessions
        inter_session_file = join(session_dir, "inter_graph.g2o")
        assert os.path.isfile(
            inter_session_file
        ), f"Missing inter-session file: {inter_session_file}"
        print(f"Reading inter-session measurements from {inter_session_file}")
        inter_session_measurements, _ = read_g2o_file(inter_session_file)
        rekeyed_inter_session_measurements = rekey_inter_session_measurements(
            inter_session_measurements, session_offsets
        )
        measurements.extend(rekeyed_inter_session_measurements)

        # run some sanity checks on the keys
        if run_sanity_tests:
            for m in (
                rekeyed_within_session_measurements + rekeyed_inter_session_measurements
            ):
                assert 0 <= m.i < num_poses, f"Invalid key {m.i} for pose {num_poses}"
                assert 0 <= m.j < num_poses, f"Invalid key {m.j} for pose {num_poses}"

    return measurements, num_poses


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [path to dataset] [optional: --run-greedy]")
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
    # data_dir = "/home/alan/mac-mission-control/nclt-processing/data/NCLT-1-3"
    # run_greedy = True

    dataset_name = "nclt"
    data_dir = sys.argv[1]

    # load the NCLT data
    print(f"Reading NCLT data from {data_dir}")
    start = timer()
    measurements, num_poses = read_nclt_experiments(data_dir)
    end = timer()
    print(f"Read {len(measurements)} measurements from {num_poses} poses in {end-start:.2f} seconds")

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

    # plot connectivity vs. percent_lc
    our_objective_vals = [mac.evaluate_objective(result) for result in results]
    naive_objective_vals = [mac.evaluate_objective(naive_result) for naive_result in naive_results]
    unrounded_objective_vals = [mac.evaluate_objective(unrounded) for unrounded in unrounded_results]
    madow_objective_vals = [mac.evaluate_objective(madow) for madow in madow_results]
    if run_greedy:
        # greedy_eig_objective_vals = [mac.evaluate_objective(ge_result) for ge_result in greedy_eig_results]
        greedy_esp_objective_vals = [mac.evaluate_objective(ge_result) for ge_result in greedy_esp_results]

    plt.plot(100.0*np.array(percent_lc), our_objective_vals, label='MAC Nearest (Ours)')
    plt.plot(100.0*np.array(percent_lc), madow_objective_vals, label='MAC Madow (Ours)', linestyle='-.', marker='x', color='C5')

    plt.plot(100.0*np.array(percent_lc), upper_bounds, label='Dual Upper Bound', linestyle='--', color='C0')
    plt.fill_between(100.0*np.array(percent_lc), our_objective_vals, upper_bounds, alpha=0.2, label='Suboptimality Gap')
    plt.fill_between(100.0*np.array(percent_lc), madow_objective_vals, upper_bounds, alpha=0.2, label='Suboptimality Gap')
    plt.plot(100.0*np.array(percent_lc), unrounded_objective_vals, label='Unrounded', c='C2')
    plt.plot(100.0*np.array(percent_lc), naive_objective_vals, label='Naive Method', color='red', linestyle='-.')
    if run_greedy:
        # plt.plot(100.0*np.array(percent_lc), greedy_eig_objective_vals, label='Greedy E-Opt', color='orange')
        plt.plot(100.0*np.array(percent_lc), greedy_esp_objective_vals, label='Greedy ESP', color='orange')
        pass
    plt.ylabel(r'Algebraic Connectivity $\lambda_2$')
    plt.xlabel(r'\% Edges Added')
    plt.legend()
    plt.savefig(f"alg_conn_nclt.png", dpi=600, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.semilogy(100.0*np.array(percent_lc), times, label='MAC Nearest (Ours)')
    plt.semilogy(100.0*np.array(percent_lc), madow_times, label='MAC Madow (Ours)', linestyle='-.', marker='x', color='C5')
    if run_greedy:
        # plt.plot(100.0*np.array(percent_lc), greedy_eig_times, label='Greedy E-Opt', color='orange')
        plt.semilogy(100.0*np.array(percent_lc), greedy_esp_times, label='Greedy ESP', color='orange')
    plt.xlim([0.0, 90.0])
    plt.ylabel(r'Time (s)')
    plt.xlabel(r'\% Edges Added')
    plt.savefig(f"comp_time_nclt.png", dpi=600, bbox_inches='tight')
    plt.legend()
    plt.show()

    #############################
    # Run SE-Sync
    #############################

    # SE-Sync Setup
    d = odom_measurements[0].R.shape[0]
    opts = PySESync.SESyncOpts()
    opts.num_threads = 4
    opts.verbose=True
    opts.r0 = d + 1  # Start at level d + 1 of the Riemannian Staircase

    sesync_results = []
    sesync_madow = []
    sesync_naive = []
    sesync_eig = []
    sesync_esp = []
    for i in range(len(naive_results)):
        our_selected_lc = select_measurements(lc_measurements, results[i])
        our_meas = odom_measurements + our_selected_lc
        sesync_our_meas = to_sesync_format(our_meas)
        sesync_result = PySESync.SESync(sesync_our_meas, opts)
        sesync_results.append(sesync_result)

        madow_selected_lc = select_measurements(lc_measurements, madow_results[i])
        madow_meas = odom_measurements + madow_selected_lc
        sesync_result_madow = PySESync.SESync(to_sesync_format(madow_meas), opts)
        sesync_madow.append(sesync_result_madow)

        naive_selected_lc = select_measurements(lc_measurements, naive_results[i])
        naive_meas = odom_measurements + naive_selected_lc
        sesync_result_naive = PySESync.SESync(to_sesync_format(naive_meas), opts)
        sesync_naive.append(sesync_result_naive)

        if run_greedy:
            esp_selected_lc = select_measurements(lc_measurements, greedy_esp_results[i])
            esp_meas = odom_measurements + esp_selected_lc
            sesync_result_esp = PySESync.SESync(to_sesync_format(esp_meas), opts)
            sesync_esp.append(sesync_result_esp)



    plt.figure()
    plt.plot(100.0*np.array(percent_lc), [res.total_computation_time for res in sesync_results], label='MAC Nearest (Ours)')
    plt.plot(100.0*np.array(percent_lc), [res.total_computation_time for res in sesync_madow], label='MAC Madow (Ours)', linestyle='-.', marker='x', color='C5')
    if run_greedy:
        plt.plot(100.0*np.array(percent_lc), [res.total_computation_time for res in sesync_esp], label='Greedy ESP', color='orange', linestyle='-')
    plt.plot(100.0*np.array(percent_lc), [res.total_computation_time for res in sesync_naive], label='Naive Method', color='red', linestyle='-.')
    plt.legend()
    plt.xlabel(r'\% Edges Added')
    plt.ylabel(r'Time (s)')
    plt.savefig(f"sesync_comp_time_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.show()

    # Compute the SE-Sync cost for each solution under the *full* set of measurements
    M = construct_sesync_quadratic_form_matrix(measurements)
    LGrho = construct_LGrho(measurements)
    sesync_full = PySESync.SESync(to_sesync_format(measurements), opts)

    our_rot_costs = []
    our_full_costs = []
    our_SOd_orbdists = []

    madow_rot_costs = []
    madow_full_costs = []
    madow_SOd_orbdists = []

    naive_rot_costs = []
    naive_full_costs = []
    naive_SOd_orbdists = []

    esp_rot_costs = []
    esp_full_costs = []
    esp_SOd_orbdists = []
    for i in range(len(percent_lc)):
        print(f"Percent LC: {percent_lc[i]}")

        xhat_ours = sesync_results[i].xhat
        xhat_madow = sesync_madow[i].xhat
        xhat_naive = sesync_naive[i].xhat

        our_selected_lc = select_measurements(lc_measurements, results[i])
        our_meas = odom_measurements + our_selected_lc

        madow_selected_lc = select_measurements(lc_measurements, madow_results[i])
        madow_meas = odom_measurements + madow_selected_lc

        naive_selected_lc = select_measurements(lc_measurements, naive_results[i])
        naive_meas = odom_measurements + naive_selected_lc

        our_rot_cost = evaluate_sesync_rotation_objective(LGrho, xhat_ours[:, num_poses:])
        our_full_cost = evaluate_sesync_objective(M, xhat_ours)
        our_SOd_orbdist = orbit_distance_dS(sesync_full.xhat[:,num_poses:], xhat_ours[:,num_poses:])

        our_rot_costs.append(our_rot_cost)
        our_full_costs.append(our_full_cost)
        our_SOd_orbdists.append(our_SOd_orbdist)

        plot_poses(xhat_ours, our_meas, show=False)
        plt.savefig(f"ours_{dataset_name}_{str(percent_lc[i])}.png", dpi=600)
        plt.close()

        # Print error for our method
        print(f"Our rotation cost: {our_rot_cost}")
        print(f"Our full cost {our_full_cost}")
        print(f"Our SO orbdist: {our_SOd_orbdist}")

        madow_rot_cost = evaluate_sesync_rotation_objective(LGrho, xhat_madow[:, num_poses:])
        madow_full_cost = evaluate_sesync_objective(M, xhat_madow)
        madow_SOd_orbdist = orbit_distance_dS(sesync_full.xhat[:,num_poses:], xhat_madow[:,num_poses:])

        madow_rot_costs.append(madow_rot_cost)
        madow_full_costs.append(madow_full_cost)
        madow_SOd_orbdists.append(madow_SOd_orbdist)

        plot_poses(xhat_madow, madow_meas, show=False)
        plt.savefig(f"madow_{dataset_name}_{str(percent_lc[i])}.png", dpi=600)
        plt.close()

        naive_rot_cost = evaluate_sesync_rotation_objective(LGrho, xhat_naive[:, num_poses:])
        naive_full_cost = evaluate_sesync_objective(M, xhat_naive)
        naive_SOd_orbdist = orbit_distance_dS(sesync_full.xhat[:,num_poses:], xhat_naive[:,num_poses:])

        naive_rot_costs.append(naive_rot_cost)
        naive_full_costs.append(naive_full_cost)
        naive_SOd_orbdists.append(naive_SOd_orbdist)

        plot_poses(xhat_naive, naive_meas, show=False)
        plt.savefig(f"naive_{dataset_name}_{str(percent_lc[i])}.png", dpi=600)
        plt.close()

        # Print Error for naive method
        print(f"Naive rotation cost: {naive_rot_cost}")
        print(f"Naive full cost: {naive_full_cost}")
        print(f"Naive SO orbdist: {naive_SOd_orbdist}")

        if run_greedy:
            xhat_esp = sesync_esp[i].xhat
            esp_selected_lc = select_measurements(lc_measurements, greedy_esp_results[i])
            esp_meas = odom_measurements + esp_selected_lc

            esp_rot_cost = evaluate_sesync_rotation_objective(LGrho, xhat_esp[:, num_poses:])
            esp_full_cost = evaluate_sesync_objective(M, xhat_esp)
            esp_SOd_orbdist = orbit_distance_dS(sesync_full.xhat[:,num_poses:], xhat_esp[:,num_poses:])

            esp_rot_costs.append(esp_rot_cost)
            esp_full_costs.append(esp_full_cost)
            esp_SOd_orbdists.append(esp_SOd_orbdist)

            plot_poses(xhat_esp, esp_meas, show=False)
            plt.savefig(f"esp_{dataset_name}_{str(percent_lc[i])}.png", dpi=600)
            plt.close()


    plt.figure()
    plt.semilogy(100.0*np.array(percent_lc), our_full_costs, label='MAC Nearest (Ours)')
    plt.semilogy(100.0*np.array(percent_lc), madow_full_costs, label='MAC Madow (Ours)', linestyle='-.', marker='x', color='C5')
    plt.semilogy(100.0*np.array(percent_lc), naive_full_costs, label='Naive Method', color='red', linestyle='-.')
    if run_greedy:
        plt.semilogy(100.0*np.array(percent_lc), esp_full_costs, label='Greedy ESP', color='orange', linestyle='-')
    plt.xlabel(r'\% Edges Added')
    plt.ylabel(r'Objective Value')
    plt.legend()
    plt.savefig(f"obj_val_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(100.0*np.array(percent_lc), our_SOd_orbdists, label='MAC Nearest (Ours)')
    plt.plot(100.0*np.array(percent_lc), madow_SOd_orbdists, label='MAC Madow (Ours)', linestyle='-.', marker='x', color='C5')
    plt.plot(100.0*np.array(percent_lc), naive_SOd_orbdists, label='Naive Method', color='red', linestyle='-.')
    if run_greedy:
        plt.plot(100.0*np.array(percent_lc), esp_SOd_orbdists, label='Greedy ESP', color='orange', linestyle='-')
    plt.ylabel(r'$\mathrm{SO}(d)$ orbit distance')
    plt.xlabel(r'\% Edges Added')
    plt.legend()
    plt.savefig(f"orbdist_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.show()

    # Extract translational states from solution xhat
    xhat = sesync_result.xhat
    print(xhat.shape)
    R0inv = np.linalg.inv(xhat[:, num_poses : num_poses + 2])
    t = np.matmul(R0inv, xhat[:, 0:num_poses])

    print("Our rotation cost: ", evaluate_sesync_rotation_objective(LGrho, xhat[:, num_poses:]))
    print("Our cost: ", evaluate_sesync_objective(M, xhat))

    plot_poses(xhat, our_meas)

    xhat = sesync_result_naive.xhat
    R0inv = np.linalg.inv(xhat[:, num_poses : num_poses + 2])
    t = np.matmul(R0inv, xhat[:, 0:num_poses])

    print("Naive rotation cost: ", evaluate_sesync_rotation_objective(LGrho, xhat[:, num_poses:]))
    print("Naive cost: ", evaluate_sesync_objective(M, xhat))


