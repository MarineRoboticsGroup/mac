import sys
import numpy as np
from utils import read_g2o_file, split_measurements, plot_poses
from timeit import default_timer as timer
from fwac import FWAC
from naive_greedy import NaiveGreedy
from wafr_greedy import GreedyTree
import networkx as nx
import random

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 16})

# SE-Sync setup
sesync_lib_path = "/Users/kevin/repos/SESync/C++/build/lib"
sys.path.insert(0, sesync_lib_path)

import PySESync

def orbit_distance_dS(X, Y, compute_G_S=False):
    """
    Compute SO(d) orbit distance between X and Y
    if compute_G_S = True, also returns optimal registration
    """
    d = X.shape[0]
    n = int(X.shape[1] / d)

    XYt = X @ Y.T

    u, s, vh = np.linalg.svd(XYt)

    uvh = u @ vh
    Sigma = np.diag(s)

    Xi_diag = np.diag(np.ones(d))
    Xi_diag[d-1, d-1] = np.copysign(1.0, np.linalg.det(uvh))

    dS = np.sqrt(np.abs(
        2 * d * n - 2 * np.trace(Xi_diag @ Sigma)
        ))

    if compute_G_S:
        G_S = u @ Xi_diag @ vh
        return dS, G_S
    return dS

def construct_LGrho(measurements):
    d = len(measurements[0].t) if len(measurements) > 0 else 0

    # Scan through the measurements to determine the total number of poses
    num_poses = 0
    for measurement in measurements:
        max_pair = max(measurement.i, measurement.j)
        if max_pair > num_poses:
            num_poses = max_pair
    num_poses = num_poses + 1

    # Build empty rotation graph Laplacian matrix
    LGrho = np.zeros([d * num_poses, d * num_poses])

    # Now scan through measurements again and populate M
    for measurement in measurements:
        i = measurement.i
        j = measurement.j

        # Add elements for L(G^rho)
        # Elements of ith block-diagonal
        for k in range(0, d):
            LGrho[d * i + k, d * i + k] += measurement.kappa

        # Element of jth block-diagonal
        for k in range(0, d):
            LGrho[d * j + k, d * j + k] += measurement.kappa

        # Elements of ij block
        for r in range(0, d):
            for c in range(0, d):
                LGrho[i * d + r, j * d + c] += -measurement.kappa * measurement.R[r, c]

        # Elements of ji block
        for r in range(0, d):
            for c in range(0, d):
                LGrho[j * d + r, i * d + c] += -measurement.kappa * measurement.R[c, r]

    return LGrho

def evaluate_sesync_rotation_objective(LGrho, R):
    return np.trace(R @ LGrho @ R.T)

def construct_sesync_quadratic_form_matrix(measurements):
    """
    This function emulates the behavior of the `construct_quadratic_form_matrix`
    function in SE-Sync. It builds the matrix M for the SE(d) synchronization
    problem defined as

    min [t; vec(R)]' (M \otimes I_d) [t; vec(R)]

    Note: This is the "translation explicit" version of the problem (i.e.
    without translations marginalized out).

    """

    d = len(measurements[0].t) if len(measurements) > 0 else 0

    # Scan through the measurements to determine the total number of poses
    num_poses = 0
    for measurement in measurements:
        max_pair = max(measurement.i, measurement.j)
        if max_pair > num_poses:
            num_poses = max_pair
    num_poses = num_poses + 1

    # Build empty quadratic form matrix
    M = np.zeros([(d + 1) * num_poses, (d + 1) * num_poses])

    # Now scan through measurements again and populate M
    for measurement in measurements:
        i = measurement.i
        j = measurement.j

        # Add elements for L(W^tau)
        M[i, i] += measurement.tau
        M[j, j] += measurement.tau
        M[i, j] += -measurement.tau
        M[j, i] += -measurement.tau

        # Add elements for V (upper-right block)
        for k in range(0, d):
            M[i, num_poses + i * d + k] += measurement.tau * measurement.t[k]
        for k in range(0, d):
            M[j, num_poses + i * d + k] += -measurement.tau * measurement.t[k]

        # Add elements for V' (lower-left block)
        for k in range(0, d):
            M[num_poses + i * d + k, i] += measurement.tau * measurement.t[k]
        for k in range(0, d):
            M[num_poses + i * d + k, j] += -measurement.tau * measurement.t[k]

        # Add elements for L(G^rho)
        # Elements of ith block-diagonal
        for k in range(0, d):
            M[num_poses + d * i + k,
              num_poses + d * i + k] += measurement.kappa

        # Element of jth block-diagonal
        for k in range(0, d):
            M[num_poses + d * j + k,
              num_poses + d * j + k] += measurement.kappa

        # Elements of ij block
        for r in range(0, d):
            for c in range(0, d):
                M[num_poses + i * d + r, num_poses + j * d +
                  c] += -measurement.kappa * measurement.R[r, c]

        # Elements of ji block
        for r in range(0, d):
            for c in range(0, d):
                M[num_poses + j * d + r, num_poses + i * d +
                  c] += -measurement.kappa * measurement.R[c, r]

        # Add elements for Sigma
        for r in range(0, d):
            for c in range(0, d):
                M[num_poses + i * d + r, num_poses + i * d +
                  c] += measurement.tau * measurement.t[r] * measurement.t[c]

    return M

def evaluate_sesync_objective(M, Xhat):
    return np.trace(Xhat @ M @ Xhat.T)


def nx_rot_G_w(measurements, num_poses):
    G = nx.generators.classic.empty_graph(num_poses)
    for measurement in measurements:
        i = measurement.i
        j = measurement.j
        w = measurement.kappa
        G.add_edge(i, j, weight=w)
    return G

def nx_tran_G_w(measurements, num_poses):
    G = nx.generators.classic.empty_graph(num_poses)
    for measurement in measurements:
        i = measurement.i
        j = measurement.j
        w = measurement.tau
        G.add_edge(i, j, weight=w)
    return G

def select_measurements(measurements, w):
    assert(len(measurements) == len(w))
    meas_out = []
    for i, meas in enumerate(measurements):
        if w[i] == 1.0:
            meas_out.append(meas)
    return meas_out

def to_sesync_format(measurements):
    sesync_measurements = []
    for meas in measurements:
        sesync_meas = PySESync.RelativePoseMeasurement()
        sesync_meas.i = meas.i
        sesync_meas.j = meas.j
        sesync_meas.kappa = meas.kappa
        sesync_meas.tau = meas.tau
        sesync_meas.R = meas.R
        sesync_meas.t = meas.t
        sesync_measurements.append(sesync_meas)
    return sesync_measurements

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [.g2o file]")
        sys.exit()

    dataset_name = sys.argv[1].split('/')[-1].split('.')[0]
    print(f"Loading dataset: {dataset_name}")

    # Load a g2o file
    print("Reading g2o file")
    start = timer()
    measurements, num_poses = read_g2o_file(sys.argv[1])
    end = timer()
    print("Success! elapsed time: ", (end - start))

    # Set random seed so trials are deterministic
    # random.seed(7)

    # Split measurements into odom and loop closures
    odom_measurements, lc_measurements = split_measurements(measurements)

    # Randomly shuffle lc_measurements to prevent taking advantage of order in g2o file.
    # random.shuffle(lc_measurements)

    # We use networkx to compute graph Laplacians, though using scipy directly
    # would probably be faster
    # G_odom = nx_rot_G_w(odom_measurements, num_poses)
    G_lc = nx_rot_G_w(lc_measurements, num_poses)


    # G_odom should have a single connected component
    # print([len(c) for c in sorted(nx.connected_components(G_odom), key=len, reverse=True)])

    # Print dataset stats
    print(f"Loaded {len(measurements)} total measurements with: ")
    print(f"\t {len(odom_measurements)} base (odometry) measurements and")
    print(f"\t {len(lc_measurements)} candidate (loop closure) measurements")

    # Make a FWAC Solver
    fwac = FWAC(odom_measurements, lc_measurements, num_poses)

    # Make a Naive Solver
    greedy = NaiveGreedy(G_lc)

    #############################
    # Running the tests!
    #############################

    # Test between 100% and 0% loop closures
    # percent_lc = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    percent_lc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # percent_lc = np.linspace(0.01, 1.0, num=30, endpoint=True)
    # percent_lc = [0.0]
    # percent_lc = [1.0, 0.9]

    # Containers for results
    results = []
    unrounded_results = []
    upper_bounds = []
    greedy_results = []
    random_results = []
    times = []

    for pct_lc in percent_lc:
        num_lc = int(pct_lc * len(lc_measurements))
        print("Num LC to accept: ", num_lc)

        # Compute a solution using the naive method. This serves both as a
        # baseline and as a sparse initializer for our method.
        greedy_result = greedy.subset(num_lc)
        greedy_results.append(greedy_result)

        # Alternatively (k/m) in each slot
        # w_init = (num_lc / len(lc_measurements)) * np.ones(len(lc_measurements))

        w_init = greedy_result
        # print(w_init)

        # Solve the relaxed maximum algebraic connectivity augmentation problem.
        start = timer()
        result, unrounded, upper = fwac.fw_subset(w_init, num_lc, max_iters=20)
        end = timer()
        times.append(end - start)
        # print(result)
        results.append(result)
        upper_bounds.append(upper)
        unrounded_results.append(unrounded)

    # Display the algebraic connectivity for each method
    for i in range(len(greedy_results)):
        pct_lc = percent_lc[i]
        print(f"Greedy AC at {pct_lc * 100.0} % loop closures: {fwac.evaluate_objective(greedy_results[i])}")
        print(f"Our AC at {pct_lc * 100.0} % loop closures: {fwac.evaluate_objective(results[i])}")
        print(f"Our unrounded AC at {pct_lc * 100.0} % loop closures: {fwac.evaluate_objective(unrounded_results[i])}")
        print(f"Dual at {pct_lc * 100.0} % loop closures: {upper_bounds[i]}")

    #############################
    # Plot the Results
    #############################

    # plot connectivity vs. percent_lc
    our_objective_vals = [fwac.evaluate_objective(result) for result in results]
    naive_objective_vals = [fwac.evaluate_objective(greedy_result) for greedy_result in greedy_results]
    unrounded_objective_vals = [fwac.evaluate_objective(unrounded) for unrounded in unrounded_results]

    plt.plot(100.0*np.array(percent_lc), our_objective_vals, label='Ours')
    plt.plot(100.0*np.array(percent_lc), upper_bounds, label='Dual Upper Bound', linestyle='--', color='C0')
    plt.fill_between(100.0*np.array(percent_lc), our_objective_vals, upper_bounds, alpha=0.2, label='Suboptimality Gap')
    plt.plot(100.0*np.array(percent_lc), unrounded_objective_vals, label='Unrounded', c='C2')
    plt.plot(100.0*np.array(percent_lc), naive_objective_vals, label='Naive Method', color='red', linestyle='-.')
    plt.ylabel(r'Algebraic Connectivity $\lambda_2$')
    plt.xlabel(r'\% Edges Added')
    plt.legend()
    plt.savefig(f"alg_conn_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(100.0*np.array(percent_lc), times)
    plt.xlim([0.0, 90.0])
    plt.ylabel(r'Time (s)')
    plt.xlabel(r'\% Edges Added')
    plt.savefig(f"comp_time_{dataset_name}.png", dpi=600, bbox_inches='tight')
    # plt.title('Computation times')
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
    sesync_greedy = []
    for i in range(len(greedy_results)):
        our_selected_lc = select_measurements(lc_measurements, results[i])
        our_meas = odom_measurements + our_selected_lc
        sesync_our_meas = to_sesync_format(our_meas)
        sesync_result = PySESync.SESync(sesync_our_meas, opts)
        sesync_results.append(sesync_result)
        greedy_selected_lc = select_measurements(lc_measurements, greedy_results[i])
        greedy_meas = odom_measurements + greedy_selected_lc
        sesync_result_greedy = PySESync.SESync(to_sesync_format(greedy_meas), opts)
        sesync_greedy.append(sesync_result_greedy)

    plt.figure()
    plt.plot(100.0*np.array(percent_lc), [res.total_computation_time for res in sesync_results], label='Ours')
    plt.plot(100.0*np.array(percent_lc), [res.total_computation_time for res in sesync_greedy], label='Naive Method', color='red', linestyle='-.')
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

    naive_rot_costs = []
    naive_full_costs = []
    naive_SOd_orbdists = []
    for i in range(len(percent_lc)):
        print(f"Percent LC: {percent_lc[i]}")

        xhat_ours = sesync_results[i].xhat
        xhat_naive = sesync_greedy[i].xhat

        our_selected_lc = select_measurements(lc_measurements, results[i])
        our_meas = odom_measurements + our_selected_lc

        greedy_selected_lc = select_measurements(lc_measurements, greedy_results[i])
        greedy_meas = odom_measurements + greedy_selected_lc

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

        naive_rot_cost = evaluate_sesync_rotation_objective(LGrho, xhat_naive[:, num_poses:])
        naive_full_cost = evaluate_sesync_objective(M, xhat_naive)
        naive_SOd_orbdist = orbit_distance_dS(sesync_full.xhat[:,num_poses:], xhat_naive[:,num_poses:])

        naive_rot_costs.append(naive_rot_cost)
        naive_full_costs.append(naive_full_cost)
        naive_SOd_orbdists.append(naive_SOd_orbdist)

        plot_poses(xhat_naive, greedy_meas, show=False)
        plt.savefig(f"naive_{dataset_name}_{str(percent_lc[i])}.png", dpi=600)
        plt.close()

        # Print Error for naive method
        print(f"Naive rotation cost: {naive_rot_cost}")
        print(f"Naive full cost: {naive_full_cost}")
        print(f"Naive SO orbdist: {naive_SOd_orbdist}")

    # plt.subplot(131)
    # plt.title('Rotation Cost')
    # plt.legend()


    plt.figure()
    plt.semilogy(100.0*np.array(percent_lc), our_full_costs, label='Ours')
    plt.semilogy(100.0*np.array(percent_lc), naive_full_costs, label='Naive Method', color='red', linestyle='-.')
    plt.xlabel(r'\% Edges Added')
    plt.ylabel(r'Objective Value')
    plt.legend()
    plt.savefig(f"obj_val_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(100.0*np.array(percent_lc), our_SOd_orbdists, label='Ours')
    plt.plot(100.0*np.array(percent_lc), naive_SOd_orbdists, label='Naive Method', color='red', linestyle='-.')
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

    xhat = sesync_result_greedy.xhat
    R0inv = np.linalg.inv(xhat[:, num_poses : num_poses + 2])
    t = np.matmul(R0inv, xhat[:, 0:num_poses])

    print("Naive rotation cost: ", evaluate_sesync_rotation_objective(LGrho, xhat[:, num_poses:]))
    print("Naive cost: ", evaluate_sesync_objective(M, xhat))

    # print("Greedy orbit: ", evaluate_sesync_objective(M, xhat))

