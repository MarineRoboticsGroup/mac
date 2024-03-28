import sys
import math
from tqdm import tqdm
import numpy as np
from imac_pgo import iPGO
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from pose_graph_utils import read_g2o_file, plot_poses, rpm_to_mac, RelativePoseMeasurement

from mac.utils import split_edges, Edge

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 16})

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

    # Split measurements into odom and loop closures
    odom_measurements, lc_measurements = split_edges(measurements)

    # Get problem dimension
    d = odom_measurements[0].R.shape[0]

    # Print dataset stats
    print(f"Loaded {len(measurements)} total measurements with: ")
    print(f"\t {len(odom_measurements)} base (odometry) measurements and")
    print(f"\t {len(lc_measurements)} candidate (loop closure) measurements")

    #############################
    # Running the tests!
    #############################

    # Set up a collection of iPGO instances
    ipgo_instances = [
        ("full", iPGO(budget=(lambda n: 9223372036854775807), d=d, rounding="nearest")), # Hack to ensure this guy never sparsifies
        ("fixed_10_nearest", iPGO(budget=(lambda n: int(0.10*num_poses)), d=d, rounding="nearest")),
        ("fixed_10_nearest_cache", iPGO(budget=(lambda n: int(0.10*num_poses)), d=d, rounding="nearest", use_cache=True))]
        # ("linear_10_nearest", iPGO(budget=(lambda n: int(0.1*n)), d=d, rounding="nearest")),
        # ("linear_10_nearest_cache", iPGO(budget=(lambda n: int(0.1*n)), d=d, rounding="nearest", use_cache=True))]
        # ("fixed_30_nearest", iPGO(budget=(lambda n: int(0.3*num_poses)), d=d, rounding="nearest")),
        # ("linear_30_nearest", iPGO(budget=(lambda n: int(0.3*n)), d=d, rounding="nearest"))]

    plot_every = int(num_poses / num_poses)
    pose_idx = []
    l2s = []
    times = []
    sync_times = []
    for i in tqdm(range(num_poses)):
        # Get odom edges and add as fixed edges
        odom_meas = [e for e in odom_measurements if e.i == i]

        # Add odom measurements to all the iPGO instances
        for name, ipgo in ipgo_instances:
            ipgo.add_odom_measurements(odom_meas)

        # Add the set of loop closures from the current pose to previous poses
        lc_meas = [e for e in lc_measurements if e.i == i and e.j < i]
        # Add the set of loop closures to the current pose from a previous pose
        lc_meas += [e for e in lc_measurements if e.j == i and e.i < i]

        # Add the loop closures to all iPGO instances
        for name, ipgo in ipgo_instances:
            ipgo.add_lc_measurements(lc_meas, sparsify=False)
        if i % plot_every == 0 or i == num_poses - 1:
            l2s.append({})
            times.append({})
            sync_times.append({})
            # Sparsify the graph
            if len(lc_meas) > 0:
                for name, ipgo in ipgo_instances:
                    start = timer()
                    ipgo.sparsify()
                    end = timer()
                    times[i][name] = end - start
                    res = ipgo.solve()
                    sync_times[i][name] = res.total_computation_time
                    pass
                pass
            else:
                for name, ipgo in ipgo_instances:
                    times[i][name] = 0.0
                    sync_times[i][name] = 0.0
            # Solve the graph
            # res_log = ipgo_log.solve()

            # Get all selected measurements from iPGO
            # all_meas = ipgo.fixed_meas + ipgo.candidate_meas
            # plot_poses(res.xhat, all_meas, show=True)

            # Store some results
            pose_idx.append(i)
            for name, ipgo in ipgo_instances:
                l2s[i][name] = ipgo.imac.evaluate_objective(np.ones(len(ipgo.candidate_meas)))
            # Get first instance
            tmp_ipgo = ipgo_instances[0][1]
            l2s[i]["odom"] = tmp_ipgo.imac.evaluate_objective(np.zeros(len(tmp_ipgo.candidate_meas)))
            pass
        pass

    l2s_dense = np.array([l2["full"] for l2 in l2s])
    for name, instance in ipgo_instances:
        if name != "full":
            l2s_sparse = np.array([l2[name] for l2 in l2s])
            plt.plot(pose_idx, l2s_sparse / l2s_dense, label=name)
        pass
    # l2s_odom = [l2["odom"] for l2 in l2s]
    # plt.plot(pose_idx, l2s_odom / l2s_dense, label="odom")
    plt.ylabel(r'Algebraic Connectivity $\lambda_2 / \lambda_2(\mathcal{G})$')
    plt.xlabel('Pose')
    plt.legend()
    plt.show()

    ## Plot times
    for name, instance in ipgo_instances:
        if name != "full":
            times_sparse = np.array([time[name] for time in times])
            plt.plot(pose_idx, np.cumsum(times_sparse), label=name)
        pass
    plt.ylabel(r'Cumulative Sparsification Time (s)')
    plt.xlabel('Pose')
    plt.legend()
    plt.show()

    ## Plot solve times
    for name, instance in ipgo_instances:
        times_sparse = np.array([time[name] for time in sync_times])
        plt.plot(pose_idx, np.cumsum(times_sparse), label=name)
        pass


    plt.ylabel(r'Cumulative Solve Time (s)')
    plt.xlabel('Pose')
    plt.legend()
    plt.show()

    for name, instance in ipgo_instances:
        times_solve = np.array([time[name] for time in sync_times])
        if name == "full":
            plt.plot(pose_idx, np.cumsum(times_solve), label=name)
        if name != "full":
            times_sparse = np.array([time[name] for time in times])
            times_total = times_sparse + times_solve
            plt.plot(pose_idx, np.cumsum(times_total), label=name)
        pass

    plt.ylabel(r'Cumulative Total Time (s)')
    plt.xlabel('Pose')
    plt.legend()
    plt.show()
    # for meas in measurements:
    #     ipgo.add_edge(meas)

